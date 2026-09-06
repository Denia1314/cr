from __future__ import annotations

"""Small, deterministic temporal helpers used by the M2 perception loop.

The helpers in this module deliberately do not infer a card or a unit from a
single stale frame.  They keep bounded state, expose age/confidence, and are
safe to copy as part of the policy action snapshot.
"""

from dataclasses import dataclass
from typing import Iterable

from .battle_perception import HandCardMatch


@dataclass
class HandSlotState:
    slot_index: int
    card_id: str | None = None
    confidence: float = 0.0
    observed_at: float = -1_000.0
    stable_frames: int = 0
    empty: bool = False
    blocked_card_id: str | None = None

    def age(self, now: float) -> float:
        if self.observed_at < -999.0:
            return float("inf")
        return max(0.0, float(now) - self.observed_at)

    def usable_card_id(self, now: float, max_age_s: float, min_confidence: float) -> str | None:
        if self.card_id is None or self.blocked_card_id == self.card_id:
            return None
        if self.age(now) > max_age_s or self.confidence < min_confidence:
            return None
        return self.card_id


class HandHistory:
    """Bounded per-slot hand memory with stale and post-action guards.

    A transient empty/unknown result is retained only for a short grace
    period, with decayed confidence.  A confirmed action blocks the old card
    until a different card is observed, so a delayed screenshot cannot cause
    the same slot to be played again.
    """

    def __init__(
        self,
        *,
        max_age_s: float = 1.2,
        min_confidence: float = 0.40,
        unknown_grace_s: float = 0.45,
        empty_grace_s: float = 0.32,
        confidence_decay_s: float = 1.2,
    ) -> None:
        self.max_age_s = max(0.1, float(max_age_s))
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self.unknown_grace_s = max(0.0, float(unknown_grace_s))
        self.empty_grace_s = max(0.0, float(empty_grace_s))
        self.confidence_decay_s = max(0.1, float(confidence_decay_s))
        self.slots: dict[int, HandSlotState] = {}
        self.last_update_at = -1_000.0

    def reset(self) -> None:
        self.slots = {}
        self.last_update_at = -1_000.0

    @staticmethod
    def _safe_confidence(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _decayed_confidence(self, state: HandSlotState, now: float) -> float:
        age = state.age(now)
        if age <= 0.0:
            return state.confidence
        return self._safe_confidence(
            state.confidence * max(0.0, 1.0 - age / self.confidence_decay_s)
        )

    def update(self, matches: Iterable[HandCardMatch], now: float) -> None:
        now = float(now)
        seen_slots: set[int] = set()
        for match in matches:
            slot = int(match.slot_index)
            seen_slots.add(slot)
            previous = self.slots.get(slot)
            recognized = match.card_id is not None and not match.empty
            if recognized:
                card_id = str(match.card_id)
                if previous is not None and previous.blocked_card_id == card_id:
                    # The card has been confirmed as used.  Keep the block
                    # while the game is still showing the pre-replacement
                    # frame; a different card will release it below.
                    previous.observed_at = now
                    previous.confidence = self._safe_confidence(match.confidence)
                    previous.empty = False
                    continue
                if previous is not None and previous.card_id == card_id:
                    stable_frames = previous.stable_frames + 1
                    confidence = max(previous.confidence, self._safe_confidence(match.confidence))
                    blocked = previous.blocked_card_id
                else:
                    stable_frames = 1
                    confidence = self._safe_confidence(match.confidence)
                    blocked = None
                self.slots[slot] = HandSlotState(
                    slot_index=slot,
                    card_id=card_id,
                    confidence=confidence,
                    observed_at=now,
                    stable_frames=stable_frames,
                    empty=False,
                    blocked_card_id=blocked,
                )
                continue

            # Do not erase a card on one animation/occlusion frame.  The
            # positive observation timestamp stays unchanged, which means it
            # still expires according to max_age_s.
            if previous is not None and previous.card_id is not None:
                grace = self.empty_grace_s if match.empty else self.unknown_grace_s
                if previous.age(now) <= grace and previous.blocked_card_id is None:
                    previous.confidence = self._decayed_confidence(previous, now) * 0.85
                    previous.empty = bool(match.empty)
                    continue
            self.slots[slot] = HandSlotState(
                slot_index=slot,
                card_id=None,
                confidence=self._safe_confidence(match.confidence),
                observed_at=now,
                stable_frames=0,
                empty=bool(match.empty),
                blocked_card_id=(previous.blocked_card_id if previous else None),
            )

        # Missing slots are treated as unknown, not as a new empty card.  They
        # can still be returned as stale metadata but never as usable cards.
        for slot, state in self.slots.items():
            if slot not in seen_slots and state.age(now) > self.max_age_s:
                state.card_id = None
                state.confidence = 0.0
                state.empty = False
        self.last_update_at = now

    def consume(self, slot_index: int, card_id: str | None, now: float) -> None:
        state = self.slots.get(int(slot_index))
        if state is None:
            state = HandSlotState(slot_index=int(slot_index))
            self.slots[int(slot_index)] = state
        state.blocked_card_id = str(card_id) if card_id else state.card_id
        state.card_id = state.blocked_card_id
        state.observed_at = float(now)
        state.confidence = max(state.confidence, self.min_confidence)
        state.empty = False

    def matches_for_decision(
        self,
        observed: Iterable[HandCardMatch],
        now: float,
    ) -> list[HandCardMatch]:
        """Return observed slots with bounded, age-aware card identities."""
        result: list[HandCardMatch] = []
        for match in observed:
            state = self.slots.get(int(match.slot_index))
            card_id = (
                state.usable_card_id(now, self.max_age_s, self.min_confidence)
                if state is not None
                else None
            )
            confidence = (
                self._decayed_confidence(state, now)
                if state is not None and card_id is not None
                else self._safe_confidence(match.confidence)
            )
            if card_id is None:
                confidence = min(confidence, 0.0 if state is not None and state.age(now) > self.max_age_s else 0.49)
            result.append(
                HandCardMatch(
                    slot_index=match.slot_index,
                    card_id=card_id,
                    confidence=round(confidence, 4),
                    good_matches=match.good_matches,
                    second_good_matches=match.second_good_matches,
                    keypoints=match.keypoints,
                    empty=match.empty,
                )
            )
        return result

    def metadata(self, now: float) -> dict[str, dict[str, float | int | str | None]]:
        return {
            str(slot): {
                "card_id": state.usable_card_id(now, self.max_age_s, self.min_confidence),
                "age_s": round(state.age(now), 3),
                "confidence": round(self._decayed_confidence(state, now), 4),
                "stable_frames": state.stable_frames,
                "blocked_card_id": state.blocked_card_id,
            }
            for slot, state in sorted(self.slots.items())
        }


class TimingStats:
    """Bounded stage timing samples with reproducible percentile summaries."""

    def __init__(self, *, max_samples: int = 512) -> None:
        self.max_samples = max(16, int(max_samples))
        self.samples: dict[str, list[float]] = {}

    def reset(self) -> None:
        self.samples = {}

    def record(self, stage: str, duration_s: float) -> None:
        key = str(stage).strip() or "unknown"
        values = self.samples.setdefault(key, [])
        values.append(max(0.0, float(duration_s)))
        if len(values) > self.max_samples:
            del values[: len(values) - self.max_samples]

    @staticmethod
    def percentile(values: Iterable[float], percentile: float) -> float | None:
        ordered = sorted(max(0.0, float(value)) for value in values)
        if not ordered:
            return None
        if len(ordered) == 1:
            return round(ordered[0], 6)
        rank = max(0.0, min(1.0, float(percentile))) * (len(ordered) - 1)
        lower = int(rank)
        upper = min(len(ordered) - 1, lower + 1)
        fraction = rank - lower
        value = ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
        return round(value, 6)

    def summary(self) -> dict[str, dict[str, float | int | None]]:
        return {
            stage: {
                "count": len(values),
                "p50_s": self.percentile(values, 0.50),
                "p95_s": self.percentile(values, 0.95),
            }
            for stage, values in sorted(self.samples.items())
        }

