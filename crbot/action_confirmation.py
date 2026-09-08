"""Confirmation of card actions from post-click game evidence.

ADB accepting a tap is not the same thing as the game accepting a card.  This
module deliberately keeps confirmation independent from the policy so it can
be replayed in tests and used by both live execution and diagnostics.
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from PIL import Image

from .battle_perception import HandCardMatch
from .vision import crop_normalized, patch_similarity


ACTION_STATUSES = frozenset({"proposed", "sent", "confirmed", "rejected", "unknown"})


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value))


def _match_at(matches: Iterable[HandCardMatch] | None, slot_index: int) -> HandCardMatch | None:
    if matches is None:
        return None
    for match in matches:
        if int(match.slot_index) == int(slot_index):
            return match
    return None


@dataclass(frozen=True)
class ActionConfirmation:
    action_id: str
    status: str
    confidence: float
    reason: str
    evidence: dict[str, Any]
    elapsed_s: float = 0.0
    observed_frames: int = 0
    failure_code: str | None = None

    def __post_init__(self) -> None:
        if self.status not in ACTION_STATUSES:
            raise ValueError(f"未知动作状态：{self.status}")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["confidence"] = round(float(self.confidence), 6)
        value["elapsed_s"] = round(float(self.elapsed_s), 6)
        return value


def assess_action_evidence(
    *,
    pre_image: Image.Image | None,
    post_image: Image.Image | None,
    pre_matches: Iterable[HandCardMatch] | None,
    post_matches: Iterable[HandCardMatch] | None,
    slot_index: int,
    pre_elixir: float | None,
    post_elixir: float | None,
    card_cost: float | None,
    slot_center: list[float] | None = None,
    vision_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return independent evidence for one observation.

    A single empty/uncertain slot is never enough.  Confirmation requires at
    least two independent signals, normally a stable hand replacement plus a
    fee decrease or a visible slot change.
    """
    before = _match_at(pre_matches, slot_index)
    after = _match_at(post_matches, slot_index)
    before_id = before.card_id if before is not None else None
    after_id = after.card_id if after is not None else None
    before_conf = float(before.confidence) if before is not None else 0.0
    after_conf = float(after.confidence) if after is not None else 0.0

    hand_change = bool(
        before_id
        and (
            (after_id is not None and after_id != before_id and after_conf >= 0.40)
            or (after_id is None and after is not None and after.empty)
        )
    )
    # The recognizer may temporarily fail while the replacement card is still
    # animating.  Keep this as a weaker signal; the second signal is mandatory.
    hand_uncertain_change = bool(
        before_id
        and after is not None
        and after_id is None
        and not after.empty
        and after_conf < 0.40
    )

    elixir_drop = 0.0
    if _finite(pre_elixir) and _finite(post_elixir):
        elixir_drop = float(pre_elixir) - float(post_elixir)
    expected_cost = max(1.0, float(card_cost or 3.0))
    min_drop = max(0.55, min(1.35, expected_cost * 0.32))
    elixir_change = bool(elixir_drop >= min_drop)

    visual_delta = 0.0
    visual_change = False
    if pre_image is not None and post_image is not None and slot_center and vision_config:
        half_width = float(vision_config.get("card_roi_half_width", 0.09))
        top = float(vision_config.get("card_roi_top", 0.825))
        bottom = float(vision_config.get("card_roi_bottom", 0.955))
        center_x = float(slot_center[0])
        roi = [center_x - half_width, top, center_x + half_width, bottom]
        visual_delta = max(
            0.0,
            1.0 - patch_similarity(
                crop_normalized(pre_image, roi), crop_normalized(post_image, roi)
            ),
        )
        visual_change = visual_delta >= float(vision_config.get("action_visual_delta", 0.055))

    independent_signals = int(hand_change) + int(elixir_change) + int(visual_change)
    strong_hand_signal = hand_change and after_conf >= 0.55
    confidence = min(
        1.0,
        0.35 * independent_signals
        + (0.20 if strong_hand_signal else 0.0)
        + (min(0.20, elixir_drop / 10.0) if elixir_change else 0.0)
        + (min(0.20, visual_delta) if visual_change else 0.0),
    )
    return {
        "hand_change": hand_change,
        "hand_uncertain_change": hand_uncertain_change,
        "elixir_change": elixir_change,
        "elixir_drop": round(elixir_drop, 4),
        "visual_change": visual_change,
        "visual_delta": round(visual_delta, 4),
        "independent_signals": independent_signals,
        "before_card_id": before_id,
        "after_card_id": after_id,
        "before_confidence": round(before_conf, 4),
        "after_confidence": round(after_conf, 4),
        "confidence": round(confidence, 4),
        "frame_fingerprint": (
            hashlib.sha256(post_image.resize((16, 16)).convert("L").tobytes()).hexdigest()[:16]
            if post_image is not None
            else None
        ),
    }


class ActionConfirmationTracker:
    """Bounded, idempotent state machine for live action attempts."""

    def __init__(
        self,
        *,
        timeout_s: float = 1.35,
        poll_interval_s: float = 0.18,
        stable_frames: int = 2,
    ) -> None:
        self.timeout_s = max(0.2, float(timeout_s))
        self.poll_interval_s = max(0.03, float(poll_interval_s))
        self.stable_frames = max(1, int(stable_frames))
        self._pending: dict[str, dict[str, Any]] = {}
        self._resolved: dict[str, ActionConfirmation] = {}
        self._outcome_counts: dict[str, int] = {}
        self._failure_counts: dict[str, int] = {}
        self._elapsed_samples: list[float] = []
        self._frame_samples: list[int] = []

    @staticmethod
    def _new_pending_state(sent_at: float) -> dict[str, Any]:
        return {
            "sent_at": float(sent_at),
            "frames": 0,
            "positive_frames": 0,
            "last_signature": None,
            "last_evidence": {},
            "best_evidence": {},
            "partial_seen": False,
            "observed_signals": set(),
            "last_frame_fingerprint": None,
            "duplicate_frames": 0,
        }

    @staticmethod
    def _has_partial_evidence(evidence: dict[str, Any]) -> bool:
        return bool(
            int(evidence.get("independent_signals", 0)) > 0
            or evidence.get("hand_change")
            or evidence.get("hand_uncertain_change")
            or evidence.get("elixir_change")
            or evidence.get("visual_change")
        )

    @staticmethod
    def _evidence_rank(evidence: dict[str, Any]) -> tuple[int, int, float]:
        partial_count = sum(
            int(bool(evidence.get(key)))
            for key in (
                "hand_change",
                "hand_uncertain_change",
                "elixir_change",
                "visual_change",
            )
        )
        return (
            int(evidence.get("independent_signals", 0)),
            partial_count,
            float(evidence.get("confidence", 0.0)),
        )

    def _remember_evidence(
        self,
        state: dict[str, Any],
        evidence: dict[str, Any],
    ) -> None:
        observation = dict(evidence)
        state["last_evidence"] = observation
        state["partial_seen"] = bool(
            state.get("partial_seen") or self._has_partial_evidence(observation)
        )
        observed_signals = state.setdefault("observed_signals", set())
        for key in (
            "hand_change",
            "hand_uncertain_change",
            "elixir_change",
            "visual_change",
        ):
            if observation.get(key):
                observed_signals.add(key)
        best = dict(state.get("best_evidence") or {})
        if not best or self._evidence_rank(observation) > self._evidence_rank(best):
            state["best_evidence"] = observation

    @staticmethod
    def _preserved_evidence(state: dict[str, Any]) -> dict[str, Any]:
        evidence = dict(state.get("best_evidence") or state.get("last_evidence") or {})
        observed_signals = sorted(
            str(value) for value in state.get("observed_signals", set())
        )
        evidence.update(
            {
                "evidence_frames": int(state.get("frames", 0)),
                "partial_evidence_seen": bool(state.get("partial_seen", False)),
                "best_independent_signals": int(
                    (state.get("best_evidence") or {}).get("independent_signals", 0)
                ),
                "observed_signals": observed_signals,
                "cumulative_independent_signals": sum(
                    signal in observed_signals
                    for signal in ("hand_change", "elixir_change", "visual_change")
                ),
                "duplicate_frames": int(state.get("duplicate_frames", 0)),
            }
        )
        return evidence

    def _finish_timeout(
        self,
        action_id: str,
        state: dict[str, Any],
        current: float,
    ) -> ActionConfirmation:
        evidence = self._preserved_evidence(state)
        has_partial = bool(
            state.get("partial_seen") or int(state.get("positive_frames", 0)) > 0
        )
        status = "unknown" if has_partial else "rejected"
        failure_code = self._failure_code(state, status=status)
        reason = (
            "超时但曾观察到部分或不稳定证据，保留为未知"
            if status == "unknown"
            else "超时且未观察到手牌、费用或卡槽变化"
        )
        return self._finish(
            action_id,
            ActionConfirmation(
                action_id,
                status,
                max(0.0, min(1.0, float(evidence.get("confidence", 0.0)))),
                reason,
                evidence,
                max(0.0, current - float(state["sent_at"])),
                int(state.get("frames", 0)),
                failure_code,
            ),
        )

    @staticmethod
    def _failure_code(state: dict[str, Any], *, status: str) -> str | None:
        if status == "confirmed":
            return None
        evidence = dict(state.get("best_evidence") or state.get("last_evidence") or {})
        frames = int(state.get("frames", 0))
        if evidence.get("click_error"):
            return "click_error"
        if evidence.get("capture_error"):
            return "capture_error"
        if evidence.get("stale_frame") or int(state.get("duplicate_frames", 0)) > 0:
            return "stale_frame"
        if evidence.get("evidence_conflict"):
            return "evidence_conflict"
        if frames < 2:
            return "second_frame_unavailable"
        if evidence.get("hand_uncertain_change") or (
            evidence.get("hand_change") and int(evidence.get("independent_signals", 0)) < 2
        ):
            return "hand_change_unstable"
        if not evidence.get("elixir_change"):
            return "insufficient_elixir_evidence"
        return "unclassified"

    def summary(self) -> dict[str, Any]:
        """Return bounded, non-mutating confirmation coverage diagnostics."""
        from .temporal import TimingStats

        return {
            "outcome_counts": dict(sorted(self._outcome_counts.items())),
            "failure_counts": dict(sorted(self._failure_counts.items())),
            "elapsed_s": {
                "count": len(self._elapsed_samples),
                "p50_s": TimingStats.percentile(self._elapsed_samples, 0.50),
                "p95_s": TimingStats.percentile(self._elapsed_samples, 0.95),
            },
            "observed_frames": {
                "count": len(self._frame_samples),
                "p50": TimingStats.percentile(self._frame_samples, 0.50),
                "p95": TimingStats.percentile(self._frame_samples, 0.95),
            },
        }

    def register(self, action_id: str, *, sent_at: float | None = None) -> ActionConfirmation:
        action_id = str(action_id).strip()
        if not action_id:
            raise ValueError("action_id 不能为空")
        existing = self._resolved.get(action_id)
        if existing is not None:
            return existing
        if action_id not in self._pending:
            registered_at = time.monotonic() if sent_at is None else float(sent_at)
            self._pending[action_id] = self._new_pending_state(registered_at)
        return ActionConfirmation(action_id, "sent", 0.0, "等待出牌证据", {}, 0.0, 0)

    def observe(
        self,
        action_id: str,
        evidence: dict[str, Any],
        *,
        now: float | None = None,
    ) -> ActionConfirmation:
        action_id = str(action_id).strip()
        if action_id in self._resolved:
            return self._resolved[action_id]
        current = time.monotonic() if now is None else float(now)
        state = self._pending.setdefault(
            action_id,
            self._new_pending_state(current),
        )
        elapsed = max(0.0, current - float(state["sent_at"]))
        state["frames"] += 1
        fingerprint = evidence.get("frame_fingerprint")
        duplicate_frame = bool(
            fingerprint and fingerprint == state.get("last_frame_fingerprint")
        )
        if fingerprint:
            state["last_frame_fingerprint"] = fingerprint
        if duplicate_frame:
            state["duplicate_frames"] = int(state.get("duplicate_frames", 0)) + 1
            evidence = {**evidence, "stale_frame": True}
        self._remember_evidence(state, evidence)
        signals = int(evidence.get("independent_signals", 0))
        signature = tuple(
            bool(evidence.get(key)) for key in ("hand_change", "elixir_change", "visual_change")
        )
        if signals >= 2 and not duplicate_frame:
            if signature == state.get("last_signature"):
                state["positive_frames"] += 1
            else:
                state["positive_frames"] = 1
            state["last_signature"] = signature
        else:
            state["positive_frames"] = 0
            state["last_signature"] = signature

        if signals >= 2 and state["positive_frames"] >= self.stable_frames:
            result = self._finish(
                action_id,
                ActionConfirmation(
                    action_id,
                    "confirmed",
                    max(0.0, min(1.0, float(evidence.get("confidence", 0.0)))),
                    "稳定观察到至少两项独立出牌证据",
                    dict(evidence),
                    elapsed,
                    int(state["frames"]),
                ),
            )
            return result

        if elapsed >= self.timeout_s:
            return self._finish_timeout(action_id, state, current)

        return ActionConfirmation(
            action_id,
            "sent",
            max(0.0, min(1.0, float(evidence.get("confidence", 0.0)))),
            "继续等待稳定出牌证据",
            dict(evidence),
            elapsed,
            int(state["frames"]),
        )

    def timeout(self, action_id: str, *, now: float | None = None) -> ActionConfirmation:
        action_id = str(action_id).strip()
        resolved = self._resolved.get(action_id)
        if resolved is not None:
            return resolved
        state = self._pending.get(action_id)
        if state is None:
            return ActionConfirmation(
                action_id, "unknown", 0.0, "未注册的动作", {}, 0.0, 0
            )
        current = time.monotonic() if now is None else float(now)
        current = max(current, float(state["sent_at"]) + self.timeout_s)
        return self._finish_timeout(action_id, state, current)

    def force_unknown(
        self,
        action_id: str,
        *,
        reason: str,
        evidence: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> ActionConfirmation:
        """Close an interrupted attempt without treating it as a success."""
        action_id = str(action_id).strip()
        existing = self._resolved.get(action_id)
        if existing is not None:
            return existing
        current = time.monotonic() if now is None else float(now)
        state = self._pending.setdefault(
            action_id,
            self._new_pending_state(current),
        )
        if evidence:
            self._remember_evidence(state, evidence)
        preserved = self._preserved_evidence(state)
        result = ActionConfirmation(
            action_id,
            "unknown",
            max(0.0, min(1.0, float(preserved.get("confidence", 0.0)))),
            reason,
            preserved,
            max(0.0, current - float(state.get("sent_at", current))),
            int(state.get("frames", 0)),
            self._failure_code(state, status="unknown"),
        )
        return self._finish(action_id, result)

    def _finish(self, action_id: str, result: ActionConfirmation) -> ActionConfirmation:
        self._pending.pop(action_id, None)
        self._resolved[action_id] = result
        self._outcome_counts[result.status] = self._outcome_counts.get(result.status, 0) + 1
        if result.failure_code:
            self._failure_counts[result.failure_code] = self._failure_counts.get(result.failure_code, 0) + 1
        self._elapsed_samples.append(float(result.elapsed_s))
        self._frame_samples.append(int(result.observed_frames))
        if len(self._elapsed_samples) > 512:
            del self._elapsed_samples[:-512]
            del self._frame_samples[:-512]
        # Resolved IDs are only an idempotence guard; avoid unbounded growth in
        # a long-running bot while retaining a useful recent window.
        if len(self._resolved) > 256:
            oldest = next(iter(self._resolved))
            self._resolved.pop(oldest, None)
        return result
