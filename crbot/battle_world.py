"""Observations and estimates are separate from hypothetical search states."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field


@dataclass
class Track:
    track_id: int
    card_id: str
    side: int
    x: float
    y: float
    first_seen: float
    last_seen: float
    confidence: float
    observations: int = 1
    hp_fraction: float | None = None
    variant: str = "unknown"
    hypotheses: tuple[str, ...] = ()
    vx: float = 0
    vy: float = 0
    hp_observed_at: float | None = None
    hp_confidence: float = 0.


@dataclass(frozen=True)
class WorldSnapshot:
    revision: int
    at: float
    elapsed: float
    elixir: float
    enemy_elixir: tuple[float, float]
    hand: tuple[tuple[int, str], ...]
    tracks: tuple[Track, ...]
    enemy_seen: tuple[str, ...]
    enemy_recent: tuple[str, ...]
    uncertainty: tuple[str, ...]
    tower_health: tuple[float | None, ...] = (None, None, None, None)
    enemy_elixir_estimate: float = 5.0
    confirmed_placements: tuple[dict, ...] = ()
    observation_delay_s: float = 0.


class BattleWorld:
    def __init__(self):
        self.reset()

    def reset(self):
        self.revision = 0
        self.at: float | None = None
        self.tracks: dict[int, Track] = {}
        self.next_id = 1
        self.enemy_elixir = (0.0, 10.0)
        self.enemy_elixir_estimate = 5.0
        self.enemy_seen: dict[str, float] = {}
        self.events: list[dict] = []
        self._event_keys: set[str] = set()
        self.confirmed: set[str] = set()
        self.tower_health=(None,)*4

    def update(self, detections: list[dict], *, now: float, elapsed: float, elixir: float,
               hand: list[tuple[int, str]], costs: dict[str, float], seconds_per_elixir: float,
               uncertain: list[str] | None = None, tower_health=None) -> WorldSnapshot:
        if self.at is not None and now < self.at:
            raise ValueError("观测时间倒退")
        dt = max(0, now - self.at) if self.at is not None else 0
        rate = dt / max(.1, seconds_per_elixir)
        low, high = self.enemy_elixir
        low, high = min(10, low + rate), min(10, high + rate)
        self.enemy_elixir_estimate = min(10, self.enemy_elixir_estimate + rate)
        low = 0  # unseen spells/deployments remain possible with camera-only partial observations
        # After a gap we cannot assume no hidden plays happened.
        if dt > 2:
            low = 0
        self.at = now
        self.revision += 1
        claimed: set[int] = set()
        for d in sorted(detections, key=lambda v: -float(v.get("confidence", 0))):
            cid, side = str(d["card_id"]), int(d.get("side", -1))
            x, y, confidence = float(d["x"]), float(d["y"]), float(d.get("confidence", 0))
            if not all(math.isfinite(v) for v in (x, y, confidence)) or confidence < .45:
                continue
            options = [t for t in self.tracks.values() if t.track_id not in claimed and t.card_id == cid
                       and t.side == side and now - t.last_seen <= 3
                       and math.hypot(t.x - x, t.y - y) <= .07 + min(.18, dt * .06)]
            if options:
                t = min(options, key=lambda t: math.hypot(t.x - x, t.y - y))
                age = now - t.last_seen
                if .05 <= age <= 1.5:
                    t.vx = max(-.3, min(.3, (x-t.x)/age))
                    t.vy = max(-.3, min(.3, (y-t.y)/age))
                else:
                    t.vx = t.vy = 0
                t.x, t.y, t.last_seen, t.confidence = x, y, now, confidence
                t.observations += 1
            else:
                t = Track(self.next_id, cid, side, x, y, now, now, confidence)
                self.tracks[t.track_id] = t
                self.next_id += 1
            claimed.add(t.track_id)
            if d.get("hp_fraction") is not None:
                hp = float(d["hp_fraction"])
                if math.isfinite(hp) and 0 <= hp <= 1 and float(d.get("hp_confidence", 1)) >= .8:
                    t.hp_fraction, t.hp_observed_at = hp, now
                    t.hp_confidence = float(d.get('hp_confidence',1))
            t.variant = str(d.get("variant", "unknown"))
            t.hypotheses = tuple(d.get("hypotheses", ()))
            if side == -1:
                if not cid.startswith("unknown:"):
                    self.enemy_seen[cid] = now
                # A detected entity is not proof of a deployment. Only an explicit,
                # independently identified deploy event can constrain both endpoints.
                event_id = str(d.get("deploy_event_id") or "")
                if event_id and d.get("deployment_confirmed") and event_id not in self._event_keys:
                    self._event_keys.add(event_id)
                    cost = costs.get(cid)
                    if cost is not None:
                        low, high = max(0, low - cost), max(0, high - cost)
                        self.enemy_elixir_estimate = max(0, self.enemy_elixir_estimate - cost)
                    self.events.append({"id": event_id, "card_id": cid, "at": now, "kind": "deployment"})
                elif t.observations == 1:
                    # Possible appearance/spawn: widen, never count each swarm member as a card.
                    low = 0
                    if not any(e["card_id"] == cid and now - e["at"] < 2 for e in self.events):
                        # Soft evidence only, discounted for pre-existing/produced entities.
                        likelihood = .6 if y < .45 else .2
                        self.enemy_elixir_estimate = max(0, self.enemy_elixir_estimate - costs.get(cid, 0) * likelihood)
                        self.events.append({"id": f"sighting-{t.track_id}", "card_id": cid,
                                            "at": now, "kind": "sighting", "deployment_likelihood": likelihood})
        self.tower_health=tuple(tower_health or (None,)*4)
        self.enemy_elixir = low, high
        self.tracks = {k: t for k, t in self.tracks.items() if now - t.last_seen <= 3}
        for track in self.tracks.values():
            if track.hp_observed_at is not None and now - track.hp_observed_at > .6:
                track.hp_fraction = None
                track.hp_confidence = 0.
        self.events = self.events[-128:]
        recent = tuple(e["card_id"] for e in self.events[-8:] if e["kind"] == "deployment")
        reasons = list(uncertain or [])
        if high - low > 2:
            reasons.append("enemy_elixir_interval")
        if any(t.hp_fraction is None for t in self.tracks.values()):
            reasons.append("unit_health_estimated")
        if tower_health is None or any(v is None for v in tower_health):
            reasons.append("tower_health_unknown")
        return WorldSnapshot(self.revision, now, elapsed, elixir, (low, high), tuple(hand),
                             tuple(Track(**asdict(t)) for t in self.tracks.values()),
                             tuple(self.enemy_seen), recent, tuple(reasons),
                             tuple(tower_health or (None, None, None, None)), self.enemy_elixir_estimate,
                             tuple(dict(e) for e in self.events if e["kind"] == "ally_deployment" and now - e["at"] < 8))

    def confirm(self, action_id: str, card_id: str, x: float, y: float, now: float):
        if action_id in self.confirmed:
            return
        self.confirmed.add(action_id)
        # Confirmed placement is an event, not an invented alive-unit observation.
        self.events.append({"id": action_id, "card_id": card_id, "at": now,
                            "kind": "ally_deployment", "x": x, "y": y})

    def status(self) -> dict:
        return {"revision": self.revision, "enemy_elixir_interval": list(self.enemy_elixir),
                "enemy_elixir_estimate": round(self.enemy_elixir_estimate, 2),
                "enemy_seen": list(self.enemy_seen), "tracks": len(self.tracks),
                "recent_events": self.events[-8:],
                "at": self.at,"tower_health": self.tower_health,
                "observations": [asdict(t) for t in self.tracks.values()],
                "unit_health": [{"track_id": t.track_id, "side": t.side, "hp_fraction": t.hp_fraction,
                                 "observed_at": t.hp_observed_at} for t in self.tracks.values()]}
