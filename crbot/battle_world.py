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
    identity_observed_at: float | None = None
    level: int | None = None
    side_evidence: str | None = None
    identity_lock_id: str | None = None
    deployment_id: str | None = None
    identity_source: str | None = None
    visual_card_id: str | None = None
    bbox: tuple[float, ...] = ()
    hp_estimate: float | None = None
    hp_estimate_low: float | None = None
    hp_estimate_high: float | None = None
    hp_estimate_source: str | None = None


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
    unconfirmed_units: tuple[dict, ...] = ()


class BattleWorld:
    def __init__(self):
        self.reset()

    def reset(self):
        self.battle_generation = getattr(self, 'battle_generation', 0) + 1
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
        self.confirmed_cards: dict[str,str] = {}
        self.tower_health=(None,)*4
        self._placement_state=(self.battle_generation,())

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
            if d.get('identity_ambiguous'):
                continue
            cid, side = str(d["card_id"]), int(d.get("side", -1))
            x, y, confidence = float(d["x"]), float(d["y"]), float(d.get("confidence", 0))
            bound = (d.get('identity_source') == 'confirmed_deployment_tracking' and bool(d.get('identity_lock_id'))
                     and self.confirmed_cards.get(d.get('deployment_id'))==cid)
            enemy_bound=(side==-1 and d.get('identity_source')=='enemy_yolo_tracking' and
                         bool(d.get('identity_lock_id')) and float(d.get('identity_anchor_confidence',0))>=.8)
            bound=bound or enemy_bound
            if not all(math.isfinite(v) for v in (x, y, confidence)) or confidence < (.1 if bound else .45):
                continue
            def identity_matches(t):
                if enemy_bound:
                    return t.identity_lock_id == d['identity_lock_id'] or (not t.identity_lock_id and
                        (t.card_id==cid or t.card_id.startswith('unknown:')))
                if bound:
                    return (t.identity_lock_id == d['identity_lock_id'] or
                            (not t.identity_lock_id and t.first_seen >= d.get('deployment_at', now)))
                if t.identity_lock_id:
                    return False
                return t.card_id == cid or t.card_id.startswith('unknown:') or cid.startswith('unknown:')
            def separation(t):
                age=min(.75,max(0,now-t.last_seen))
                return math.hypot(t.x+t.vx*age-x,t.y+t.vy*age-y)
            options = [t for t in self.tracks.values() if t.track_id not in claimed and identity_matches(t)
                       and t.side == side and now - t.last_seen <= 3
                       and ((bound and t.identity_lock_id==d['identity_lock_id']) or separation(t) <= .07 + min(.18, dt * .06))]
            if options:
                t = min(options, key=lambda t: (separation(t),t.card_id != cid))
                age = now - t.last_seen
                if .05 <= age <= 1.5:
                    t.vx = max(-.3, min(.3, (x-t.x)/age))
                    t.vy = max(-.3, min(.3, (y-t.y)/age))
                else:
                    t.vx = t.vy = 0
                t.x, t.y, t.last_seen, t.confidence = x, y, now, confidence
                t.observations += 1
                if not cid.startswith('unknown:'):
                    t.card_id=cid
            else:
                t = Track(self.next_id, cid, side, x, y, now, now, confidence)
                self.tracks[t.track_id] = t
                self.next_id += 1
            claimed.add(t.track_id)
            t.level=d.get('level')
            t.side_evidence=d.get('side_evidence')
            if bound:
                t.identity_lock_id=d['identity_lock_id']
                t.deployment_id=d.get('deployment_id')
                t.identity_source=d['identity_source']
                t.visual_card_id=d.get('visual_card_id')
                t.bbox=tuple(d.get('bbox',()))
                t.hypotheses=()
                for event in self.events:
                    if event['id']==t.deployment_id:
                        event['identity_bound']=True
            if not cid.startswith('unknown:'):
                t.identity_observed_at=now
            elif not t.card_id.startswith('unknown:') and t.identity_observed_at is not None and now-t.identity_observed_at > 2:
                t.card_id=cid
            if d.get("hp_fraction") is not None:
                hp = float(d["hp_fraction"])
                if math.isfinite(hp) and 0 <= hp <= 1 and float(d.get("hp_confidence", 1)) >= .8:
                    t.hp_fraction, t.hp_observed_at = hp, now
                    t.hp_confidence = float(d.get('hp_confidence',1))
            if cid == t.card_id or not cid.startswith('unknown:'):
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
        incoming=tuple(tower_health or (None,)*len(self.tower_health))
        self.tower_health=tuple(0. if i<len(self.tower_health) and self.tower_health[i]==0 else v
                                for i,v in enumerate(incoming))
        self.enemy_elixir = low, high
        self.tracks = {k: t for k, t in self.tracks.items() if now - t.last_seen <= 3}
        # Retain timestamped visual HP anchors. Stale bars are propagated by
        # CombatHealth rather than replaced with a full-health hypothesis.
        self.events = self.events[-128:]
        recent = tuple(e["card_id"] for e in self.events[-8:] if e["kind"] == "deployment")
        reasons = list(uncertain or [])
        if any(d.get('identity_ambiguous') for d in detections):
            reasons.append('ally_identity_association_ambiguous')
        if high - low > 2:
            reasons.append("enemy_elixir_interval")
        if any(t.hp_fraction is None for t in self.tracks.values()):
            reasons.append("unit_health_estimated")
        if any(v is None for v in self.tower_health):
            reasons.append("tower_health_unknown")
        return WorldSnapshot(self.revision, now, elapsed, elixir, (low, high), tuple(hand),
                             tuple(Track(**asdict(t)) for t in self.tracks.values()),
                             tuple(self.enemy_seen), recent, tuple(reasons),
                             self.tower_health, self.enemy_elixir_estimate,
                             tuple(dict(e) for e in self.events if e["kind"] == "ally_deployment" and now - e["at"] < 8))

    def confirm(self, action_id: str, card_id: str, x: float, y: float, now: float, *, issued_at=None):
        if action_id in self.confirmed:
            return
        self.confirmed.add(action_id)
        self.confirmed_cards[action_id]=card_id
        # Confirmed placement is an event, not an invented alive-unit observation.
        self.events.append({"id": action_id, "card_id": card_id, "at": now,
                            "kind": "ally_deployment", "x": x, "y": y,
                            "issued_at": now if issued_at is None else min(now,issued_at)})
        self._placement_state=(self.battle_generation,tuple(dict(e) for e in self.events if e['kind']=='ally_deployment'))

    def status(self) -> dict:
        return {"revision": self.revision, "enemy_elixir_interval": list(self.enemy_elixir),
                "enemy_elixir_estimate": round(self.enemy_elixir_estimate, 2),
                "enemy_seen": list(self.enemy_seen), "tracks": len(self.tracks),
                "recent_events": self.events[-8:],
                "at": self.at,"tower_health": self.tower_health,
                "observations": [asdict(t) for t in self.tracks.values()],
                "unit_health": [{"track_id": t.track_id, "side": t.side, "hp_fraction": t.hp_fraction,
                                 "observed_at": t.hp_observed_at} for t in self.tracks.values()]}

    def placement_state(self):
        # Generation and events are published together in one assignment.
        # Stream consumers never mutate policy tracks or events.
        generation,events=self._placement_state
        return generation,tuple(dict(e) for e in events)
