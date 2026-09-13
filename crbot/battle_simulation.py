"""Bounded spatial forward model. All results are estimates, never observations."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

from .knowledge import KnowledgeBase, UnitSpec


@dataclass(frozen=True)
class SimAction:
    card_id: str | None = None
    slot: int = -1
    x: float = .5
    y: float = .65
    wait_s: float = 0.0

    @property
    def label(self):
        return "WAIT" if self.card_id is None else f"{self.card_id}@{self.x:.2f},{self.y:.2f}"


@dataclass
class Entity:
    uid: int
    spec: UnitSpec
    side: int
    x: float
    y: float
    hp: float
    shield: float = 0
    ready_at: float = 0
    expires_at: float = math.inf
    spawn_at: float = math.inf
    target: int | None = None
    value: float = 0
    tower: bool = False
    walked: float = 0
    locked_at: float = 0
    stunned_until: float = 0


@dataclass
class SimState:
    time: float = 0
    entities: list[Entity] = field(default_factory=list)
    elixir: dict[int, float] = field(default_factory=lambda: {1: 5., -1: 5.})
    hands: dict[int, list[tuple[int, str]]] = field(default_factory=lambda: {1: [], -1: []})
    damage: dict[int, float] = field(default_factory=lambda: {1: 0., -1: 0.})
    impacts: list[tuple] = field(default_factory=list)
    uncertainties: set[str] = field(default_factory=set)
    next_uid: int = 1
    seconds_per_elixir: float = 2.8

    def clone(self):
        # Specs are immutable and can be shared across all branches.
        return SimState(self.time, [copy.copy(e) for e in self.entities], dict(self.elixir),
                        {k: list(v) for k, v in self.hands.items()}, dict(self.damage),
                        list(self.impacts), set(self.uncertainties), self.next_uid, self.seconds_per_elixir)


class Simulator:
    def __init__(self, knowledge: KnowledgeBase, level: int = 11, step: float = .25):
        self.kb, self.level, self.step = knowledge, level, max(.05, min(.5, step))

    @staticmethod
    def xy(x, y):
        return (x - .05) / .9 * 18, (y - .18) / .64 * 32

    def add(self, s: SimState, spec: UnitSpec, side: int, x: float, y: float, *, hp_fraction=1.,
            value=0., tower=False, deployed=False):
        e = Entity(s.next_uid, spec, side, x, y, spec.hp * hp_fraction, spec.shield,
                   s.time + (spec.deploy if deployed else 0),
                   s.time + spec.lifetime if spec.lifetime else math.inf,
                   s.time + spec.spawn_start if spec.spawn else math.inf,
                   value=value, tower=tower)
        s.next_uid += 1
        s.entities.append(e)
        s.uncertainties.update(spec.unsupported)
        return e

    def apply(self, s: SimState, a: SimAction, side: int) -> bool:
        if a.card_id is None:
            return True
        card = self.kb.cards.get(a.card_id)
        if not card or card.get("elixir") is None or s.elixir[side] + 1e-6 < card["elixir"]:
            return False
        if a.card_id not in [cid for _, cid in s.hands[side]]:
            return False
        if not (.05 <= a.x <= .95 and .18 <= a.y <= .82):
            return False
        if card["kind"] != "spell" and ((side == 1 and a.y < .51) or (side == -1 and a.y > .49)):
            return False
        roster, spell = self.kb.roster(a.card_id, self.level), self.kb.spell(a.card_id, self.level)
        if not roster and not spell:
            return False
        s.elixir[side] -= card["elixir"]
        s.hands[side] = [(slot, cid) for slot, cid in s.hands[side] if cid != a.card_id]
        x, y = self.xy(a.x, a.y)
        if roster:
            total = sum(n for _, n in roster)
            index = 0
            for spec, count in roster:
                for _ in range(min(20, count)):
                    self.add(s, spec, side, x + ((index % 3) - 1) * .4,
                             y + (index // 3) * .35 * side, value=card["elixir"] / total, deployed=True)
                    index += 1
        elif spell:
            pulses = max(1, int(spell["duration"] / spell["period"]))
            for i in range(min(40, pulses)):
                s.impacts.append((s.time + spell["delay"] + i * spell["period"], side, None,
                                  x, y, spell["damage"], spell["radius"], spell["air"], spell["tower_multiplier"],
                                  spell["stun"], spell["pushback"]))
        s.uncertainties.update(card.get("unsupported", []))
        return True

    @staticmethod
    def distance(a: Entity, b: Entity):
        return max(0, math.hypot(a.x - b.x, a.y - b.y) - a.spec.radius - b.spec.radius)

    def hit(self, s: SimState, e: Entity, damage: float):
        if e.hp <= 0:
            return
        if e.shield > 0:
            e.shield = max(0, e.shield - damage)  # shield consumes hit, no overflow
            return
        dealt = min(e.hp, max(0, damage))
        e.hp -= dealt
        if e.tower:
            s.damage[e.side] += dealt

    def advance(self, s: SimState, duration: float, *, deadline=None, clock=None):
        end = s.time + max(0, duration)
        while s.time < end - 1e-6:
            if deadline is not None and clock() >= deadline:
                raise TimeoutError("推演预算耗尽")
            dt = min(self.step, end - s.time)
            s.time += dt
            for side in (1, -1):
                s.elixir[side] = min(10, s.elixir[side] + dt / s.seconds_per_elixir)
            impacts, s.impacts = s.impacts, []
            for impact in impacts:
                at, side, target, x, y, damage, radius, air, tower_mult, *control = impact
                if at > s.time:
                    s.impacts.append(impact)
                    continue
                for e in s.entities:
                    if e.side == side or e.hp <= 0 or (e.spec.air and not air):
                        continue
                    if (radius > 0 and math.hypot(e.x - x, e.y - y) <= radius + e.spec.radius) or (radius == 0 and e.uid == target):
                        self.hit(s, e, damage * (tower_mult if e.tower else 1))
                        if control and not e.tower:
                            stun, pushback = control
                            e.stunned_until = max(e.stunned_until, s.time + stun)
                            if stun:
                                e.walked, e.locked_at = 0, s.time + stun
                            if pushback and not e.spec.building:
                                e.y = max(0, min(32, e.y - side * pushback))
            for e in list(s.entities):
                if e.hp <= 0:
                    continue
                if e.expires_at <= s.time:
                    e.hp = 0
                    continue
                if s.time < max(e.ready_at, e.stunned_until):
                    continue
                if e.spawn_at <= s.time and len(s.entities) < 96:
                    spec = self.kb.unit(e.spec.spawn, self.level)
                    if spec:
                        for i in range(min(8, e.spec.spawn_count)):
                            self.add(s, spec, e.side, e.x + i * .3, e.y, deployed=True)
                    e.spawn_at = s.time + max(1, e.spec.spawn_period)
                enemies = [t for t in s.entities if t.side != e.side and t.hp > 0
                           and ("air" if t.spec.air else "ground") in e.spec.targets
                           and (not e.spec.building_only or t.spec.building or t.tower)]
                target = next((t for t in enemies if t.uid == e.target and self.distance(e, t) <= e.spec.sight), None)
                if target is None:
                    nearby = [t for t in enemies if self.distance(e, t) <= e.spec.sight]
                    towers = [t for t in enemies if t.tower]
                    target = min(nearby or towers, key=lambda t: self.distance(e, t), default=None)
                    if target and target.uid != e.target:
                        e.target = target.uid
                        e.locked_at = s.time
                        e.ready_at = s.time + e.spec.first_hit
                if target is None:
                    continue
                if self.distance(e, target) <= e.spec.reach:
                    if s.time < e.ready_at or e.spec.damage <= 0:
                        continue
                    delay = self.distance(e, target) / e.spec.projectile_speed if e.spec.projectile_speed else 0
                    damage = e.spec.damage
                    if e.spec.charge_distance and e.walked >= e.spec.charge_distance:
                        damage = e.spec.charge_damage or damage
                    elif e.spec.ramp_times[0]:
                        held = s.time - e.locked_at
                        if held >= sum(e.spec.ramp_times):
                            damage = e.spec.ramp_damage[1]
                        elif held >= e.spec.ramp_times[0]:
                            damage = e.spec.ramp_damage[0]
                    e.walked = 0
                    s.impacts.append((s.time + delay, e.side, target.uid, target.x, target.y,
                                      damage, e.spec.splash, "air" in e.spec.targets, 1., e.spec.stun, e.spec.pushback))
                    e.ready_at = s.time + e.spec.period
                elif e.spec.speed > 0:
                    tx, ty = target.x, target.y
                    if not e.spec.air and (e.y - 16) * (ty - 16) < 0:
                        # Ground units cross on a bridge before heading for the target.
                        tx = 4.5 if e.x < 9 else 13.5
                        ty = 16 - e.side * .5
                    dist = math.hypot(tx - e.x, ty - e.y)
                    if dist:
                        multiplier = e.spec.charge_multiplier if e.spec.charge_distance and e.walked >= e.spec.charge_distance else 1
                        step = min(e.spec.speed * multiplier * dt, dist)
                        e.walked += step
                        e.x += (tx - e.x) / dist * step
                        e.y += (ty - e.y) / dist * step
            dead = [e for e in s.entities if e.hp <= 0]
            s.entities = [e for e in s.entities if e.hp > 0]
            for e in dead:
                if e.spec.death_damage:
                    s.impacts.append((s.time, e.side, None, e.x, e.y, e.spec.death_damage,
                                      e.spec.death_radius, True, 1.))
                spec = self.kb.unit(e.spec.death_spawn, self.level) if e.spec.death_spawn else None
                if spec and len(s.entities) < 96:
                    for i in range(min(8, e.spec.death_count)):
                        self.add(s, spec, e.side, e.x + i * .35, e.y, deployed=True)
            if len(s.entities) >= 96:
                s.uncertainties.add("entity_budget")

    @staticmethod
    def evaluate(s: SimState) -> tuple[float, dict]:
        material = sum(e.side * e.value * max(0, e.hp / e.spec.hp) for e in s.entities if not e.tower)
        reserve = s.elixir[1] - s.elixir[-1]
        own_loss, enemy_loss = s.damage[1], s.damage[-1]
        own_towers = sum(e.tower and e.side == 1 for e in s.entities)
        enemy_towers = sum(e.tower and e.side == -1 for e in s.entities)
        score = (enemy_loss - 1.5 * own_loss) / 150 + .65 * material + .35 * reserve
        score += 18 * (own_towers - enemy_towers)
        score -= .08 * len(s.uncertainties)
        return score, {"own_tower_damage": round(own_loss, 1), "enemy_tower_damage": round(enemy_loss, 1),
                       "material": round(material, 2), "elixir_advantage": round(reserve, 2),
                       "uncertainties": sorted(s.uncertainties)}
