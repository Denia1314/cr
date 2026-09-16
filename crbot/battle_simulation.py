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
    haste_until: float = 0
    haste: float = 1
    slow_until: float = 0
    slow: float = 1
    cloned: bool = False
    observed_vx: float = 0
    observed_vy: float = 0
    resource_at: float = float('inf')
    track_id: int | None = None
    active_at: float = 0.
    winding_target: int | None = None
    tower_kind: str = ''
    active: bool = True
    attack_slow: float = 1
    dash_phase: str = ''
    dash_target: int | None = None
    dash_at: float = 0
    dash_ready_at: float = 0
    dash_x: float = 0
    dash_y: float = 0


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
    tower_shots: dict[int, int] = field(default_factory=lambda: {1: 0, -1: 0})
    effects: list[dict] = field(default_factory=list)

    def clone(self):
        # Specs are immutable and can be shared across all branches.
        return SimState(self.time, [copy.copy(e) for e in self.entities], dict(self.elixir),
                        {k: list(v) for k, v in self.hands.items()}, dict(self.damage),
                        list(self.impacts), set(self.uncertainties), self.next_uid, self.seconds_per_elixir,
                        dict(self.tower_shots), [dict(e) for e in self.effects])


class Simulator:
    def __init__(self, knowledge: KnowledgeBase, level: int = 11, step: float = .25):
        self.kb, self.level, self.step = knowledge, level, max(.05, min(.5, step))
        from .arena_geometry import ArenaGeometry
        self.geometry=ArenaGeometry()

    def xy(self,x,y):
        return self.geometry.xy(x,y)

    def screen(self,x,y):
        return self.geometry.screen(x,y)

    def legal_placement(self, s, card_id, side, x, y):
        left,top,right,bottom=self.geometry.bounds
        if not (left <= x <= right and top <= y <= bottom):
            return False
        card = self.kb.cards.get(card_id, {})
        px,py=self.xy(x,y)
        home=(py+1e-8>=16.5 if side==1 else py-1e-8<=15.5)
        if card_id == 'royal_delivery' and not home:
            return False
        if card.get('kind') == 'spell':
            return True
        if not home:
            return False
        px, py = self.xy(x, y)
        # King-tower footprint remains excluded even though king activation is not modeled.
        if abs(px-9) < 2 and abs(py-(30 if side == 1 else 2)) < 2:
            return False
        for e in s.entities:
            if e.hp > 0 and (e.tower or e.spec.building):
                if math.hypot(px-e.x, py-e.y) < e.spec.radius + .6:
                    return False
        return True

    def add(self, s: SimState, spec: UnitSpec, side: int, x: float, y: float, *, hp_fraction=1.,
            value=0., tower=False, deployed=False):
        e = Entity(s.next_uid, spec, side, x, y, spec.hp * hp_fraction, spec.shield,
                   s.time + (spec.deploy if deployed else 0),
                   s.time + spec.lifetime if spec.lifetime else math.inf,
                   s.time + spec.spawn_start if spec.spawn else math.inf,
                   value=value, tower=tower)
        e.active_at=s.time+(spec.deploy if deployed else 0)
        s.next_uid += 1
        if spec.resource_period:
            e.resource_at=s.time+spec.deploy+spec.resource_period
        s.entities.append(e)
        s.uncertainties.update(spec.unsupported)
        if deployed:
            from .unit_mechanics import queue_spawn_pulse
            queue_spawn_pulse(self,s,e)
        return e

    def apply(self, s: SimState, a: SimAction, side: int) -> bool:
        if a.card_id is None:
            return True
        card = self.kb.cards.get(a.card_id)
        if not card or card.get("elixir") is None or s.elixir[side] + 1e-6 < card["elixir"]:
            return False
        if a.card_id not in [cid for _, cid in s.hands[side]]:
            return False
        if not self.legal_placement(s, a.card_id, side, a.x, a.y):
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
            from .unit_mechanics import deployment_pulses
            for event in deployment_pulses(self.kb,a.card_id,self.level):
                s.effects.append(dict(event,at=s.time+event['delay'],side=side,x=x,y=y))
        elif spell:
            if spell.get('pattern'):
                from .card_effects import queue_effect
                queue_effect(self,s,dict(spell,summon_value=card['elixir']/max(1,spell.get('spawn_count',1))),side,x,y)
                s.uncertainties.add('public_spell_pattern_spatial_approximation')
                return True
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

    def hit(self, s: SimState, e: Entity, damage: float, attacker=None):
        if e.hp <= 0 or (e.spec.dash_immune and e.dash_phase=='flight'):
            return False
        if attacker is not None and e.spec.reflected_damage and attacker.side!=e.side and self.distance(e,attacker)<=e.spec.reflected_radius:
            if self.hit(s,attacker,e.spec.reflected_damage):
                from .unit_mechanics import control
                control(s,attacker,stun=e.spec.reflected_stun)
            s.uncertainties.add('retaliation_source_and_level_scaling_estimate')
        if e.shield > 0:
            e.shield = max(0, e.shield - damage)  # shield consumes hit, no overflow
            return True
        dealt = min(e.hp, max(0, damage))
        e.hp -= dealt
        if e.tower:
            s.damage[e.side] += dealt
        return True

    def advance(self, s: SimState, duration: float, *, deadline=None, clock=None):
        end = s.time + max(0, duration)
        while s.time < end - 1e-6:
            if deadline is not None and clock() >= deadline:
                raise TimeoutError("推演预算耗尽")
            dt = min(self.step, end - s.time)
            s.time += dt
            from .card_effects import resolve_effects
            if s.effects:
                resolve_effects(self,s)
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
                        attacker=next((a for a in s.entities if a.uid==control[2]),None) if len(control)>2 else None
                        if not self.hit(s, e, damage * (tower_mult if e.tower else 1),attacker):continue
                        if control:
                            from .unit_mechanics import control as apply_control
                            stun, pushback = control[:2]
                            apply_control(s,e,stun=stun,pushback=pushback,x=x,y=y+side*.001)
            for e in list(s.entities):
                if e.hp <= 0:
                    continue
                if e.expires_at <= s.time:
                    e.hp = 0
                    continue
                if s.time < max(e.active_at, e.stunned_until):
                    continue
                if s.time >= e.resource_at:
                    s.elixir[e.side]=min(10,s.elixir[e.side]+e.spec.resource_amount)
                    e.resource_at=s.time+e.spec.resource_period
                if e.spawn_at <= s.time and len(s.entities) < 96:
                    spec = self.kb.unit(e.spec.spawn, self.level)
                    if spec:
                        for i in range(min(8, e.spec.spawn_count)):
                            child=self.add(s, spec, e.side, e.x + i * .3, e.y, deployed=True,
                                           hp_fraction=1/spec.hp if e.cloned else 1)
                            child.cloned=e.cloned
                            if e.cloned:
                                child.shield=1 if spec.shield else 0
                    e.spawn_at = s.time + max(1, e.spec.spawn_period)
                if e.tower_kind == 'king' and not e.active:
                    e.active = e.hp < e.spec.hp or sum(t.tower and t.side == e.side and t.tower_kind != 'king' for t in s.entities) < 2
                    if not e.active:
                        continue
                from .grid_world import target_for
                target = target_for(self,s,e)
                from .unit_mechanics import advance_dash
                if advance_dash(self,s,e,target):continue
                if target and target.uid != e.target:
                    e.target=target.uid
                    e.winding_target=None
                if target is None:
                    continue
                if self.distance(e, target) <= e.spec.reach:
                    if e.winding_target != target.uid:
                        e.winding_target=target.uid
                        e.locked_at=s.time
                        e.ready_at=max(e.ready_at,s.time+e.spec.first_hit)
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
                    victims=[target]
                    if e.spec.chain_count>1:
                        while len(victims)<e.spec.chain_count:
                            previous=victims[-1]
                            nearby=[v for v in s.entities if v.side!=e.side and v.hp>0 and v not in victims
                                    and ('air' if v.spec.air else 'ground') in e.spec.targets
                                    and self.distance(previous,v)<=e.spec.chain_radius]
                            if not nearby:break
                            victims.append(min(nearby,key=lambda v:(self.distance(previous,v),v.uid)))
                        s.uncertainties.add('chain_travel_time_estimate')
                    elif e.spec.attack_targets>1:
                        others=sorted((v for v in s.entities if v.side!=e.side and v.hp>0 and v.uid!=target.uid
                                       and ('air' if v.spec.air else 'ground') in e.spec.targets
                                       and self.distance(e,v)<=e.spec.reach),key=lambda v:(self.distance(e,v),v.uid))
                        victims+=others[:e.spec.attack_targets-1]
                    # Source damage is per projectile/target; a lone target receives both bolts.
                    chain_delay=0.;previous_source=e
                    for i in range(len(victims) if e.spec.chain_count>1 else e.spec.attack_targets):
                        victim=victims[i%len(victims)]
                        delay=self.distance(e,victim)/e.spec.projectile_speed if e.spec.projectile_speed else 0
                        if e.spec.chain_count>1:
                            chain_delay+=self.distance(previous_source,victim)/e.spec.projectile_speed if e.spec.projectile_speed else 0
                            delay=chain_delay;previous_source=victim
                        s.impacts.append((s.time+delay,e.side,victim.uid,victim.x,victim.y,
                                          damage,e.spec.splash,'air' in e.spec.targets,1.,e.spec.stun,e.spec.pushback,e.uid))
                    speed_factor=(e.haste if s.time<e.haste_until else 1)*(e.attack_slow if s.time<e.slow_until else 1)
                    e.ready_at = s.time + e.spec.period / speed_factor
                    if e.tower:
                        s.tower_shots[e.side] += 1
                elif e.spec.speed > 0 and not e.tower:
                    e.winding_target=None
                    if getattr(self, 'grid_navigation', False):
                        from .grid_navigation import move
                        multiplier = e.spec.charge_multiplier if e.spec.charge_distance and e.walked >= e.spec.charge_distance else 1
                        speed_factor=(e.haste if s.time < e.haste_until else 1)*(e.slow if s.time < e.slow_until else 1)
                        e.walked += move(s,e,target,self.geometry,e.spec.speed*multiplier*speed_factor*dt)
                        continue
                    tx, ty = target.x, target.y
                    if not e.spec.air and (e.y - 16) * (ty - 16) < 0:
                        # Ground units cross on a bridge before heading for the target.
                        tx = self.geometry.bridges[0] if e.x < 9 else self.geometry.bridges[1]
                        ty = 16 - e.side * .5
                    dist = math.hypot(tx - e.x, ty - e.y)
                    if dist:
                        multiplier = e.spec.charge_multiplier if e.spec.charge_distance and e.walked >= e.spec.charge_distance else 1
                        speed_factor=(e.haste if s.time < e.haste_until else 1)*(e.slow if s.time < e.slow_until else 1)
                        step = min(e.spec.speed * multiplier * speed_factor * dt, dist)
                        e.walked += step
                        e.x += (tx - e.x) / dist * step
                        e.y += (ty - e.y) / dist * step
            dead = [e for e in s.entities if e.hp <= 0]
            s.entities = [e for e in s.entities if e.hp > 0]
            for e in dead:
                if e.spec.resource_on_death:
                    s.elixir[e.side]=min(10,s.elixir[e.side]+e.spec.resource_on_death)
                if e.spec.death_damage:
                    s.impacts.append((s.time, e.side, None, e.x, e.y, e.spec.death_damage,
                                      e.spec.death_radius, True, 1.))
                spec = self.kb.unit(e.spec.death_spawn, self.level) if e.spec.death_spawn else None
                if spec and len(s.entities) < 96:
                    for i in range(min(8, e.spec.death_count)):
                        child=self.add(s, spec, e.side, e.x + i * .35, e.y, deployed=True,
                                       hp_fraction=1/spec.hp if e.cloned else 1)
                        child.cloned=e.cloned
                        if e.cloned:
                            child.shield=1 if spec.shield else 0
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
        # Penalize imminent damage beyond the horizon, especially at a nearly fallen tower.
        exposure = 0.
        for enemy in s.entities:
            if enemy.side != -1 or enemy.tower or enemy.spec.damage <= 0:
                continue
            tower = min((t for t in s.entities if t.side == 1 and t.tower),
                        key=lambda t: Simulator.distance(enemy, t), default=None)
            if tower is None:
                continue
            if enemy.target is not None and enemy.target != tower.uid:
                continue
            eta = max(0, Simulator.distance(enemy, tower) - enemy.spec.reach) / max(.1, enemy.spec.speed)
            exposure += enemy.spec.damage / max(.1, enemy.spec.period) * max(0, 3-eta) * (2-tower.hp/tower.spec.hp)
        score -= exposure / 150
        score -= .08 * len(s.uncertainties)
        return score, {"own_towers_remaining": own_towers, "own_tower_damage": round(own_loss, 1), "enemy_tower_damage": round(enemy_loss, 1),
                       "own_tower_shots": s.tower_shots[1], "enemy_tower_shots": s.tower_shots[-1],
                       "imminent_tower_exposure": round(exposure, 1),
                       "material": round(material, 2), "elixir_advantage": round(reserve, 2),
                       "uncertainties": sorted(s.uncertainties)}
