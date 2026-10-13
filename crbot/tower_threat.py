"""Finite-HP tail threat using current target locks, shots and active defenders.

This short, conservative estimate complements completed combat branches. It
does not invent future deployments or credit a defender marching into range.
"""
import math


def assess(state, enemy, tower, window=3., targets=None):
    from .battle_simulation import Simulator
    from .grid_world import target_for
    distance = Simulator.distance(enemy, tower)
    eta = max(0., distance-enemy.spec.reach)/max(.1, enemy.spec.speed)
    arrival = max(eta, enemy.active_at-state.time, enemy.stunned_until-state.time,
                  enemy.ready_at-state.time, 0.)
    if distance > enemy.spec.reach or enemy.winding_target != tower.uid:
        arrival += enemy.spec.first_hit
    if enemy.spec.projectile_speed:
        arrival += max(0., distance-enemy.spec.speed*eta)/enemy.spec.projectile_speed
    shots = []
    for defender in state.entities:
        if defender.side == enemy.side or defender.hp <= 0 or defender.spec.damage <= 0:
            continue
        target = targets.get(defender.uid) if targets is not None else target_for(Simulator,state,defender)
        if target is None or target.uid != enemy.uid or Simulator.distance(defender,enemy) > defender.spec.reach:
            continue
        at = max(0.,defender.ready_at-state.time,defender.active_at-state.time,defender.stunned_until-state.time)
        if defender.winding_target != enemy.uid:
            at += defender.spec.first_hit
        if defender.spec.projectile_speed:
            at += Simulator.distance(defender,enemy)/defender.spec.projectile_speed
        # Attack acquisition and impact resolution happen on discrete ticks;
        # round defender hits late rather than claiming a kill too early.
        at = math.ceil((at+.25)/.25)*.25
        # Do not count shots after a defender is expected to die to its attackers.
        incoming = sum(e.spec.damage/max(.1,e.spec.period) for e in state.entities
                       if e.side != defender.side and e.target == defender.uid and e.hp > 0
                       and Simulator.distance(e,defender) <= e.spec.reach)
        survival = defender.hp/incoming if incoming else math.inf
        while at <= min(window,survival)+1e-6:
            travel=min(eta,at)*enemy.spec.speed
            center_distance=max(.001,math.hypot(tower.x-enemy.x,tower.y-enemy.y))
            px=enemy.x+(tower.x-enemy.x)*travel/center_distance
            py=enemy.y+(tower.y-enemy.y)*travel/center_distance
            future_distance=max(0.,math.hypot(defender.x-px,defender.y-py)-defender.spec.radius-enemy.spec.radius)
            if future_distance <= defender.spec.reach:
                shots.append((at,defender.spec.damage))
            at += max(.1,defender.spec.period)
    for impact in state.impacts:
        at,side,target,x,y,damage,radius,air,*rest = impact
        if side != enemy.side and (not enemy.spec.air or air) and (target == enemy.uid or
                radius > 0 and math.hypot(enemy.x-x,enemy.y-y) <= radius+enemy.spec.radius):
            shots.append((max(0.,at-state.time),damage))
    hp = enemy.hp+enemy.shield
    death = None
    for at,damage in sorted(shots):
        hp -= damage
        if hp <= 0:
            death = at
            break
    damage = 0.
    hit = arrival
    while hit <= window+1e-6 and (death is None or hit < death-1e-6):
        damage += enemy.spec.damage
        hit += max(.1,enemy.spec.period)
    if death is not None and enemy.spec.death_damage and death+enemy.spec.death_delay <= window:
        travel=min(eta,death)*enemy.spec.speed
        center_distance=max(.001,math.hypot(tower.x-enemy.x,tower.y-enemy.y))
        px=enemy.x+(tower.x-enemy.x)*travel/center_distance
        py=enemy.y+(tower.y-enemy.y)*travel/center_distance
        if math.hypot(px-tower.x,py-tower.y) <= enemy.spec.death_radius+tower.spec.radius:
            damage += enemy.spec.death_damage*enemy.spec.death_tower_multiplier
    return dict(estimated_hp=round(enemy.hp,1),first_tower_hit_s=round(arrival,3),
                estimated_kill_s=None if death is None else round(death,3),
                tower_damage=round(min(tower.hp,damage),1),credited_shots=len(shots),
                status='combat_estimate')
