"""Database-derived screening features, not a replacement for combat simulation."""
from __future__ import annotations

import math


def matchup(spec, enemy, count=1):
    """Compare discrete hits and survival; never turn overkill into useful DPS.

    This is a local duel estimate. Splash density, movement, tower fire and
    multiple attackers still require the planner's completed combat branches.
    """
    target = enemy.spec
    hit = (not spec.building_only or target.building) and (
        'air' if target.air else 'ground') in spec.targets
    retaliates = (not target.building_only or spec.building) and (
        'air' if spec.air else 'ground') in target.targets
    hits = (math.ceil(enemy.hp / spec.damage) + math.ceil(enemy.shield / spec.damage)
            if hit and spec.damage > 0 else None)
    kill_s = (spec.first_hit + max(0, math.ceil(hits / max(1, count))-1)*spec.period
              if hits is not None else None)
    incoming_hits = (math.ceil(spec.hp / target.damage) + math.ceil(spec.shield / target.damage)
                     if retaliates and target.damage > 0 else None)
    # A splash attacker can remove a compact swarm together. A single-target
    # attacker must spend attacks on each body; shields consume separate hits.
    survival_s = (target.first_hit + max(0, incoming_hits*(1 if target.splash > 0 else count)-1)*target.period
                  if incoming_hits is not None else None)
    attack = 0. if kill_s is None else 1. / (1. + kill_s / 2.)
    if kill_s is not None and survival_s is not None and kill_s > survival_s:
        attack *= (survival_s + .1) / (kill_s + .1)
    stall = 0. if not retaliates else (1. if survival_s is None else survival_s/(survival_s+2.))
    return dict(can_hit=hit, can_be_targeted=retaliates, hits_to_kill=hits,
                kill_s=kill_s, survival_s=survival_s,
                attack_weight=attack, stall_weight=stall)


def card_evidence(kb, cid, level, enemies):
    roster = kb.roster(cid, level)
    from .unit_mechanics import mechanism_status, deployment_pulses
    return dict(knowledge_version=kb.version, assumed_level=level,
                source=kb.cards.get(cid, {}).get('metadata_source', kb.source),
                approximation='local_duel_screen_then_spatial_combat',
                special_mechanisms=mechanism_status(kb,cid,level),
                deployment_effects=deployment_pulses(kb,cid,level),
                spell=kb.spell(cid, level),
                units=[dict(name=s.name, count=n, hp=s.hp, shield=s.shield,
                            damage=s.damage, period_s=s.period, first_hit_s=s.first_hit,
                            reach=s.reach, speed=s.speed, deploy_s=s.deploy,
                            targets=list(s.targets), building_only=s.building_only,
                            air=s.air, building=s.building,
                            splash=s.splash, unsupported=list(s.unsupported),
                            matchups=[dict(enemy_uid=e.uid, enemy=e.spec.name,
                                           **matchup(s, e, n)) for e in enemies])
                       for s, n in roster])
