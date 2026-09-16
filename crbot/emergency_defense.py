"""Bounded database-based last resort when detailed search cannot finish."""
from __future__ import annotations

import math
from dataclasses import asdict

from .battle_simulation import SimAction
from .card_matchup import matchup


def emergency_defense(sim, state):
    """Consider only enemies already near a living home tower, including king.

    No learned score, reserve gate, grid enumeration or simulated damage claim.
    The router still enforces fresh hand, observation expiry and confirmation.
    """
    towers = [e for e in state.entities if e.side == 1 and e.tower and e.hp > 0]
    threats = []
    for enemy in state.entities:
        if enemy.side != -1 or enemy.tower or enemy.hp <= 0 or enemy.y < 16 or enemy.spec.damage <= 0:
            continue
        if 'ground' not in enemy.spec.targets:
            continue
        tower = min(towers, key=lambda t: math.dist((enemy.x, enemy.y), (t.x, t.y)), default=None)
        if tower is None:
            continue
        gap = max(0., sim.distance(enemy, tower)-enemy.spec.reach-enemy.spec.radius-tower.spec.radius)
        if gap <= max(2., enemy.spec.speed*1.2):
            threats.append((enemy, tower, gap))
    threats.sort(key=lambda v: (v[2], -v[0].spec.damage/max(.1, v[0].spec.period)))
    threats = threats[:12]
    audit = dict(active=bool(threats), mode='database_emergency_estimate',
                 completed_combat_comparison=False, global_best_claimed=False,
                 reason='no_near_tower_enemy', options=[], threats=[e.uid for e, _, _ in threats])
    if not threats:
        return None, audit
    best = None
    for slot, cid in state.hands[1]:
        cost = sim.kb.cards.get(cid, {}).get('elixir')
        if cost is None or cost > state.elixir[1]:
            continue
        roster = sim.kb.roster(cid, sim.level)
        spell = sim.kb.spell(cid, sim.level)
        # Special spell timing/line effects cannot safely use circular impact estimates.
        simple_spell = spell and not spell.get('friendly') and spell.get('damage', 0) > 0 and not spell.get('pattern')
        if not roster and not simple_spell:
            continue
        points = set()
        for enemy, _, _ in threats:
            points.add((enemy.x, enemy.y))
            for radius in sorted({1.2, 2.5, *(min(6., max(1.2, u.reach)) for u, _ in roster)}):
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    points.add((enemy.x+dx*radius, enemy.y+dy*radius))
        card_best = None
        for x, y in sorted(points):
            sx, sy = sim.screen(x, y)
            if not sim.legal_placement(state, cid, 1, sx, sy):
                continue
            value = 0.
            for enemy, tower, gap in threats:
                distance = math.dist((x, y), (enemy.x, enemy.y))
                weight = (enemy.spec.damage/max(.1, enemy.spec.period))/(1+gap)
                if simple_spell and (not enemy.spec.air or spell.get('air')) and distance <= spell['radius']+enemy.spec.radius:
                    value += weight * min(1., spell['damage']/max(1., enemy.hp+enemy.shield))/(1+spell['delay'])
                for spec, count in roster:
                    comparison = matchup(spec, enemy, count)
                    contact = max(0., distance-spec.reach-spec.radius-enemy.spec.radius)
                    # A stationary building with no target in range cannot intercept.
                    if contact > 0 and spec.speed <= 0:
                        continue
                    arrival = spec.deploy+spec.first_hit+contact/max(.1, spec.speed)
                    if not comparison['can_hit'] or spec.damage <= 0 or arrival > 3.:
                        continue
                    useful = min(1., spec.damage*count/max(1., enemy.hp+enemy.shield))
                    useful += comparison['attack_weight']
                    # Prefer range separation for fragile defenders over dropping
                    # them directly into the enemy's attack radius.
                    exposed = comparison['can_be_targeted'] and distance <= enemy.spec.reach+enemy.spec.radius+spec.radius
                    survival = comparison['survival_s']
                    if exposed and survival is not None:
                        useful *= .25 + .75*min(1., survival/max(.1, arrival+spec.period))
                    value += weight*useful/(1+arrival)
            if value <= 0:
                continue
            action = SimAction(cid, slot, sx, sy)
            rank = (value/max(1., cost)**.5, -cost, -slot, -sx, -sy)
            if card_best is None or rank > card_best[0]:
                card_best = (rank, action)
        if card_best:
            audit['options'].append(dict(action=asdict(card_best[1]), estimate=card_best[0][0], cost=cost))
            if best is None or card_best[0] > best[0]:
                best = card_best
    audit['reason'] = 'search_incomplete_near_tower_defense' if best else 'no_affordable_interacting_defense'
    if best:
        audit['selected'] = asdict(best[1])
    return (best[1] if best else None), audit
