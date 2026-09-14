"""Scene-conditioned placement shortlist, followed by the planner's combat search."""
from __future__ import annotations

import math


def screen(x, y):
    return .05 + x * .05, .18 + y * .02


def placement_points(sim, state, card_id, side, *, limit=12):
    card = sim.kb.cards[card_id]
    roster = sim.kb.roster(card_id, sim.level)
    spell = sim.kb.spell(card_id, sim.level)
    enemies = [e for e in state.entities if e.side != side and not e.tower and e.hp > 0]
    towers = [e for e in state.entities if e.side == side and e.tower and e.hp > 0]
    enemies.sort(key=lambda e: min((math.hypot(e.x-t.x, e.y-t.y) for t in towers), default=0))
    enemies = enemies[:8]
    if spell:
        if spell.get('friendly'):
            enemies = [e for e in state.entities if e.side == side and not e.tower and not e.spec.building and e.hp>0][:8]
        # Center on groups as well as individuals; project through spell arrival delay.
        projected = [(e.x, max(0, min(32, e.y + side * e.spec.speed * spell['delay']))) for e in enemies]
        points = list(projected)
        for i, a in enumerate(projected):
            for b in projected[i+1:]:
                if math.dist(a, b) <= 2 * spell['radius']:
                    points.append(((a[0]+b[0])/2, (a[1]+b[1])/2))
        def value(p):
            return sum((e.spec.damage if spell.get('friendly') else min(e.hp, spell['damage'])) * (1 + e.value) for e, q in zip(enemies, projected)
                       if (not e.spec.air or spell['air']) and math.dist(p, q) <= spell['radius'] + e.spec.radius)
    elif roster:
        spec = roster[0][0]
        # Cover the full conservative home deployment region, not a list of lane anchors.
        points = [(float(x), float(y if side == 1 else 32-y))
                  for x in range(1, 18, 2) for y in range(17, 30, 2)]
        projected = []
        for enemy in enemies:
            tower = min(towers, key=lambda t: math.hypot(enemy.x-t.x, enemy.y-t.y), default=None)
            tx, ty = (tower.x, tower.y) if tower else (enemy.x, 32 if side == 1 else 0)
            distance = max(.01, math.hypot(tx-enemy.x, ty-enemy.y))
            travel = min(distance, enemy.spec.speed * (spec.deploy + spec.first_hit))
            q = (enemy.x + (tx-enemy.x)*travel/distance, enemy.y + (ty-enemy.y)*travel/distance)
            if enemy.observed_vx or enemy.observed_vy:
                window=min(.75,spec.deploy+spec.first_hit)
                portion=window/max(.1,spec.deploy+spec.first_hit)
                q=(max(0,min(18,q[0]+(enemy.observed_vx*window-(q[0]-enemy.x)*portion)*.5)),
                   max(0,min(32,q[1]+(enemy.observed_vy*window-(q[1]-enemy.y)*portion)*.5)))
            projected.append((enemy, q))
            # Continuous offsets around the predicted contact point enable precise interceptions.
            for radius in (1., 2.5, max(1., spec.reach)):
                for angle in range(0, 360, 45):
                    points.append((q[0]+radius*math.cos(math.radians(angle)),
                                   q[1]+radius*math.sin(math.radians(angle))))
            for t in towers:
                points.append(((q[0]+t.x)/2, (q[1]+t.y)/2))
        def value(p):
            total = 0.
            for enemy, q in projected:
                can_hit = ('air' if enemy.spec.air else 'ground') in spec.targets
                can_pull = (not enemy.spec.building_only or spec.building) and ('air' if spec.air else 'ground') in enemy.spec.targets
                distance = math.dist(p, q)
                gap = max(0, distance - spec.reach - spec.radius - enemy.spec.radius)
                contact = gap / max(.5, spec.speed + (enemy.spec.speed if can_pull else 0))
                urgency = 1 / (1 + min((max(0, math.dist(q, (t.x,t.y))-enemy.spec.reach-t.spec.radius) for t in towers), default=12)/5)
                threat = (1+enemy.value) * urgency
                # A pulled target moves toward the defender; use that contact location for tower cover.
                fight = ((p[0]+q[0])/2, (p[1]+q[1])/2) if can_pull else q
                tower_dps = sum(t.spec.damage/max(.1,t.spec.period) for t in towers
                    if ('air' if enemy.spec.air else 'ground') in t.spec.targets
                    and math.dist(fight, (t.x,t.y)) <= t.spec.reach+t.spec.radius+enemy.spec.radius)
                interception = math.exp(-contact/2)
                total += threat * interception * ((2 if can_hit else 0) + (1 if can_pull else 0) + tower_dps/100)
                if can_hit:
                    total -= threat * abs(distance - min(4., spec.reach) * .85) * .15
                # Ranged units should use reach; do not drop fragile support on top of attackers.
                if can_hit and spec.reach > 2 and can_pull:
                    total -= threat * max(0, min(spec.reach, 4)-distance) * .8
                if not can_hit and not can_pull:
                    total -= threat
            if not enemies:
                # Quiet-board development also follows the surviving towers and existing formation.
                total -= min((math.dist(p, (t.x,t.y-side*2)) for t in towers), default=abs(p[0]-9))*.2
            return total
    else:
        return []
    ranked = []
    seen = set()
    for point in points:
        x, y = screen(*point)
        key = (round(x, 4), round(y, 4))
        if key in seen or not sim.legal_placement(state, card_id, side, x, y):
            continue
        seen.add(key)
        ranked.append((value(point), point, (x,y)))
    ranked.sort(key=lambda r: r[0], reverse=True)
    # Keep alternatives spatially distinct so combat search compares genuinely different defenses.
    selected = []
    for _, point, normalized in ranked:
        if all(math.dist(point, other) >= 1.5 for other, _ in selected):
            selected.append((point, normalized))
        if len(selected) == limit:
            break
    return [normalized for _, normalized in selected]
