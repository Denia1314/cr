"""Scene-conditioned placement shortlist, followed by the planner's combat search."""
from __future__ import annotations

import math


def screen(x, y):
    return round(.05 + x * .05,12), round(.18 + y * .02,12)


def spatial_rounds(ranked, side):
    """Visit front, middle and rear before spending the budget on adjacent cells."""
    groups = {}
    for row in ranked:
        depth = row[1][1] if side == 1 else 32-row[1][1]
        band = 0 if depth < 21 else 1 if depth < 26 else 2
        groups.setdefault(band, []).append(row)
    result = []
    # Preserve heuristic order between bands, and every point within each band.
    while groups:
        for band in list(groups):
            result.append(groups[band].pop(0))
            if not groups[band]:
                del groups[band]
    return result


def placement_points(sim, state, card_id, side, *, limit=12, quick=False):
    card = sim.kb.cards[card_id]
    roster = sim.kb.roster(card_id, sim.level)
    from .unit_mechanics import deployment_pulses, pulse, placement_special_bonus
    special_events=deployment_pulses(sim.kb,card_id,sim.level)
    for spec,count in roster:
        raw=sim.kb.units.get(spec.name,{})
        event=pulse(sim.kb,sim.kb.effects.get(raw.get('spawn_area_object')),sim.level)
        if event:special_events.extend([event]*count)
    spell = sim.kb.spell(card_id, sim.level)
    enemies = [e for e in state.entities if e.side != side and not e.tower and e.hp > 0]
    towers = [e for e in state.entities if e.side == side and e.tower and e.hp > 0]
    enemies.sort(key=lambda e: min((math.hypot(e.x-t.x, e.y-t.y) for t in towers), default=0))
    enemies = enemies[:24 if getattr(sim, "placement_batch", None) else 8]
    if spell:
        if not spell.get('friendly'):
            enemies += [e for e in state.entities if e.side != side and e.tower and e.hp > 0]
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
            return sum((e.spec.damage if spell.get('friendly') else min(e.hp + e.shield, spell['damage'] * (spell.get('tower_multiplier',1.) if e.tower else 1.))) * (1 + e.value) for e, q in zip(enemies, projected)
                       if (not e.spec.air or spell['air']) and math.dist(p, q) <= spell['radius'] + e.spec.radius)
    elif roster:
        # Cover the full conservative home deployment region, not a list of lane anchors.
        step = 4 if quick else (1 if getattr(sim, "placement_batch", None) else 2)
        points = [(float(x), float(y if side == 1 else 32-y))
                  for x in range(1, 18, step)
                  for y in range(17, 30, step)]
        projected = []
        for enemy in enemies:
            tower = min(towers, key=lambda t: math.hypot(enemy.x-t.x, enemy.y-t.y), default=None)
            if tower is not None:
                from .defense_timing import project_approach
                q=project_approach(sim,state,enemy,tower,max(u.deploy for u, _ in roster)+getattr(sim,'defense_pipeline_s',.35))
            else:
                q=(enemy.x,enemy.y)
            projected.append((enemy, q))
            # Continuous offsets around the predicted contact point enable precise interceptions.
            for radius in sorted({1., 2.5, *(max(1., u.reach) for u, _ in roster)}):
                for angle in range(0, 360, 45):
                    points.append((q[0]+radius*math.cos(math.radians(angle)),
                                   q[1]+radius*math.sin(math.radians(angle))))
            for t in towers:
                points.append(((q[0]+t.x)/2, (q[1]+t.y)/2))
        def unit_value(p, spec, count):
            from .card_matchup import matchup
            total = 0.
            for enemy, q in projected:
                can_hit = (not spec.building_only or enemy.spec.building) and ('air' if enemy.spec.air else 'ground') in spec.targets
                can_pull = (not enemy.spec.building_only or spec.building) and ('air' if spec.air else 'ground') in enemy.spec.targets
                distance = math.dist(p, q)
                if spec.building and enemy.spec.building_only:
                    edge=distance-spec.radius-enemy.spec.radius
                    tower_gap=min((math.dist(q,(t.x,t.y))-t.spec.radius-enemy.spec.radius for t in towers),default=32.)
                    can_pull=can_pull and edge<=enemy.spec.sight and edge<tower_gap
                gap = max(0, distance - spec.reach - spec.radius - enemy.spec.radius)
                contact = gap / max(.5, spec.speed + (enemy.spec.speed if can_pull else 0))
                urgency = 1 / (1 + min((max(0, math.dist(q, (t.x,t.y))-enemy.spec.reach-t.spec.radius) for t in towers), default=12)/5)
                threat = (1+enemy.value) * urgency
                # A pulled target moves toward the defender; use that contact location for tower cover.
                fight = ((p[0]+q[0])/2, (p[1]+q[1])/2) if can_pull else q
                tower_dps = sum(t.spec.damage/max(.1,t.spec.period) for t in towers
                    if t.active and ('air' if enemy.spec.air else 'ground') in t.spec.targets
                    and math.dist(fight, (t.x,t.y)) <= t.spec.reach+t.spec.radius+enemy.spec.radius)
                interception = math.exp(-contact/2)
                comparison = matchup(spec, enemy, count)
                total += threat * interception * ((2 + 2*comparison['attack_weight'] if can_hit else 0)
                    + (1 + comparison['stall_weight'] if can_pull else 0) + tower_dps/100)
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
        def value(p):
            return sum(unit_value(p, member, count)*count for member, count in roster) / sum(n for _, n in roster)
    else:
        return []
    if limit is None:
        from .placement_domain import deployment_domain
        points = [sim.xy(x,y) for x,y in deployment_domain(sim,state,card_id,side,step=getattr(sim,"placement_grid_step",1.))]
    ranked = []
    seen = set()
    for point in points:
        x, y = sim.screen(*point)
        if limit is None:
            x, y = round(x,12), round(y,12)
        key = (round(x, 4), round(y, 4))
        if key in seen or (limit is not None and not sim.legal_placement(state, card_id, side, x, y)):
            continue
        seen.add(key)
        ranked.append((0., point, (x,y)))
    batch = getattr(sim, "placement_batch", None)
    values = None
    if batch and ranked:
        positions = [r[1] for r in ranked]
        if spell and hasattr(batch, 'spell_score'):
            values = batch.spell_score(positions,enemies,projected,spell)
        elif roster and not enemies and hasattr(batch, 'quiet_score'):
            values = batch.quiet_score(positions,towers,side)
        elif roster:
            # Every constituent participates; e.g. a front unit cannot hide the
            # ranged/air-targeting members of a mixed card.
            scores = [batch.score(positions, member, projected, towers, count=count)
                      for member, count in roster] if hasattr(batch, 'roster_scoring') else [None]
            if all(v is not None for v in scores):
                values = [sum(v[i]*count for v, (_, count) in zip(scores, roster))/sum(n for _, n in roster)
                          for i in range(len(positions))]
    ranked = [((values[i] if values is not None else value(point))+
               (placement_special_bonus(point,roster,projected,special_events) if roster and not spell else 0), point, normalized)
              for i, (_, point, normalized) in enumerate(ranked)]
    ranked.sort(key=lambda r: r[0], reverse=True)
    if roster and not spell:
        ranked = spatial_rounds(ranked, side)
    if limit is None:
        return [normalized for _, _, normalized in ranked]
    # Keep alternatives spatially distinct so combat search compares genuinely different defenses.
    selected = []
    for _, point, normalized in ranked:
        if all(math.dist(point, other) >= 1.5 for other, _ in selected):
            selected.append((point, normalized))
        if len(selected) == limit:
            break
    return [normalized for _, normalized in selected]
