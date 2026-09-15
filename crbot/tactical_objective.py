"""Shared attack/defense objective with explicit reserve and response hypotheses."""

def phase_for(state):
    enemies = [e for e in state.entities if e.side == -1 and not e.tower and e.hp > 0]
    if enemies:
        # Coordinates are calibrated arena tiles. Only a distant, slow approach
        # grants preparation time; bridge pressure or ongoing contact is defense.
        allies = [e for e in state.entities if e.side == 1 and not e.tower and e.hp > 0]
        distant = all(e.y < 12 and e.spec.speed > 0
                      and (16-e.y)/max(.1, e.spec.speed) >= 4 for e in enemies)
        contact = any((e.x-a.x)**2+(e.y-a.y)**2 <= (max(e.spec.reach,a.spec.reach)+2)**2
                      for e in enemies for a in allies)
        return 'prepare' if distant and not contact else 'defend'
    if any(e.side==1 and not e.tower and e.hp>0 for e in state.entities):return 'counterpush'
    return 'develop'


def avoid_overflow(best, candidates, world, kb, reserve=3., score_key='score', preparation=False):
    """Break repeated WAIT near cap using completed, conservative single-card roots.

    A receding two-card plan cannot promise that its deferred play will execute.
    Compare immediate roots instead; never fabricate a placement on timeout.
    """
    threshold = 7. if preparation else 9.5
    diagnostic = dict(active=world.elixir >= threshold, changed=False,
                      elixir=world.elixir, threshold=threshold, preparation=preparation, reason='below_threshold')
    if world.elixir < threshold:
        return best, diagnostic
    diagnostic['reason'] = 'already_playing'
    if best['action']['card_id'] is not None:
        return best, diagnostic
    diagnostic['reason'] = 'no_safe_development'
    waiting = next((r for r in candidates if r['action']['card_id'] is None), None)
    if waiting is None or not waiting.get('branches'):
        return best, diagnostic

    def loss(row):
        return max(-100000*b['own_towers_remaining'] + b['own_tower_damage']
                   + b.get('imminent_tower_exposure', 0) for b in row['branches'])

    # No extra tower damage is accepted just to spend elixir. Keep the safety
    # result of both the immediate WAIT and the selected conditional plan.
    ceiling = min(loss(waiting), loss(best))
    safe = []
    for row in candidates:
        action = row['action']; cid = action['card_id']
        if not cid or (action['slot'], cid) not in world.hand:
            continue
        card = kb.cards[cid]; cost = card.get('elixir')
        if cost is None or cost > world.elixir-reserve or not kb.roster(cid) or card.get('kind') in {'spell', 'building'}:
            continue
        if not row.get('branches') or loss(row) > ceiling + 1e-6:
            continue
        if max(b.get('deployment_exposure_penalty', 0) for b in row['branches']) > .25:
            continue
        # Reject strongly losing developments even if they do not lose a tower
        # inside this short simulation horizon.
        if row.get(score_key, row['score']) < waiting.get(score_key, waiting['score'])-3.:
            continue
        safe.append(row)
    if safe:
        best = min(safe, key=lambda r: (kb.cards[r['action']['card_id']]['elixir'],
                                       -r.get(score_key, r['score']), loss(r)))
        diagnostic.update(changed=True, reason='safe_preparation' if preparation else 'safe_immediate_development',
                          card_id=best['action']['card_id'])
    return best, diagnostic


def score_action(score,parts,world,kb,action,phase,reserve=3.,geometry=None):
    cost=float(kb.cards[action.card_id].get('elixir') or 0) if action.card_id else 0
    # A short horizon sees early bridge damage but misses the risk of committing
    # unsupported troops before the opponent responds. Charge that exposure;
    # do not ban aggressive placements or override measured tower mitigation.
    exposure = 0.
    if action.card_id and kb.roster(action.card_id) and kb.cards[action.card_id].get('kind') != 'building':
        from .arena_geometry import ArenaGeometry
        geometry = geometry or ArenaGeometry()
        px, py = geometry.xy(action.x, action.y)
        towers = [geometry.xy(*point) for i, point in enumerate(geometry.tower_points[:2])
                  if world.tower_health[i] is None or world.tower_health[i] > 0]
        if towers:
            tx, ty = min(towers, key=lambda p: (p[0]-px)**2 + (p[1]-py)**2)
            tower = kb.unit('PrincessTower')
            reach = tower.reach if tower else 0.
            forward = min(1., max(0., (ty-py-reach*.5)/max(1., ty-16.-reach*.5)))
            supported = any(t.side == 1 and not t.hypotheses and not t.card_id.startswith('unknown:') and t.confidence >= .8
                            and t.hp_fraction is not None and t.hp_fraction >= .35
                            and t.hp_observed_at is not None and world.at-t.hp_observed_at <= .6
                            and world.at-t.last_seen <= .6
                            and abs(geometry.xy(t.x,t.y)[0]-px) <= 3
                            and 0 <= py-geometry.xy(t.x,t.y)[1] <= 5
                            for t in world.tracks)
            exposure = forward * cost * (.15 if supported else .75)
    parts['deployment_exposure_penalty'] = round(exposure, 4)
    score -= exposure
    if phase=='defend':return score
    reserve_penalty=max(0,reserve-(world.elixir-cost))*1.5
    # Attack commitments must leave an answer; tower-finishing damage retains its large benefit.
    score-=reserve_penalty
    if action.card_id is None:
        score-=max(0,world.elixir-8)*.6
    if action.card_id:
        card=kb.cards[action.card_id]
        if card.get('kind')=='building':score-=3
        if 'win_condition' in card.get('roles',[]):score+=.3
        if phase=='develop' and world.elixir<reserve+cost and parts.get('enemy_tower_damage',0)<100:
            score-=2
    return score


def guard_development_reserve(candidates, world, kb, phase, reserve=3., *, urgent=False, level=11, immediate_candidates=None):
    """Reserve the cost of a remaining defense, unless this play mitigates tower loss."""
    diagnostic = dict(active=False, blocked=0, reserve=reserve, reason='wait_comparison_unavailable',
                      phase=phase, urgent=urgent, future_draw_assumed=False, options=[])
    immediate = candidates if immediate_candidates is None else immediate_candidates
    def action_key(row):
        a = row['action']
        return a['slot'], a['card_id'], a['x'], a['y']
    immediate_by_action = {action_key(r): r for r in immediate}
    waiting = next((r for r in immediate if r['action']['card_id'] is None), None)
    if waiting is None or not waiting.get('branches'):
        return candidates, diagnostic

    def tower_loss(row):
        return max(-100000*b.get('own_towers_remaining', 2)
                   + b.get('own_tower_damage', 0) + b.get('imminent_tower_exposure', 0)
                   for b in row['branches'])

    # A disposable cycle card is not a substitute for a sustained defender.
    # This is a conservative capability proxy, not a guarantee of winning a fight.
    defenders = []
    for slot, cid in world.hand:
        cost = kb.cards[cid].get('elixir')
        if cost is None:
            continue
        roster = [(spec, count) for spec, count in kb.roster(cid, level)
                  if not spec.building_only and spec.damage > 0 and spec.period > 0]
        layers = {layer for layer in ('ground', 'air')
                  if sum((spec.hp+spec.shield)*count for spec,count in roster
                         if layer in spec.targets) >= 300}
        if layers:
            defenders.append((slot, cid, float(cost), layers))

    required_by_slot = {}
    for played_slot, _ in world.hand:
        remaining = [d for d in defenders if d[0] != played_slot]
        coverable = set().union(*(d[3] for d in remaining)) if remaining else set()
        combinations = []
        for mask in range(1, 1 << len(remaining)):
            selected = [d for i,d in enumerate(remaining) if mask & (1 << i)]
            if set().union(*(d[3] for d in selected)) >= coverable:
                combinations.append((sum(d[2] for d in selected), [d[1] for d in selected]))
        cost, cards = min(combinations, default=(0., []), key=lambda item:item[0])
        required_by_slot[played_slot] = (max(reserve, cost), cards, sorted({'ground','air'}-coverable))

    waiting_loss = tower_loss(waiting)
    allowed = []
    for row in candidates:
        action = row['action']; cid = action['card_id']
        if not cid:
            allowed.append(row)
            continue
        cost = kb.cards[cid].get('elixir')
        required, cards, missing = required_by_slot.get(action['slot'], (reserve, [], ['ground','air']))
        after = world.elixir-cost if cost is not None else -1
        proof = immediate_by_action.get(action_key(row))
        # A promised future combo is not proof that spending the reserve now is safe.
        mitigation = bool(proof and proof.get('branches')) and tower_loss(proof) < waiting_loss-30
        accepted = after >= required or mitigation
        if accepted:
            allowed.append(row)
        diagnostic['options'].append(dict(card_id=cid, slot=action['slot'], x=action['x'], y=action['y'],
            elixir_after=round(after,3), required=required, retained_defenders=cards,
            uncovered_layers=missing, tower_mitigation_override=mitigation, accepted=accepted))
    diagnostic.update(active=True, blocked=len(candidates)-len(allowed),
                      reason='reserve_next_defense' if len(allowed)<len(candidates) else 'reserve_satisfied')
    return allowed, diagnostic
