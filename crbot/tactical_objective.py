"""Shared attack/defense objective with explicit reserve and response hypotheses."""

def phase_for(state):
    if any(e.side==-1 and not e.tower and e.hp>0 for e in state.entities):return 'defend'
    if any(e.side==1 and not e.tower and e.hp>0 for e in state.entities):return 'counterpush'
    return 'develop'


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
