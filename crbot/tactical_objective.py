"""Shared attack/defense objective with explicit reserve and response hypotheses."""

def phase_for(state):
    if any(e.side==-1 and not e.tower and e.hp>0 for e in state.entities):return 'defend'
    if any(e.side==1 and not e.tower and e.hp>0 for e in state.entities):return 'counterpush'
    return 'develop'


def score_action(score,parts,world,kb,action,phase,reserve=3.):
    if phase=='defend':return score
    cost=float(kb.cards[action.card_id].get('elixir') or 0) if action.card_id else 0
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
