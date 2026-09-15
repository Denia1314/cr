import unittest
from dataclasses import asdict
from unittest.mock import patch

from crbot.battle_simulation import SimAction
from crbot.battle_world import Track
from crbot.predictive_planner import PredictivePlanner, PlanResult
from tests.test_prediction import knowledge, world


def row(action, damage, score=0, follow=None):
    return dict(action=asdict(action), label=action.card_id or 'WAIT', score=score,
                branches=[dict(own_towers_remaining=2, own_tower_damage=damage,
                               enemy_response='WAIT', own_followup=follow or 'WAIT',
                               followup_action=asdict(SimAction(follow, 0)),
                               deployment_exposure_penalty=0)])


class UrgentTowerDefenseTests(unittest.TestCase):
    def setUp(self):
        self.p = PredictivePlanner(knowledge(), dict(unified_tactics=True, fast_defense=True, budget_ms=160))
        self.p.clock = lambda: 0.
        self.wait = SimAction()
        self.play = SimAction('knight', 0, .3, .70)
        self.w = world(elixir=10, hand=((0, 'knight'),),
                       tracks=(Track(1, 'giant', -1, .28, .71, 99, 100, .9),))

    def test_single_card_saving_tower_survives_combo_replacement(self):
        with patch.object(self.p, 'candidates', return_value=[self.wait, self.play]), \
             patch.object(self.p, 'evaluate_root', side_effect=lambda w,a,*args: row(a, 253 if a.card_id is None else 0)), \
             patch.object(self.p, 'refine_combinations', return_value=[row(self.wait,253),row(self.play,253,10)]):
            result = self.p.plan(self.w)
        self.assertEqual(result.action.card_id, 'knight')
        self.assertTrue(result.compute['urgent_tower_defense'])

    def test_urgent_wait_cannot_claim_unexecuted_future_defense(self):
        deferred = row(self.wait, 0, 10, 'knight')
        deferred['planned_cost'] = 3
        with patch.object(self.p, 'candidates', return_value=[self.wait,self.play]), \
             patch.object(self.p, 'evaluate_root', side_effect=lambda w,a,*args: row(a,253 if a.card_id is None else 20)), \
             patch.object(self.p, 'refine_combinations', return_value=[deferred]):
            result = self.p.plan(self.w)
        self.assertEqual(result.action.card_id,'knight')

    def test_urgent_first_round_uses_remaining_budget(self):
        elapsed = [0.]
        self.p.clock = lambda: elapsed[0]
        def evaluate(w,a,scenarios,horizon,phase,deadline,*args):
            elapsed[0] += .055
            if elapsed[0] >= deadline:
                raise TimeoutError()
            return row(a,253 if a.card_id is None else 0)
        with patch.object(self.p,'candidates',return_value=[self.wait,self.play]), \
             patch.object(self.p,'evaluate_root',side_effect=evaluate), \
             patch.object(self.p,'refine_combinations',return_value=[]):
            result=self.p.plan(self.w)
        self.assertEqual(result.status,'ready')
        self.assertGreater(result.elapsed_ms,80)

    def test_defensive_combo_shortlist_keeps_tower_saving_location(self):
        result=PlanResult(1,0,1,'ready',tactical_phase='defend')
        exposed=SimAction('knight',0,.3,.45)
        result.candidates=[row(exposed,253,10),row(self.play,0,0)]
        selected=[]
        def evaluate(w,a,*args):
            selected.append(a);return row(a,0)
        with patch.object(self.p,'evaluate_combination',side_effect=evaluate):
            self.p.refine_combinations(self.w,result,[],[(0,1,False)],8,1)
        self.assertEqual(selected,[self.play])

    def test_nonurgent_search_does_not_timeout_at_half_budget(self):
        elapsed=[0.];self.p.clock=lambda:elapsed[0]
        def evaluate(w,a,scenarios,horizon,phase,deadline,*args):
            elapsed[0]+=.055
            if elapsed[0]>=deadline:raise TimeoutError()
            return row(a,0,0 if a.card_id is None else 1)
        with patch.object(self.p,'candidates',return_value=[self.wait,self.play]), \
             patch.object(self.p,'evaluate_root',side_effect=evaluate), \
             patch.object(self.p,'refine_combinations',return_value=[]):
            result=self.p.plan(world(tracks=(),hand=((0,'knight'),),elixir=10))
        self.assertFalse(result.compute['urgent_tower_defense'])
        self.assertEqual(result.status,'ready')
        self.assertGreater(result.elapsed_ms,80)


class DefenseTimingTests(unittest.TestCase):
    def test_marching_hog_can_be_pulled_but_active_tower_attack_stays_locked(self):
        from crbot.battle_simulation import SimState
        from crbot.grid_world import target_for
        p=PredictivePlanner(knowledge(),{})
        state=SimState()
        tower=p.sim.add(state,p.kb.unit('PrincessTower'),1,4,28,tower=True)
        hog=p.sim.add(state,p.kb.roster('hog_rider')[0][0],-1,4,21)
        hog.target=tower.uid
        cannon=p.sim.add(state,p.kb.roster('cannon')[0][0],1,6,23)
        self.assertEqual(target_for(p.sim,state,hog).uid,cannon.uid)
        hog.x,hog.y=4,26.5
        hog.winding_target=tower.uid
        self.assertEqual(target_for(p.sim,state,hog).uid,tower.uid)

    def test_pipeline_is_simulated_before_play_without_extending_horizon(self):
        p=PredictivePlanner(knowledge(),dict(fast_defense=True,defense_pipeline_s=.35))
        calls=[]
        advance=p.sim.advance
        apply=p.sim.apply
        def step(state,seconds,**kwargs):
            calls.append(('advance',seconds))
            return advance(state,seconds,**kwargs)
        def play(state,action,side):
            calls.append(('apply',action.card_id))
            return apply(state,action,side)
        with patch.object(p.sim,'advance',side_effect=step),patch.object(p.sim,'apply',side_effect=play):
            result=p.evaluate_root(world(),SimAction('knight',0,.3,.65),[(0,1,False)],8,'defend',float('inf'))
        self.assertEqual(calls[0],('advance',.35))
        self.assertEqual(calls[1],('apply','knight'))
        self.assertAlmostEqual(sum(value for name,value in calls if name=='advance'),8)
        self.assertEqual(result['branches'][0]['pipeline_delay_s'],.35)

    def test_fast_hog_gets_earlier_intervention_and_pipeline_margin(self):
        from crbot.defense_timing import forecast
        from dataclasses import replace
        p=PredictivePlanner(knowledge(),{})
        s=p.initial(world(tracks=(Track(1,'hog_rider',-1,.28,.52,99,100,.9),)),7,1)
        f=forecast(p.sim,s,s.hands[1],.35)[0]
        enemy=next(e for e in s.entities if not e.tower)
        enemy.spec=replace(enemy.spec,speed=enemy.spec.speed/2)
        slower=forecast(p.sim,s,s.hands[1],.35)[0]
        self.assertLess(f['intervention_slack_s'],slower['intervention_slack_s'])
        self.assertGreater(f['defense_lead_s'],.35)
        self.assertLess(f['intervention_slack_s'],f['unopposed_tower_eta_s'])

    def test_ground_projection_uses_bridge_and_stops_at_attack_range(self):
        from crbot.defense_timing import project_approach
        p=PredictivePlanner(knowledge(),{})
        s=p.initial(world(tracks=(Track(1,'hog_rider',-1,.5,.4,99,100,.9),)),7,1)
        enemy=next(e for e in s.entities if not e.tower)
        tower=next(e for e in s.entities if e.side==1 and e.tower)
        point=project_approach(p.sim,s,enemy,tower,100)
        import math
        self.assertGreaterEqual(math.dist(point,(tower.x,tower.y)),tower.spec.radius+enemy.spec.reach-.3)
        self.assertNotEqual(point,(tower.x,tower.y))

    def test_complete_combo_survives_later_timeout(self):
        p=PredictivePlanner(knowledge(),{})
        a,b=SimAction('knight',0),SimAction('musketeer',1)
        r=PlanResult(1,0,1,'ready',tactical_phase='defend')
        r.candidates=[row(a,0),row(b,10)]
        with patch.object(p,'evaluate_combination',side_effect=[row(a,0),TimeoutError()]):
            finished=p.refine_combinations(world(),r,[],[(0,1,False)],8,1)
        self.assertEqual(len(finished),1)
        self.assertFalse(r.compute['combination_search']['complete'])
        self.assertEqual(r.compute['combination_search']['completed_roots'],1)

if __name__=='__main__':unittest.main()
