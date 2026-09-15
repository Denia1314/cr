import unittest
from crbot.placement_domain import deployment_domain
from crbot.predictive_planner import PredictivePlanner
from tests.test_prediction import knowledge,world


class DeploymentDomainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.planner=PredictivePlanner(knowledge(),{})
        cls.state=cls.planner.initial(world(),7,1)

    def test_spell_includes_every_half_tile_and_all_boundaries(self):
        points=deployment_domain(self.planner.sim,self.state,'fireball',1)
        self.assertEqual(len(points),37*65)
        for corner in ((.05,.18),(.95,.18),(.05,.82),(.95,.82)):
            self.assertIn(corner,points)

    def test_troops_include_every_legal_half_tile_without_spacing_filter(self):
        sim=self.planner.sim
        points=set(deployment_domain(sim,self.state,'knight',1))
        for x in range(37):
            for y in range(65):
                p=(round(.05+x*.025,12),round(.18+y*.01,12))
                self.assertEqual(p in points,sim.legal_placement(self.state,'knight',1,*p))
        self.assertGreater(len(points),900)

    def test_pixel_domain_enumerates_every_legal_pixel(self):
        sim=self.planner.sim
        points=set(deployment_domain(sim,self.state,'royal_delivery',1,image_size=(100,200)))
        expected={(x/100,y/200) for x in range(100) for y in range(200)
                  if sim.legal_placement(self.state,'royal_delivery',1,x/100,y/200)}
        self.assertEqual(points,expected)

    def test_planner_passes_entire_grid_to_combat_candidates(self):
        from collections import Counter
        p=PredictivePlanner(self.planner.kb,dict(all_placement_points=True,placement_grid_step=1))
        state=p.initial(world(),7,1)
        actions=p.candidates(state,1,limit=None)
        counts=Counter(a.card_id for a in actions)
        for slot,cid in state.hands[1]:
            expected=set(deployment_domain(p.sim,state,cid,1,step=1))
            actual={(a.x,a.y) for a in actions if a.card_id==cid}
            self.assertEqual(actual,expected)
            self.assertGreater(counts[cid],24)

    def test_complete_spatial_coverage_is_not_reported_as_complete_combat(self):
        p=PredictivePlanner(self.planner.kb,dict(all_placement_points=True,fast_defense=True,budget_ms=10))
        result=p.plan(world())
        self.assertGreater(result.compute['spatial_scored'],1000)
        self.assertFalse(result.compute['combat_complete'])
        self.assertGreater(result.compute['spatial_scored'],result.compute['combat_evaluated'])

    def test_full_grid_keeps_first_fair_round_when_refinement_times_out(self):
        from unittest.mock import patch
        from crbot.battle_simulation import SimAction
        from tests.test_urgent_tower_defense import row
        p=PredictivePlanner(self.planner.kb,dict(all_placement_points=True,fast_defense=True,budget_ms=160))
        p.clock=lambda:0
        roots=[SimAction(),SimAction('knight',0,.3,.65),SimAction('musketeer',1,.3,.68),
               SimAction('knight',0,.3,.7),SimAction('musketeer',1,.3,.72)]
        calls=[]
        def evaluate(w,a,*args):
            calls.append(a)
            if len(calls)>3:raise TimeoutError()
            return row(a,253 if a.card_id is None else 0)
        with patch.object(p,'candidates',return_value=roots),patch.object(p,'evaluate_root',side_effect=evaluate):
            result=p.plan(world(hand=((0,'knight'),(1,'musketeer'))))
        self.assertEqual(result.status,'ready')
        self.assertTrue(result.budget_exhausted)
        self.assertEqual(len(result.candidates),3)
        self.assertEqual({r['action']['card_id'] for r in result.candidates},{None,'knight','musketeer'})
        self.assertEqual(result.compute['completed_fair_rounds'],1)
        self.assertFalse(result.compute['combat_complete'])

    def test_incomplete_hand_comparison_cannot_be_called_usable(self):
        from unittest.mock import patch
        from crbot.battle_simulation import SimAction
        from tests.test_urgent_tower_defense import row
        p=PredictivePlanner(self.planner.kb,dict(all_placement_points=True,fast_defense=True))
        p.clock=lambda:0
        roots=[SimAction(),SimAction('knight',0,.3,.65),SimAction('musketeer',1,.3,.68)]
        def evaluate(w,a,*args):
            if a.card_id=='musketeer':raise TimeoutError()
            return row(a,253 if a.card_id is None else 0)
        with patch.object(p,'candidates',return_value=roots),patch.object(p,'evaluate_root',side_effect=evaluate):
            result=p.plan(world(hand=((0,'knight'),(1,'musketeer'))))
        self.assertEqual(result.status,'timeout')
        self.assertFalse(result.compute['usable_comparison'])
