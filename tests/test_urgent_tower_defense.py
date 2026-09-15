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


if __name__=='__main__':unittest.main()
