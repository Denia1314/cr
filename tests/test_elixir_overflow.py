import copy
import unittest
import json
from pathlib import Path
from PIL import Image, ImageDraw
from dataclasses import asdict
from unittest.mock import patch

from crbot.battle_simulation import SimAction
from crbot.battle_world import Track
from crbot.predictive_planner import PredictivePlanner
from crbot.tactical_objective import avoid_overflow
from crbot.vision import estimate_elixir
from tests.test_prediction import knowledge, world


class ElixirOverflowTests(unittest.TestCase):
    def test_calibrated_meter_excludes_side_padding_at_full_and_half(self):
        vision=json.loads((Path(__file__).resolve().parents[1]/'config.json').read_text(encoding='utf8'))['vision']
        for fraction, expected in ((1,10),(.5,5)):
            image=Image.new('RGB',(1080,1920),'black')
            # Physical strip inside the wider battle-UI region; its left padding
            # must not be treated as missing elixir.
            ImageDraw.Draw(image).rectangle((290,1857,290+round(759*fraction)-1,1898),fill=(190,30,210))
            value, confidence=estimate_elixir(image,vision['elixir_meter_roi'])
            self.assertAlmostEqual(value,expected,delta=.15)
            self.assertGreater(confidence,.8)
            if fraction==1:
                self.assertLess(estimate_elixir(image,vision['elixir_roi'])[0],9.5)

    def setUp(self):
        self.kb = knowledge()
        self.wait = self.row(None, 0)
        self.knight = self.row('knight', -1)
        self.snapshot = world(elixir=10)

    @staticmethod
    def row(cid, score):
        return dict(action=asdict(SimAction(cid, 0, .3, .72)), score=score,
                    branches=[dict(own_towers_remaining=2, own_tower_damage=0,
                                   deployment_exposure_penalty=0)])

    def test_cap_breaks_wait_but_normal_elixir_keeps_saving(self):
        for elixir, expected in ((7, None), (9.49, None), (9.5, 'knight'), (10, 'knight')):
            best, diagnostic = avoid_overflow(self.wait, [self.wait, self.knight],
                                             world(elixir=elixir), self.kb)
            self.assertEqual(best['action']['card_id'], expected)
            self.assertEqual(diagnostic['changed'], expected is not None)

    def test_does_not_spend_on_damage_exposure_or_losing_development(self):
        for field, value in (('own_tower_damage', 1), ('own_towers_remaining', 1),
                             ('imminent_tower_exposure', 1), ('deployment_exposure_penalty', .3)):
            bad = copy.deepcopy(self.knight); bad['branches'][0][field] = value
            best, _ = avoid_overflow(self.wait, [self.wait, bad], self.snapshot, self.kb)
            self.assertIs(best, self.wait)
        bad = self.row('knight', -4)
        self.assertIs(avoid_overflow(self.wait, [self.wait, bad], self.snapshot, self.kb)[0], self.wait)

    def test_preserves_reserve_and_does_not_cycle_spells_buildings(self):
        for cid in ('fireball', 'cannon', 'mega_knight'):
            row = self.row(cid, 0)
            snapshot = world(elixir=9.5, hand=((0, cid),))
            self.assertIs(avoid_overflow(self.wait, [self.wait, row], snapshot, self.kb)[0], self.wait)

    def test_conditional_wait_safety_and_existing_defense_are_preserved(self):
        waiting = copy.deepcopy(self.wait); waiting['branches'][0]['own_tower_damage'] = 100
        play = copy.deepcopy(self.knight); play['branches'][0]['own_tower_damage'] = 50
        # A deferred defensive follow-up saves the tower; do not replace it with
        # an immediate development that is only better than doing nothing.
        self.assertIs(avoid_overflow(self.wait, [waiting, play], self.snapshot, self.kb)[0], self.wait)
        self.assertIs(avoid_overflow(play, [waiting, self.knight], self.snapshot, self.kb)[0], play)

    def test_planner_breaks_repeated_defensive_wait_at_cap(self):
        planner = PredictivePlanner(self.kb, dict(unified_tactics=True, fast_defense=True, positions_per_card=3))
        planner.clock = lambda: 0.
        snapshot = world(elixir=10, tracks=(Track(1, 'giant', -1, .28, .30, 99, 100, .9, 3),),
                         hand=((0, 'valkyrie'), (1, 'mega_knight'), (2, 'elite_barbarians'), (3, 'ice_golem')))
        self.addCleanup(patch.stopall)
        patch('crbot.tactical_objective.phase_for', return_value='defend').start()
        with patch('crbot.tactical_objective.avoid_overflow', side_effect=lambda best, *a: (best, {'changed': False})):
            previous = planner.plan(snapshot)
        self.assertEqual(previous.status, 'wait')
        for _ in range(2):
            result = planner.plan(snapshot)
            self.assertEqual(result.status, 'ready')
            self.assertEqual(result.action.card_id, 'ice_golem')
            self.assertGreater(planner.sim.xy(result.action.x, result.action.y)[1], 21)
            self.assertTrue(result.compute['elixir_overflow']['changed'])


if __name__ == '__main__':
    unittest.main()
