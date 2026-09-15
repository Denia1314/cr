import unittest
from dataclasses import replace
from unittest.mock import patch

from crbot.battle_world import Track
from crbot.predictive_planner import PredictivePlanner
from crbot.tactical_objective import phase_for
from tests.test_prediction import knowledge, world


class EnemyBackDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.planner = PredictivePlanner(knowledge(), dict(unified_tactics=True, fast_defense=True, positions_per_card=3))
        self.planner.clock = lambda: 0.
        self.snapshot = world(elixir=7, tracks=(Track(1, 'giant', -1, .28, .30, 99, 100, .9, 3),),
                              hand=((0, 'valkyrie'), (1, 'mega_knight'), (2, 'elite_barbarians'), (3, 'ice_golem')))

    def test_seven_elixir_prepares_instead_of_waiting_for_cap(self):
        with patch('crbot.tactical_objective.phase_for', return_value='defend'):
            old = self.planner.plan(self.snapshot)
        self.assertEqual(old.status, 'wait')
        result = self.planner.plan(self.snapshot)
        self.assertEqual(result.tactical_phase, 'prepare')
        self.assertEqual(result.status, 'ready')
        self.assertGreater(self.planner.sim.xy(result.action.x, result.action.y)[1], 21)
        self.assertGreaterEqual(7-self.planner.kb.cards[result.action.card_id]['elixir'], 3)
        self.assertEqual(result.compute['elixir_overflow']['reason'], 'safe_preparation')

    def test_low_elixir_still_waits(self):
        result = self.planner.plan(replace(self.snapshot, elixir=5))
        self.assertEqual(result.status, 'wait')

    def test_bridge_fast_approach_and_contact_still_defend(self):
        state = self.planner.initial(self.snapshot, 7, 1)
        enemy = next(e for e in state.entities if not e.tower)
        self.assertEqual(phase_for(state), 'prepare')
        enemy.y = 13
        self.assertEqual(phase_for(state), 'defend')
        enemy.y = 10
        enemy.spec = replace(enemy.spec, speed=4)
        self.assertEqual(phase_for(state), 'defend')
        enemy.spec = replace(enemy.spec, speed=1)
        state.entities.append(replace(enemy, uid=999, side=1, y=11))
        self.assertEqual(phase_for(state), 'defend')

    def test_another_lane_threat_blocks_preparation(self):
        tracks = self.snapshot.tracks + (Track(2, 'giant', -1, .72, .55, 99, 100, .9, 3),)
        result = self.planner.plan(replace(self.snapshot, tracks=tracks))
        self.assertEqual(result.tactical_phase, 'defend')
        self.assertFalse(result.compute['elixir_overflow']['preparation'])


if __name__ == '__main__':
    unittest.main()
