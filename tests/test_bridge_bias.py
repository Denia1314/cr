import unittest
from unittest.mock import patch
from dataclasses import replace

from crbot.arena_geometry import ArenaGeometry
from crbot.battle_simulation import SimAction
from crbot.battle_world import Track
from crbot.placement_search import placement_points
from crbot.predictive_planner import PredictivePlanner
from crbot.tactical_objective import score_action
from tests.test_prediction import knowledge, world


class BridgeBiasTests(unittest.TestCase):
    def setUp(self):
        self.kb = knowledge()
        self.p = PredictivePlanner(self.kb, dict(all_placement_points=True, unified_tactics=True,
                                                fast_defense=True, budget_ms=160))

    def test_first_three_locations_cover_front_middle_and_rear_without_dropping_grid(self):
        for side in (1, -1):
            for tracks in ((), (Track(1, 'giant', -side, .28, .5, 99, 100, .9),)):
                state = self.p.initial(world(tracks=tracks), 7, 1)
                self.p.sim.tactical_phase = 'develop' if not tracks else 'defend'
                for cid in ('knight', 'musketeer', 'cannon'):
                    points = placement_points(self.p.sim, state, cid, side, limit=None)
                    depths = [self.p.sim.xy(*point)[1] for point in points[:3]]
                    depths = [v if side == 1 else 32-v for v in depths]
                    self.assertEqual({0 if v < 21 else 1 if v < 26 else 2 for v in depths}, {0,1,2})
                    self.assertEqual(len(points), len(set(points)))

    def test_unopposed_development_does_not_rank_bridge_first(self):
        state = self.p.initial(world(tracks=()), 7, 1)
        self.p.sim.tactical_phase = 'develop'
        for cid in ('giant', 'musketeer', 'knight'):
            first = placement_points(self.p.sim, state, cid, 1, limit=None)[0]
            self.assertGreater(self.p.sim.xy(*first)[1], 21)

    def test_exposure_uses_calibration_and_observed_support(self):
        for geometry in (ArenaGeometry(), ArenaGeometry.from_config(dict(
                bounds=[.075,.097,.925,.757], tower_points=[[.24,.625],[.76,.625],[.24,.219],[.76,.219]]))):
            def score(y, snapshot):
                parts = {}
                action = SimAction('musketeer', 0, *geometry.screen(4,y))
                value = score_action(10, parts, snapshot, self.kb, action, 'develop', 3, geometry)
                return value, parts['deployment_exposure_penalty']
            snapshot = world(tracks=(), elixir=10)
            bridge, penalty = score(17, snapshot)
            rear, rear_penalty = score(28, snapshot)
            self.assertGreater(rear, bridge)
            self.assertGreater(penalty, rear_penalty)
            x,y = geometry.screen(4,16)
            ally = Track(2,'knight',1,x,y,99,100,.9,3,hp_fraction=.8,hp_observed_at=100,hp_confidence=.85)
            supported, _ = score(17, replace(snapshot, tracks=(ally,)))
            self.assertGreater(supported, bridge)
            stale, _ = score(17, replace(snapshot, tracks=(replace(ally,last_seen=98),)))
            self.assertEqual(stale, bridge)

    def test_timeout_before_spatial_comparison_does_not_publish_bridge_only_result(self):
        elapsed = [0.]
        self.p.clock = lambda: elapsed[0]
        original = self.p.sim.evaluate
        count = [0]
        def evaluate(state):
            count[0] += 1
            value = original(state)
            # WAIT and one position for each card finish; other depths have not.
            if count[0] == 5:
                elapsed[0] = 1.
            return value
        with patch.object(self.p.sim, 'evaluate', side_effect=evaluate):
            result = self.p.plan(world(tracks=()))
        self.assertEqual(result.status, 'timeout')
        self.assertEqual(result.candidates, [])

    def test_tower_saving_defense_outweighs_unsupported_forward_penalty(self):
        snapshot = world()
        bridge = SimAction('knight',0,*self.p.sim.screen(4,17))
        rear = SimAction('knight',0,*self.p.sim.screen(4,28))
        # Equivalent positions prefer cover; a meaningful mitigation still wins.
        covered = score_action(0, {}, snapshot, self.kb, rear, 'defend')
        saving = score_action(10, {}, snapshot, self.kb, bridge, 'defend')
        self.assertGreater(saving, covered)

    def test_full_elixir_quiet_board_develops_in_rear_instead_of_waiting_forever(self):
        planner = PredictivePlanner(self.kb, dict(unified_tactics=True, fast_defense=True, positions_per_card=3))
        planner.clock = lambda: 0.
        result = planner.plan(world(tracks=(), elixir=10, hand=((0,'giant'),(1,'musketeer'),(2,'knight'))))
        self.assertEqual(result.status, 'ready')
        self.assertGreater(planner.sim.xy(result.action.x,result.action.y)[1], 21)


if __name__ == '__main__':
    unittest.main()
