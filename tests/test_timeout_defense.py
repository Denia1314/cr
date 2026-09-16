import unittest
from dataclasses import replace
from unittest.mock import patch

from crbot.battle_world import Track
from crbot.emergency_defense import emergency_defense
from crbot.predictive_planner import PredictivePlanner
from tests.test_prediction import knowledge, world, RouterTests


class TimeoutDefenseTests(unittest.TestCase):
    def setUp(self):
        self.p = PredictivePlanner(knowledge(), dict(fast_defense=True, budget_ms=160, king_towers=True))
        self.w = world(elixir=10, tracks=(Track(1, 'hog_rider', -1, .28, .71, 99, 100, .9),))

    def choose(self, w):
        s = self.p.initial(w, 10, 1)
        return emergency_defense(self.p.sim, s)

    def test_full_elixir_timeout_returns_legal_defense_without_fake_branches(self):
        with patch.object(self.p, 'candidates', side_effect=TimeoutError):
            r = self.p.plan(self.w)
        self.assertEqual(r.status, 'ready')
        self.assertIn((r.action.slot, r.action.card_id), self.w.hand)
        self.assertTrue(r.compute['timeout_tower_defense']['active'])
        self.assertFalse(r.compute['timeout_tower_defense']['completed_combat_comparison'])
        self.assertFalse(r.candidates)
        self.assertTrue(self.p.sim.legal_placement(self.p.initial(self.w, 10, 1), r.action.card_id, 1, r.action.x, r.action.y))

    def test_next_fresh_frame_bypasses_grid_that_exhausted_budget(self):
        with patch.object(self.p, 'candidates', side_effect=TimeoutError) as search:
            self.p.plan(self.w)
            result = self.p.plan(replace(self.w, revision=2, at=100.1))
        self.assertEqual(search.call_count, 1)
        self.assertEqual(result.status, 'ready')
        self.assertTrue(result.compute['fresh_frame_recovery'])

    def test_king_is_defended_after_both_princess_towers_destroyed(self):
        w = replace(self.w, tower_health=(0, 0, 1, 1, .6, 1),
                    tracks=(Track(1, 'hog_rider', -1, .44, .76, 99, 100, .9),))
        action, audit = self.choose(w)
        self.assertIsNotNone(action)
        from crbot.defense_timing import forecast
        timing = forecast(self.p.sim, self.p.initial(w, 10, 1), w.hand)
        self.assertTrue(timing)
        self.assertLess(timing[0]['intervention_slack_s'], 0)

    def test_low_elixir_never_invents_affordability(self):
        action, audit = self.choose(replace(self.w, elixir=0))
        self.assertIsNone(action)
        self.assertEqual(audit['reason'], 'no_affordable_interacting_defense')

    def test_ground_only_and_building_seekers_do_not_defend_air(self):
        w = replace(self.w, hand=((0, 'knight'), (1, 'giant')),
                    tracks=(Track(1, 'balloon', -1, .28, .71, 99, 100, .9),))
        self.assertIsNone(self.choose(w)[0])
        w = replace(w, hand=w.hand+((2, 'musketeer'),))
        self.assertEqual(self.choose(w)[0].card_id, 'musketeer')

    def test_full_elixir_alone_does_not_trigger_emergency(self):
        w = replace(self.w, tracks=(Track(1, 'hog_rider', -1, .28, .25, 99, 100, .9),))
        self.assertIsNone(self.choose(w)[0])
        self.assertIsNone(self.choose(replace(self.w, tracks=()))[0])

    def test_stale_or_dead_tracks_are_not_emergencies(self):
        for track in (replace(self.w.tracks[0], last_seen=97), replace(self.w.tracks[0], hp_fraction=0)):
            self.assertIsNone(self.choose(replace(self.w, tracks=(track,)))[0])

    def test_nonurgent_recovery_caps_candidates_instead_of_repeating_full_grid(self):
        self.p.config['all_placement_points'] = True
        s = self.p.initial(replace(self.w, tracks=()), 10, 1)
        options = self.p.candidates(s, 1, limit=None, bounded=True)
        self.assertLessEqual(len(options), 1+2*len(self.w.hand))
        self.assertEqual({a.card_id for a in options if a.card_id}, {cid for _, cid in self.w.hand})


class TimeoutRouterTests(unittest.TestCase):
    setUp = RouterTests.setUp

    def test_emergency_reaches_action_and_uses_confirmation_protocol(self):
        self.p.learned_detector.observed_enemies[0].update(y=.71)
        self.p._last_elixir_estimate_value = 10
        snapshot = self.p.snapshot_state()
        with patch.object(self.p.planner, 'candidates', side_effect=TimeoutError):
            decision = self.p.decide(self.image, None, now=100)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.card_id, 'knight')
        prepared = self.p.prepare_action(decision, snapshot)
        self.assertIsNotNone(self.p.pending_action_id)
        self.assertIsNone(self.p.decide(self.image, None, now=100.1))
        self.p.resolve_action(prepared.action_id, 'rejected', now=100.2)
        self.assertIsNone(self.p.pending_action_id)
        self.assertFalse(self.p.world.confirmed)

    def test_expired_emergency_still_requires_recapture(self):
        self.p.learned_detector.observed_enemies[0].update(y=.71)
        self.p.prediction_frame_age_s = 5
        with patch.object(self.p.planner, 'candidates', side_effect=TimeoutError):
            decision = self.p.decide(self.image, None, now=100)
        self.assertIsNone(decision)
        self.assertEqual(self.p.last_plan['fallback_reason'], 'plan_expired_recapture')
        self.assertEqual(self.p.action_sequence, 0)


if __name__ == '__main__':
    unittest.main()
