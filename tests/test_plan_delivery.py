import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image
from crbot.engine import BotEngine
from crbot.frame_stream import Frame
from crbot.perception_stream import PreparedFrame
from crbot.battle_perception import HandCardMatch, LaneThreat
from crbot.policy import BattleDecision, BattlePolicy
from tests.test_prediction import RouterTests


class FreshBattleFrameTests(unittest.TestCase):
    def setUp(self):
        self.engine = object.__new__(BotEngine)
        self.old = Image.new('RGB', (20, 20))
        self.new = Image.new('RGB', (20, 20), 'red')
        self.engine._frame_sequence = 1
        self.engine.frame_stream = SimpleNamespace(latest=lambda: Frame(self.new, 99.9, 99.95, 2))
        self.engine.config = {'automation': {'battle_ui_roi': [0, 0, 1, 1]}}
        self.engine.policy = SimpleNamespace()

    def check(self, battle=.9, confirm=None, chest=None):
        with patch('crbot.engine.battle_ui_score', return_value=battle), \
             patch('crbot.engine.find_result_confirm_button', return_value=(confirm, 1)), \
             patch('crbot.engine.find_chest_open_screen', return_value=(chest, 1)):
            return self.engine._fresh_battle_frame(self.old)

    def test_navigation_aged_frame_is_replaced_with_actual_new_timestamp(self):
        self.assertIs(self.check(), self.new)
        self.assertEqual(self.engine._last_battle_capture, (self.new, 99.9))
        self.assertEqual(self.engine._frame_sequence, 2)

    def test_screen_transition_never_reuses_old_battle_image(self):
        for args in ({'battle': .5}, {'confirm': [.5, .8]}, {'chest': [.5, .5]}):
            with self.subTest(args=args):
                self.assertIsNone(self.check(**args))
                self.assertEqual(self.engine._frame_sequence, 1)

    def test_only_recent_new_prepared_frame_is_reused(self):
        prepared_image = Image.new('RGB', (20, 20), 'blue')
        for started, expected in ((99.8, prepared_image), (98, self.new)):
            with self.subTest(started=started):
                self.engine._frame_sequence = 1
                prepared = PreparedFrame(Frame(prepared_image, started, started+.02, 2), (), (), 99.9)
                self.engine.perception_stream = SimpleNamespace(latest=lambda: prepared)
                with patch('crbot.engine.time.monotonic', return_value=100):
                    self.assertIs(self.check(), expected)

    def test_prepared_frame_errors_fail_closed(self):
        self.engine.perception_stream = SimpleNamespace(latest=Mock(side_effect=RuntimeError('capture failed')))
        with self.assertRaises(RuntimeError):
            self.check()


class PlanDeliveryTests(unittest.TestCase):
    def test_final_lane_guard_reuses_badges_without_detector_or_track_mutation(self):
        policy = object.__new__(BattlePolicy)
        image, previous = Image.new('RGB', (20, 20)), Image.new('RGB', (20, 20))
        policy.prepared_frame = PreparedFrame(Frame(image, 100, 100, 2), (), ('badge',), 100)
        policy._threat_observed_at = 99.8
        policy._observed_image = previous
        policy.learned_detector = Mock()
        with patch('crbot.policy.detect_lane_threats', return_value={'left': 'visible'}) as detect:
            self.assertEqual(policy.recheck_lane_threats(image, previous, 100), {'left': 'visible'})
        self.assertEqual(detect.call_args.args[2], ())
        self.assertEqual(detect.call_args.kwargs['candidates'], ('badge',))
        policy.learned_detector.detect.assert_not_called()
        self.assertIs(policy._observed_image, previous)

    def test_recheck_evidence_is_reused_and_image_encoding_is_off_click_path(self):
        engine = object.__new__(BotEngine)
        old, current = Image.new('RGB', (20, 20)), Image.new('RGB', (20, 20))
        clock = [100.]
        def recognize(image):
            clock[0] += .4
            return [HandCardMatch(0, 'knight', .9, 60, 1, 100, False)]
        def threats(*args):
            clock[0] += .3
            return {'left': LaneThreat('left', .5, 1, .6, 'heavy', (), ())}
        def record(event, image, payload):
            if image is not None:
                clock[0] += .9
            return {}
        engine.policy = SimpleNamespace(hand_recognizer=SimpleNamespace(recognize=Mock(side_effect=recognize)),
            _perceive_threats=threats, prepare_action=lambda d, s: d, resolve_action=Mock())
        engine.recorder = SimpleNamespace(record=Mock(side_effect=record))
        engine.frame_stream = SimpleNamespace(latest=lambda: Frame(current, 100, 100, 2))
        engine._frame_sequence = 1
        engine.config = {'vision': {'elixir_roi': [0, 0, 1, 1]}}
        engine.dry_run = True
        engine.response_timing = Mock()
        decision = BattleDecision(0, [.2, .9], [.3, .6], 'left', 'test', 5, 'vision', 0, 0,
                                  card_id='knight', card_cost=3, decision_engine='predictive',
                                  action_id='test', plan_valid_until=101, left_threat=.5)
        with patch('crbot.engine.time.monotonic', side_effect=lambda: clock[0]), \
             patch('crbot.engine.estimate_elixir', return_value=(5, .9)) as elixir:
            result = engine._execute_action(old, decision, {})
        self.assertIsNone(result[-1])
        self.assertEqual(result[0].action_status, 'unknown')  # dry run is not a success claim
        self.assertLess(clock[0], 101)
        engine.policy.hand_recognizer.recognize.assert_called_once_with(current)
        elixir.assert_called_once()
        self.assertIsNone(engine.recorder.record.call_args.args[1])


class RouterAgeTests(unittest.TestCase):
    setUp = RouterTests.setUp
    result = RouterTests.result

    def test_perception_before_router_counts_and_requests_bounded_fresh_recovery(self):
        self.p.prediction_pre_decision_s = 1.2
        with patch.object(self.p.planner, 'plan', return_value=self.result()) as plan:
            decision = self.p.decide(self.image, None, now=100)
        self.assertIsNone(decision)
        self.assertEqual(self.p.last_plan['fallback_reason'], 'plan_expired_recapture')
        self.assertTrue(self.p.planner._urgent_recovery)
        self.assertGreaterEqual(plan.call_args.args[0].observation_delay_s, 1.2)
        self.assertEqual(self.p.action_sequence, 0)


if __name__ == '__main__':
    unittest.main()
