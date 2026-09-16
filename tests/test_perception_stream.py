import time
import unittest
import queue
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch
from PIL import Image

from crbot.frame_stream import Frame, LatestFrameStream
from crbot.perception_stream import PerceptionStream, PreparedFrame
from crbot.engine import BotEngine
from crbot.battle_perception import detect_lane_threats
from crbot.policy import BattlePolicy


class PerceptionTests(unittest.TestCase):
    def test_preview_uses_capture_even_when_decision_frame_has_not_changed(self):
        from crbot.gui import RoyalTrainerApp
        old, fresh = Image.new('RGB', (20, 20)), Image.new('RGB', (20, 20), 'red')
        engine = SimpleNamespace(completed_battles=0, in_battle=True, latest_frame=old,
                                 frame_stream=SimpleNamespace(latest=lambda: Frame(fresh, 1, 1.01, 4)))
        app = SimpleNamespace(closing=False, messages=queue.Queue(), engine=engine,
                              _bot_is_running=lambda: True, battle_value=Mock(), battle_note=Mock(),
                              last_frame_identity=id(old), _show_image=Mock(), root=Mock(), _poll_messages=Mock())
        RoyalTrainerApp._poll_messages(app)
        app._show_image.assert_called_once_with(fresh)
        self.assertGreaterEqual(app.root.after.call_args.args[0], 1)
        self.assertLessEqual(app.root.after.call_args.args[0], 34)

    def test_policy_reuses_only_the_exact_prepared_image(self):
        policy = object.__new__(BattlePolicy)
        image = Image.new('RGB', (20, 20))
        policy.hand_recognizer = SimpleNamespace(recognize=Mock(return_value=['fresh']))
        policy.hand_history = SimpleNamespace(update=Mock())
        policy.temporal_perception_enabled = False
        policy.prepared_frame = PreparedFrame(Frame(image, 1, 1.01, 1), ('cached',), (), 1.02)
        self.assertEqual(policy._stable_hand_matches(image, 2), ['cached'])
        policy.hand_recognizer.recognize.assert_not_called()
        self.assertEqual(policy._stable_hand_matches(image.copy(), 3), ['fresh'])
        policy.hand_recognizer.recognize.assert_called_once()

    def test_recognition_continues_while_consumer_is_busy_and_skips_backlog(self):
        frames = LatestFrameStream(lambda: Image.new('RGB', (20, 20)), .005).start()
        count = []
        progressed = Event()
        def recognize(image):
            count.append(image)
            if len(count) >= 5:
                progressed.set()
            return ['knight']
        perception = PerceptionStream(frames, SimpleNamespace(recognize=recognize))
        with patch('crbot.perception_stream._level_badge_candidates', return_value=[]):
            perception.start()
            try:
                first = perception.get()
                self.assertTrue(progressed.wait(2))
                latest = perception.get(sequence=first.frame.sequence)
                self.assertGreater(latest.frame.sequence, first.frame.sequence)
                self.assertEqual(latest.hand, ('knight',))
                self.assertEqual(perception.status()['queue_capacity'], 1)
            finally:
                perception.close()
                frames.close()
        self.assertFalse(perception.thread.is_alive())

    def test_failure_never_reuses_previous_result(self):
        frames = LatestFrameStream(lambda: Image.new('RGB', (20, 20)), .005).start()
        perception = PerceptionStream(frames, SimpleNamespace(recognize=Mock(side_effect=ValueError('failed')))).start()
        try:
            with self.assertRaisesRegex(ValueError, 'failed'):
                perception.get()
        finally:
            perception.close()
            frames.close()

    def test_battle_consumes_prepared_frame_but_confirmation_requires_post_click_capture(self):
        engine = object.__new__(BotEngine)
        image = Image.new('RGB', (20, 20))
        frame = Frame(image, 10, 10.01, 4)
        prepared = PreparedFrame(frame, ('knight',), (), 10.02)
        engine.frame_stream = SimpleNamespace(get=Mock(return_value=Frame(image, 11, 11.01, 5)))
        engine.perception_stream = SimpleNamespace(get=Mock(return_value=prepared))
        engine.policy = SimpleNamespace()
        engine.in_battle = True
        self.assertIs(engine._capture_frame(), image)
        self.assertIs(engine.policy.prepared_frame, prepared)
        engine._capture_frame(after=10.5)
        engine.frame_stream.get.assert_called_once_with(sequence=4, after=10.5, timeout=8.)
        self.assertEqual(engine._frame_sequence, 5)
        self.assertEqual(engine.perception_stream.get.call_count, 1)

    def test_precomputed_badges_preserve_filtering_and_threat_result(self):
        image = Image.new('RGB', (100, 200))
        badges = [(0.25, 0.55, 30), (0.75, 0.4, 30)]
        with patch('crbot.battle_perception._level_badge_candidates', return_value=badges) as detect:
            expected = detect_lane_threats(image, ignore_points=((.25, .55),))
            detect.reset_mock()
            actual = detect_lane_threats(image, ignore_points=((.25, .55),), candidates=badges)
            self.assertEqual(expected, actual)
            detect.assert_not_called()
