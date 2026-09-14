import unittest
from unittest.mock import Mock, patch

from PIL import Image

from crbot.battle_transition import BattleTransitionGuard
from crbot.engine import BotEngine


class BattleTransitionTests(unittest.TestCase):
    def test_matchmaking_flash_and_intro_are_not_battle_entry(self):
        guard = BattleTransitionGuard()
        guard.observe(True, 0)
        self.assertFalse(guard.entry_ready)
        for now in (1, 2, 3, 4):
            guard.observe(False, now)
            self.assertFalse(guard.end_pending)
        guard.observe(True, 5)
        self.assertFalse(guard.entry_ready)
        guard.observe(True, 6)
        self.assertTrue(guard.entry_ready)

    def test_fast_polling_cannot_replace_elapsed_time(self):
        guard = BattleTransitionGuard()
        for i in range(20):
            guard.observe(True, i / 100)
        self.assertFalse(guard.entry_ready)
        guard.start(0)
        for i in range(20):
            guard.observe(False, 20 + i / 100)
        self.assertFalse(guard.end_pending)
        guard.observe(False, 24)
        self.assertTrue(guard.end_pending)

    def test_intro_grace_and_reset(self):
        guard = BattleTransitionGuard()
        guard.start(0)
        for now in range(1, 8):
            guard.observe(False, now)
        self.assertFalse(guard.end_pending)
        guard.observe(True, 8)
        self.assertFalse(guard.end_pending)
        guard.reset()
        self.assertIsNone(guard.started_at)
        self.assertEqual(guard.present_frames, 0)

    def run_engine(self, frames, confirm_last=False):
        # Frame times/scores model real capture sequences without ADB clicks.
        engine = object.__new__(BotEngine)
        engine.config = {'automation': {'start_battle_point': [.5, .75],
                         'battle_ui_roi': [.1, .94, .99, .999],
                         'chest_screen_enabled': False}, 'timing': {}}
        engine.device = Mock()
        engine.device.foreground_package.return_value = 'game'
        engine.recognizer = Mock()
        engine.recognizer.match_all.return_value = []
        engine._update_offline_gate = Mock(return_value=None)
        engine.offline_verified = True
        engine.offline_gate_streak = 0
        engine.in_battle = engine.battle_seen = False
        engine.completed_battles = engine.max_battles = 0
        engine.dry_run = True
        engine.last_known_at = 0
        engine.policy = Mock()
        engine.replay = Mock()
        engine.replay.policy_metadata = {}
        engine.experiment = Mock()
        engine.experiment.activate.return_value = {}
        engine.recorder = Mock()
        engine.response_timing = Mock()
        engine._record_response_timing = Mock()
        engine._play_battle = Mock()
        engine._tap = Mock()
        engine._last_screenshot_elapsed_s = .1
        image = Image.new('RGB', (20, 20))
        engine._capture_frame = Mock(return_value=image)
        position = [0]
        engine._stop_requested = lambda: position[0] >= len(frames)

        def sleep(_seconds):
            position[0] += 1
            return position[0] >= len(frames)

        engine._sleep = sleep
        with patch('crbot.engine.time.monotonic', side_effect=lambda: frames[position[0]][0]), \
             patch('crbot.engine.battle_ui_score', side_effect=lambda *_: frames[position[0]][1]), \
             patch('crbot.engine.find_result_confirm_button', side_effect=lambda *_:
                   ([.5, .9], .95) if confirm_last and position[0] == len(frames)-1 else (None, 0)), \
             patch('crbot.engine.detect_battle_result'):
            engine._run_single_marker('game')
        return engine

    def test_recorded_false_start_pattern_recovers_without_false_completion(self):
        engine = self.run_engine([(100, .86), (101, 0), (102, 0), (103, 0),
                                  (104, 0), (105, 1), (106, 1), (107, 1)])
        self.assertTrue(engine.in_battle)
        self.assertEqual(engine.completed_battles, 0)
        self.assertEqual(engine._play_battle.call_count, 2)
        engine.policy.reset_battle.assert_called_once()
        engine._tap.assert_not_called()

    def test_provisional_end_resumes_same_policy_and_episode(self):
        engine = self.run_engine([(100, 1), (101, 1), (115, 0), (116, 0),
                                  (117, 0), (118, 0), (119, 1), (120, 1)])
        events = [call.args[0] for call in engine.recorder.record.call_args_list]
        self.assertIn('battle_end_pending', events)
        self.assertIn('battle_resumed_after_ui_gap', events)
        self.assertEqual(engine.completed_battles, 0)
        engine.policy.reset_battle.assert_called_once()
        engine.replay.start_battle.assert_called_once_with(1)
        self.assertEqual(engine._play_battle.call_count, 2)

    def test_real_confirmation_finishes_once_even_during_start_grace(self):
        engine = self.run_engine([(100, 1), (101, 1), (102, 0)], confirm_last=True)
        self.assertFalse(engine.in_battle)
        self.assertEqual(engine.completed_battles, 1)
        engine._tap.assert_called_once()

    def test_confirmation_after_pending_end_does_not_double_count(self):
        engine = self.run_engine([(100, 1), (101, 1), (115, 0), (116, 0),
                                  (117, 0), (118, 0), (119, 0)], confirm_last=True)
        self.assertEqual(engine.completed_battles, 1)
        self.assertFalse(engine.in_battle)
