import unittest
from unittest.mock import Mock
from PIL import Image
from crbot.screen_recovery import ScreenRecovery
from crbot.engine import BotEngine
from tests import test_battle_transition


class ScreenRecoveryTests(unittest.TestCase):
    def test_static_delay_repeat_and_recognition_stop(self):
        guard = ScreenRecovery({})
        image = Image.new("RGB", (100, 200), "black")
        self.assertFalse(guard.observe(image, False, 0))
        self.assertFalse(guard.observe(image, False, 19))
        self.assertTrue(guard.observe(image, False, 20))
        guard.tapped(20)
        self.assertFalse(guard.observe(image, False, 20.5))
        # A changing animation after the first tap must not end recovery.
        self.assertTrue(guard.observe(Image.new("RGB", image.size, "white"), False, 21))
        self.assertFalse(guard.observe(image, True, 22))
        self.assertFalse(guard.active)
        self.assertFalse(guard.observe(image, False, 23))

    def test_motion_resets_timer_and_small_noise_does_not(self):
        guard = ScreenRecovery({})
        black = Image.new("RGB", (80, 80), (0, 0, 0))
        white = Image.new("RGB", black.size, "white")
        guard.observe(black, False, 0)
        self.assertFalse(guard.observe(white, False, 19))
        self.assertFalse(guard.observe(white, False, 20))
        self.assertTrue(guard.observe(white, False, 39))
        guard = ScreenRecovery({})
        guard.observe(black, False, 0)
        self.assertTrue(guard.observe(Image.new("RGB", black.size, (1, 1, 1)), False, 20))

    def test_dry_run_center_tap_does_not_touch_device(self):
        engine = object.__new__(BotEngine)
        engine.device = Mock()
        engine.dry_run = True
        self.assertIsNone(engine._tap([.5, .5], Image.new("RGB", (100, 200))))
        engine.device.tap_normalized.assert_not_called()
        engine.dry_run = False
        engine.device.tap_normalized.return_value = (50, 100)
        self.assertEqual(engine._tap([.5, .5], Image.new("RGB", (100, 200))), [50, 100])
        engine.device.tap_normalized.assert_called_once_with([.5, .5], (100, 200))

    def test_disabled_and_known_screens_never_click(self):
        image = Image.new("RGB", (20, 20))
        for config, known in (({"stuck_screen_recovery_enabled": False}, False), ({}, True)):
            guard = ScreenRecovery(config)
            for now in (0, 20, 100):
                self.assertFalse(guard.observe(image, known, now))

    def test_engine_recovery_resumes_same_battle_without_counting_completion(self):
        engine = test_battle_transition.BattleTransitionTests().run_engine([
            (100, 1), (101, 1), (115, 0), (135, 0), (135.5, 0),
            (136, 0), (137, 1), (138, 1)])
        self.assertEqual(engine._tap.call_count, 2)
        self.assertEqual(engine._tap.call_args.args[0], [.5, .5])
        self.assertEqual(engine.completed_battles, 0)
        self.assertTrue(engine.in_battle)
        engine.policy.reset_battle.assert_called_once()
        events = [c.args[0] for c in engine.recorder.record.call_args_list]
        self.assertIn("stuck_screen_recovered", events)


if __name__ == "__main__":
    unittest.main()
