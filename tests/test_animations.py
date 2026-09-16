import os
import tkinter as tk
import unittest
from unittest.mock import patch

from crbot.animations import Animator, mix


class AnimationLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)

    def test_destroy_cancels_scheduled_animation(self):
        widget = tk.Frame(self.root)
        animator = Animator(widget)
        with patch.dict(os.environ, {'CRBOT_REDUCED_MOTION': '0'}):
            animator.pulse('running', lambda value: None)
        scheduled = list(animator.jobs.values())
        self.assertTrue(scheduled)
        widget.destroy()
        self.assertFalse(animator.jobs)
        pending = self.root.tk.call('after', 'info')
        self.assertTrue(all(job not in pending for job in scheduled))

    def test_reduced_motion_applies_final_state_without_timer(self):
        animator = Animator(self.root)
        frames = []
        with patch.dict(os.environ, {'CRBOT_REDUCED_MOTION': '1'}):
            animator.tween('hover', frames.append)
        self.assertEqual(frames, [1.0])
        self.assertFalse(animator.jobs)

    def test_color_interpolation_is_bounded(self):
        self.assertEqual(mix('#000000', '#ffffff', -1), '#000000')
        self.assertEqual(mix('#000000', '#ffffff', 2), '#ffffff')


if __name__ == '__main__':
    unittest.main()
