from __future__ import annotations

import os
import tkinter as tk
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from crbot.ui_theme import Palette
from crbot.ui_widgets import HoverButton, ToolsPopover


class DesktopControlsTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.root.configure(bg=Palette.BG)
        self.action = Mock()
        self.button = HoverButton(self.root, text="Action", background=Palette.CARD,
                                  hover=Palette.SELECTION, command=self.action)
        self.button.pack()
        self.root.update_idletasks()

    def test_repeated_space_key_press_invokes_once_on_release(self):
        with patch.dict(os.environ, {"CRBOT_REDUCED_MOTION": "1"}):
            self.button._key_press(None)
            self.button._key_press(None)
            self.action.assert_not_called()
            self.button._key_release(None)
            self.button._key_release(None)
        self.action.assert_called_once_with()

    def test_dragging_off_button_and_disabling_during_press_cancel_action(self):
        with patch.dict(os.environ, {"CRBOT_REDUCED_MOTION": "1"}):
            self.button._press()
            self.button._leave()
            self.button._release(SimpleNamespace(x=-1, y=-1))
            self.button._press()
            self.button.configure(state="disabled")
            self.button._release(SimpleNamespace(x=1, y=1))
            self.button.invoke()
        self.action.assert_not_called()
        self.assertFalse(self.button.animator.jobs)

    def test_fast_hover_reversal_and_destroy_leave_no_animation_jobs(self):
        with patch.dict(os.environ, {"CRBOT_REDUCED_MOTION": "0"}):
            for _ in range(8):
                self.button._enter()
                self.button._leave()
            self.assertEqual(len(self.button.animator.jobs), 1)
            pending = list(self.button.animator.jobs.values())
            self.button.destroy()
        self.assertFalse(self.button.animator.jobs)
        timers = self.root.tk.call("after", "info")
        self.assertTrue(all(job not in timers for job in pending))

    def test_reduced_motion_hover_finishes_immediately(self):
        with patch.dict(os.environ, {"CRBOT_REDUCED_MOTION": "1"}):
            self.button._enter()
            self.assertEqual(self.button.cget("background"), Palette.SELECTION.lower())
            self.button._leave()
            self.assertEqual(self.button.cget("background"), Palette.CARD.lower())
        self.assertFalse(self.button.animator.jobs)

    def test_enter_activates_button_but_control_enter_reaches_window_shortcut(self):
        self.assertEqual(self.button._return(SimpleNamespace(state=0)), "break")
        self.action.assert_called_once_with()
        self.assertIsNone(self.button._return(SimpleNamespace(state=4)))
        self.action.assert_called_once_with()

    def test_tools_popover_routes_action_and_closes_before_command(self):
        menu = tk.Menu(self.root, tearoff=False)
        state = tk.BooleanVar(value=False)
        popover = ToolsPopover(self.root, self.button, menu)
        seen = []
        menu.add_command(label="Action", command=lambda: seen.append(popover.window))
        menu.add_separator()
        menu.add_checkbutton(label="Reduce motion", variable=state)
        with patch.dict(os.environ, {"CRBOT_REDUCED_MOTION": "1"}):
            popover.toggle()
            self.assertIsNotNone(popover.window)
            popover._invoke(0)
            self.assertEqual(seen, [None])
            popover.toggle()
            popover._invoke(2)
        self.assertTrue(state.get())
        self.assertIsNone(popover.window)


if __name__ == "__main__":
    unittest.main()
