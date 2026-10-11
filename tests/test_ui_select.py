from __future__ import annotations

import os
import time
import tkinter as tk
import unittest
from unittest.mock import patch

from crbot.ui_select import AnimatedSelect
from crbot.ui_theme import Palette


class AnimatedSelectTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.geometry("600x450+30+30")
        self.root.configure(bg=Palette.BG)
        self.errors = []
        self.root.report_callback_exception = lambda *args: self.errors.append(str(args[1]))
        self.variable = tk.StringVar(value="Second")
        self.select = AnimatedSelect(self.root, textvariable=self.variable,
                                     values=("First", "Second", "Third"))
        self.select.pack(fill="x", padx=20, pady=20)
        self.next_button = tk.Button(self.root, text="Next")
        self.next_button.pack()
        self.events = []
        self.select.bind("<<ComboboxSelected>>", lambda _e: self.events.append(self.variable.get()))
        self.root.update()
        self.addCleanup(self.root.destroy)

    def pump(self, seconds=.23):
        until = time.perf_counter()+seconds
        while time.perf_counter() < until:
            self.root.update()
            time.sleep(.005)

    def test_arrow_navigation_is_draft_and_enter_emits_once(self):
        with patch.dict(os.environ, {"CRBOT_REDUCED_MOTION": "1"}):
            self.select.open()
            self.root.update()
            self.assertEqual(self.select.active_index, 1)
            self.select._move(1)
            self.root.update()
            self.assertEqual(self.variable.get(), "Second")
            self.select.rows[2].event_generate("<Return>")
            self.root.update()
        self.assertEqual(self.variable.get(), "Third")
        self.assertEqual(self.events, ["Third"])
        self.assertIsNone(self.select.popup)
        self.assertFalse(self.errors)

    def test_escape_does_not_change_value_or_emit_event(self):
        with patch.dict(os.environ, {"CRBOT_REDUCED_MOTION": "1"}):
            self.select.open()
            self.select._move(-1)
            self.select._cancel_key()
        self.assertEqual(self.variable.get(), "Second")
        self.assertEqual(self.events, [])
        self.assertIsNone(self.select.popup)

    def test_disabled_field_dismisses_popup_and_blocks_reopening(self):
        self.select.open()
        popup = self.select.popup
        self.select.configure(state="disabled")
        self.select.open()
        self.assertIsNone(self.select.popup)
        self.assertFalse(popup.winfo_exists())
        self.assertEqual(str(self.select["state"]), "disabled")
        self.assertEqual(self.events, [])

    def test_open_close_and_hover_animate_then_release_all_popup_callbacks(self):
        with patch.dict(os.environ, {"CRBOT_REDUCED_MOTION": "0"}):
            self.select.open()
            popup = self.select.popup
            self.root.update()
            self.assertIn("visibility", popup.animator.jobs)
            row = self.select.rows[0]
            row.event_generate("<Enter>")
            self.assertIn("hover", row.animator.jobs)
            row.event_generate("<Leave>")
            self.pump()
            self.assertAlmostEqual(float(popup.attributes("-alpha")), 1.0)
            self.select.close()
            self.assertIs(self.select.popup, popup)
            self.pump()
        self.assertIsNone(self.select.popup)
        self.assertFalse(popup.animator.jobs)
        self.assertFalse(row.animator.jobs)
        self.assertFalse(self.errors)

    def test_reopening_during_fade_out_cancels_old_close(self):
        with patch.dict(os.environ, {"CRBOT_REDUCED_MOTION": "0"}):
            self.select.open()
            popup = self.select.popup
            self.select.close(restore_focus=True)
            self.select.open()
            self.pump()
        self.assertIs(self.select.popup, popup)
        self.assertTrue(popup.winfo_exists())
        self.assertFalse(self.select._closing)
        self.assertFalse(self.errors)

    def test_outside_click_and_owner_resize_dismiss_without_changing_selection(self):
        with patch.dict(os.environ, {"CRBOT_REDUCED_MOTION": "1"}):
            self.select.open()
            self.next_button.event_generate("<ButtonPress-1>")
            self.assertIsNone(self.select.popup)
            self.select.open()
            self.root.geometry("610x460+30+30")
            self.root.update()
            self.assertIsNone(self.select.popup)
        self.assertEqual(self.events, [])
        self.assertEqual(self.variable.get(), "Second")

    def test_destroy_removes_variable_trace_owner_bindings_and_popup(self):
        self.select.open()
        popup = self.select.popup
        traces = len(self.variable.trace_info())
        self.select.destroy()
        self.assertFalse(popup.winfo_exists())
        self.assertEqual(len(self.variable.trace_info()), traces-1)
        self.variable.set("First")
        self.pump(.05)
        self.assertFalse(self.errors)

    def test_list_opens_above_field_near_screen_bottom(self):
        bottom = self.root.winfo_screenheight()
        self.root.geometry(f"600x200+30+{bottom-210}")
        self.select.pack_configure(pady=65)
        self.root.update()
        with patch.dict(os.environ, {"CRBOT_REDUCED_MOTION": "1"}):
            self.select.open()
            self.root.update()
        popup = self.select.popup
        self.assertLess(popup.winfo_rooty(), self.select.winfo_rooty())
        self.assertLessEqual(popup.winfo_rooty()+popup.winfo_height(), bottom)

    def test_tab_closes_popup_and_focuses_next_field(self):
        with patch.dict(os.environ, {"CRBOT_REDUCED_MOTION": "1"}):
            self.select.open()
            self.root.update()
            self.select._tab(1)
            self.root.update()
        self.assertIsNone(self.select.popup)
        self.assertIs(self.root.focus_get(), self.next_button)
        self.assertEqual(self.events, [])


if __name__ == "__main__":
    unittest.main()
