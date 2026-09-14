from __future__ import annotations

import queue
import json
import tempfile
import tkinter as tk
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from crbot.gui import RoyalTrainerApp


class RunWorkerTests(unittest.TestCase):
    def test_flushes_partial_output_and_reports_completion(self) -> None:
        app = RoyalTrainerApp.__new__(RoyalTrainerApp)
        app.messages = queue.Queue()
        app._start_run_worker(lambda: print("partial", end=""), "test-run")
        app.bot_thread.join(timeout=5)
        self.assertFalse(app.bot_thread.is_alive())
        self.assertEqual(app.messages.get_nowait(), ("log", "partial"))
        self.assertEqual(app.messages.get_nowait(), ("bot_done", None))

    def test_worker_failure_still_reports_completion(self) -> None:
        app = RoyalTrainerApp.__new__(RoyalTrainerApp)
        app.messages = queue.Queue()

        def fail() -> None:
            raise RuntimeError("device disconnected")

        with patch.dict("os.environ", {"CRBOT_DEBUG": "0"}):
            app._start_run_worker(fail, "test-run")
            app.bot_thread.join(timeout=5)
        self.assertFalse(app.bot_thread.is_alive())
        self.assertEqual(app.messages.get_nowait(), ("log", "错误：device disconnected"))
        self.assertEqual(app.messages.get_nowait(), ("bot_done", "device disconnected"))


class ConsoleTests(unittest.TestCase):
    def test_predictive_selector_fits_minimum_window_and_locks_while_running(self):
        self.root.geometry("1040x700+0+0")
        self.root.deiconify()
        self.root.update_idletasks()
        self.root.update()
        self.assertIn("P1", self.app.engine_combo["values"][-1])
        from crbot import current_stage_label
        from tkinter.font import Font
        current = self.app.engine_combo["values"][-1]
        self.assertIn(current_stage_label(), current)
        label_width = Font(self.root, font=self.app.engine_combo["font"]).measure(current)
        self.assertLessEqual(label_width + 24, self.app.engine_combo.winfo_width())
        for widget in (self.app.engine_combo, self.app.model_combo, self.app.start_button,
                       self.app.stop_button, self.app.release_heading):
            left = widget.winfo_rootx() - self.root.winfo_rootx()
            top = widget.winfo_rooty() - self.root.winfo_rooty()
            self.assertGreaterEqual(left, 0)
            self.assertGreaterEqual(top, 0)
            self.assertLessEqual(left + widget.winfo_width(), 1040)
            self.assertLessEqual(top + widget.winfo_height(), 700)
            self.assertGreater(widget.winfo_width(), 20)
        self.assertGreaterEqual(self.app.engine_combo.winfo_width(), self.app.engine_combo.winfo_reqwidth())
        self.app._set_running_controls(True)
        self.assertEqual(str(self.app.engine_combo["state"]), "disabled")
        self.app._set_running_controls(False)
        self.assertEqual(str(self.app.engine_combo["state"]), "readonly")

    def setUp(self) -> None:
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk unavailable: {exc}")
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        config_path = Path(directory.name) / "config.json"
        config_path.write_text(
            json.dumps({"game": {"allowed_mode": "offline_ai_only"}}), encoding="utf-8"
        )
        with patch("crbot.gui.SyncWorker"), patch.object(self.root, "after"):
            self.app = RoyalTrainerApp(self.root, config_path)

    def test_running_controls_and_model_selection_restore_on_completion(self) -> None:
        self.app._set_running_controls(True)
        self.assertEqual(str(self.app.model_combo["state"]), "disabled")
        self.assertEqual(str(self.app.start_button["state"]), "disabled")
        self.assertEqual(str(self.app.stop_button["state"]), "normal")
        self.app._finish_bot(None)
        self.assertEqual(str(self.app.model_combo["state"]), "readonly")
        self.assertEqual(str(self.app.start_button["state"]), "normal")
        self.assertEqual(str(self.app.stop_button["state"]), "disabled")

    def test_model_updates_refresh_console_and_open_details_without_restart(self) -> None:
        registry = self.app.config_path.parent / "models/replay_policy/registry.json"
        registry.parent.mkdir(parents=True)
        self.app.show_model_versions()
        for version in ("first", "updated"):
            registry.write_text(json.dumps({"candidates": [{
                "version": version, "created_at_unix": 2, "quality_passed": False,
            }]}), encoding="utf-8")
            with patch.object(self.root, "after") as schedule:
                self.app._refresh_model_status()
            schedule.assert_called_once_with(3000, self.app._refresh_model_status)
            self.assertIn(version, self.app.model_status_button["text"])
            self.assertIn(version, self.app.model_details_text.get("1.0", "end"))
        self.assertNotIn("first", self.app.model_details_text.get("1.0", "end"))

    def test_tools_menu_routes_all_auxiliary_actions(self) -> None:
        methods = (
            "open_calibration", "open_runs_folder", "open_annotation",
            "check_learning_status", "show_self_learning_status", "start_demonstration", "sync_training_data",
        )
        with ExitStack() as stack:
            actions = [stack.enter_context(patch.object(RoyalTrainerApp, method)) for method in methods]
            stack.enter_context(patch("crbot.gui.SyncWorker"))
            stack.enter_context(patch.object(self.root, "after"))
            for widget in self.root.winfo_children():
                widget.destroy()
            app = RoyalTrainerApp(self.root, self.app.config_path)
            for index, action in enumerate(actions):
                app.tools_menu.invoke(index)
                action.assert_called_once_with()

    def test_stop_during_connection_cancels_both_run_modes(self) -> None:
        for start, engine in (
            (self.app.start_bot, "BotEngine"),
            (self.app.start_demonstration, "DemonstrationRecorder"),
        ):
            with self.subTest(engine=engine), patch("crbot.gui.MumuDevice") as device, patch(
                f"crbot.gui.{engine}"
            ) as factory, patch("crbot.gui.messagebox.askokcancel", return_value=True):
                device.return_value.connect.side_effect = lambda: self.app.stop_event.set()
                start()
                self.app.bot_thread.join(timeout=5)
                self.assertFalse(self.app.bot_thread.is_alive())
                device.return_value.connect.assert_called_once_with()
                factory.assert_not_called()
                self.assertIsNone(self.app.engine)


if __name__ == "__main__":
    unittest.main()
