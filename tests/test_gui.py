from __future__ import annotations

import queue
import json
import tempfile
import tkinter as tk
import unittest
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
    def test_live_grid_is_independent_of_slow_planner_but_snapshot_remains_available(self):
        from types import SimpleNamespace
        from PIL import Image
        old=(Image.new('RGB',(20,40)),{'revision':1,'entities':[]})
        live=(Image.new('RGB',(20,40)),{'revision':10,'entities':[]})
        self.app.engine=SimpleNamespace(policy=SimpleNamespace(grid_frame=old,yolo_frame=live),
                                       live_grid_stream=SimpleNamespace(latest=lambda:live))
        self.assertIs(self.app._grid_preview_source(),live)
        self.app.preview_mode.set('推演快照')
        self.assertIs(self.app._grid_preview_source(),old)
        self.app.engine=None

    def test_grid_is_embedded_and_switching_preserves_same_frame_pair(self):
        from types import SimpleNamespace
        from PIL import Image
        self.assertEqual(self.app.preview_mode.get(),'同帧对照')
        self.assertIs(self.app.grid_inspector.window.winfo_toplevel(),self.root)
        paired=Image.new('RGB',(100,200),'red')
        packet={'revision':1,'entities':[]}
        self.app.engine=SimpleNamespace(policy=SimpleNamespace(yolo_frame=(paired,packet)))
        self.app._show_image(Image.new('RGB',(100,200),'blue'))
        self.assertIs(self.app._grid_preview_source()[0],paired)
        self.app.preview_mode.set('游戏画面');self.app._set_preview_mode()
        self.assertEqual(self.app.grid_inspector.window.winfo_manager(),'')
        self.app.preview_mode.set('YOLO识别');self.app._set_preview_mode()
        self.assertEqual(self.app.grid_inspector.mode,'yolo')
        self.app.show_grid_world()
        self.assertEqual(self.app.grid_inspector.mode,'yolo_compare')
        self.app._toggle_preview_size()
        self.assertEqual(self.app.preview_card.grid_info()['columnspan'],2)
        self.assertEqual(self.app.log_card.winfo_manager(),'')
        self.assertNotEqual(self.app.stop_button.winfo_manager(),'')
        self.app._toggle_preview_size()
        self.assertEqual(self.app.preview_card.grid_info()['columnspan'],1)
        self.assertEqual(self.app.log_card.winfo_manager(),'grid')
        self.assertFalse(any(isinstance(w,tk.Toplevel) for w in self.root.winfo_children()))
        self.app.engine=None

    def test_main_controls_and_release_fit_minimum_window(self):
        self.root.geometry("1040x700+0+0")
        self.root.deiconify()
        self.root.update_idletasks()
        self.root.update()
        from crbot import current_stage_label, release_label
        self.assertIn(current_stage_label(), self.app.release_heading["text"])
        self.assertEqual(self.app.release_heading["text"], release_label())
        for widget in (self.app.check_button, self.app.connection_button, self.app.card_library_button, self.app.start_button,
                       self.app.stop_button, self.app.release_heading):
            left = widget.winfo_rootx() - self.root.winfo_rootx()
            top = widget.winfo_rooty() - self.root.winfo_rooty()
            self.assertGreaterEqual(left, 0)
            self.assertGreaterEqual(top, 0)
            self.assertLessEqual(left + widget.winfo_width(), 1040)
            self.assertLessEqual(top + widget.winfo_height(), 700)
            self.assertGreater(widget.winfo_width(), 20)
        self.assertIs(self.app.connection_button.master, self.app.check_button.master)
        self.assertGreater(self.app.connection_button.winfo_rootx()-self.root.winfo_rootx(), 520)
        self.assertLess(self.app.connection_button.winfo_rooty()-self.root.winfo_rooty(), 100)

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

    def test_running_controls_restore_on_completion(self) -> None:
        self.app._set_running_controls(True)
        self.assertEqual(str(self.app.start_button["state"]), "disabled")
        self.assertEqual(str(self.app.stop_button["state"]), "normal")
        self.app._finish_bot(None)
        self.assertEqual(str(self.app.start_button["state"]), "normal")
        self.assertEqual(str(self.app.stop_button["state"]), "disabled")

    def test_animated_list_routes_preview_mode(self):
        import os
        self.root.deiconify()
        self.root.update()
        with patch.dict(os.environ, {"CRBOT_REDUCED_MOTION": "1"}):
            self.app.preview_mode_combo._choose(0)
            self.assertEqual(self.app.preview_mode.get(), "游戏画面")
            self.assertEqual(self.app.grid_inspector.window.winfo_manager(), "")

    def test_compact_layout_keeps_preview_and_log_usable_and_returns_to_large_layout(self):
        self.root.geometry("1040x700+0+0")
        self.root.deiconify()
        self.root.update()
        self.assertTrue(self.app._compact)
        for widget in (self.app.preview_card, self.app.log_card):
            self.assertGreaterEqual(widget.winfo_height(), 190)
        self.app._toggle_preview_size()
        self.root.update()
        self.assertTrue(self.app.start_button.winfo_ismapped())
        self.assertTrue(self.app.stop_button.winfo_ismapped())
        self.app.expand_preview_button.invoke()
        self.root.update()
        self.assertFalse(self.app.preview_expanded)
        self.assertTrue(self.app.log_card.winfo_ismapped())
        self.root.geometry("1280x860+0+0")
        self.root.update()
        self.assertFalse(self.app._compact)

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

    def test_removed_controls_are_absent_and_status_updates_still_work(self):
        for name in ("engine_combo", "model_combo", "safety_value", "safety_note",
                     "tools_button", "tools_menu", "tools_popover", "sidebar", "overview_button"):
            self.assertFalse(hasattr(self.app, name), name)
        self.assertEqual(len(self.app.stats_card.winfo_children()), 2)
        self.app._apply_device_status({"connected": True, "installed": True,
                                      "calibrated": ["offline_ai_marker"], "serial": "test"})
        self.app._append_log("[安全门] 离线人机入口已确认")
        self.assertEqual(self.app.device_value["text"], "已连接")
        self.assertIn("离线入口已验证", self.app.run_state_label["text"])

    def test_training_keeps_saved_rule_and_engine_without_selectors(self):
        from crbot.runtime_model import save_runtime_model, save_decision_engine
        save_runtime_model(self.app.config_path, "m2")
        save_decision_engine(self.app.config_path, "shadow")
        for widget in self.root.winfo_children():
            widget.destroy()
        with patch("crbot.gui.SyncWorker"), patch.object(self.root, "after"):
            self.app = RoyalTrainerApp(self.root, self.app.config_path)
        self.assertEqual(self.app.runtime_model, "m2")
        self.assertEqual(self.app.decision_engine, "shadow")
        with patch("crbot.gui.MumuDevice"), patch("crbot.gui.BotEngine") as factory:
            self.app.start_bot()
            self.app.bot_thread.join(timeout=5)
        self.assertFalse(self.app.bot_thread.is_alive())
        factory.assert_called_once()
        effective_config = factory.call_args.args[1]
        self.assertEqual(effective_config["policy"]["runtime_model"], "m2")
        self.assertEqual(effective_config["policy"]["decision_engine"], "shadow")
        self.assertEqual(effective_config["game"]["allowed_mode"], "offline_ai_only")
        factory.return_value.run.assert_called_once_with()

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
