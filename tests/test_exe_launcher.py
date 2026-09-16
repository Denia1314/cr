import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import exe_launcher


class LauncherTests(unittest.TestCase):
    def test_frozen_root_uses_executable_not_extraction_directory(self):
        with patch.object(exe_launcher.sys, "frozen", True, create=True), patch.object(
            exe_launcher.sys, "executable", str(Path("项目 with spaces/RoyalLab.exe").resolve())
        ):
            self.assertEqual(exe_launcher.project_root(), Path("项目 with spaces").resolve())

    def test_check_detects_missing_files_without_launching(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            exe_launcher, "project_root", return_value=Path(directory)
        ), patch.object(exe_launcher.sys, "argv", ["RoyalLab.exe", "--check"]), patch.object(
            exe_launcher.subprocess, "Popen"
        ) as spawn:
            self.assertEqual(exe_launcher.main(), 1)
            for name in ("start_ui.bat", "tools/python_env.bat", "launch_ui.pyw", "crbot/gui.py"):
                path = Path(directory) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            self.assertEqual(exe_launcher.main(), 0)
            spawn.assert_not_called()

    def test_launch_preserves_bootstrap_and_project_working_directory(self):
        with patch.object(exe_launcher.tk, "Tk"), patch.object(
            exe_launcher.subprocess, "Popen"
        ) as spawn, patch.object(exe_launcher.sys, "argv", ["RoyalLab.exe"]):
            self.assertEqual(exe_launcher.main(), 0)
            self.assertEqual(spawn.call_args.args[0][-1], "start_ui.bat")
            self.assertEqual(spawn.call_args.kwargs["cwd"], exe_launcher.project_root())
