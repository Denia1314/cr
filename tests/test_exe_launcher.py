import os
import queue
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tools import exe_launcher as launcher


class LauncherTests(unittest.TestCase):
    def test_frozen_root_uses_executable_not_extraction_directory(self):
        with patch.object(launcher.sys, 'frozen', True, create=True), patch.object(
            launcher.sys, 'executable', str(Path('项目 with spaces/RoyalLab.exe').resolve())
        ):
            self.assertEqual(launcher.project_root(), Path('项目 with spaces').resolve())

    def test_external_python_does_not_inherit_frozen_tcl_or_python_paths(self):
        with patch.dict(os.environ, {'TCL_LIBRARY': 'temp/_MEI/tcl', 'TK_LIBRARY': 'temp/_MEI/tk',
                                     'PYTHONHOME': 'bad', 'PYTHONPATH': 'bad', '_PYI_PARENT_PROCESS_LEVEL': '1',
                                     'CRBOT_COMPUTE_DEVICE': 'cpu'}):
            env = launcher.child_environment()
            for key in ('TCL_LIBRARY', 'TK_LIBRARY', 'PYTHONHOME', 'PYTHONPATH', '_PYI_PARENT_PROCESS_LEVEL'):
                self.assertNotIn(key, env)
            self.assertEqual(env['CRBOT_COMPUTE_DEVICE'], 'cpu')
            self.assertEqual(env['CRBOT_NO_CONSOLE'], '1')
            self.assertEqual(os.environ['TCL_LIBRARY'], 'temp/_MEI/tcl')

    def test_check_detects_missing_files_without_launching(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            launcher, 'project_root', return_value=Path(directory)
        ), patch.object(launcher.sys, 'argv', ['RoyalLab.exe', '--check']), patch.object(
            launcher.subprocess, 'Popen'
        ) as spawn:
            self.assertEqual(launcher.main(), 1)
            for name in launcher.REQUIRED:
                path = Path(directory) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            self.assertEqual(launcher.main(), 0)
            spawn.assert_not_called()

    def test_hidden_subprocess_passes_clean_environment_and_no_console(self):
        process = Mock(returncode=0)
        process.poll.return_value = 0
        with tempfile.TemporaryFile(mode='w+') as log, patch.object(
            launcher.subprocess, 'Popen', return_value=process
        ) as spawn:
            self.assertEqual(launcher.run_hidden(['python', 'tool.py'], Path('.'), {'TEST': '1'}, log, threading.Event()), 0)
            self.assertEqual(spawn.call_args.kwargs['creationflags'], subprocess.CREATE_NO_WINDOW)
            self.assertEqual(spawn.call_args.kwargs['env'], {'TEST': '1'})
            self.assertNotIn('cmd.exe', spawn.call_args.args[0])

    def test_cancel_terminates_owned_preparation_process(self):
        event = threading.Event()
        event.set()
        process = Mock()
        process.poll.return_value = None
        with tempfile.TemporaryFile(mode='w+') as log, patch.object(
            launcher.subprocess, 'Popen', return_value=process
        ), patch.object(launcher, 'stop_process') as stop:
            with self.assertRaises(launcher.Cancelled):
                launcher.run_hidden(['python'], Path('.'), {}, log, event)
            stop.assert_called_once_with(process)

    def test_prepare_reselects_python_and_requires_real_ui_ready_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in launcher.REQUIRED:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            events = queue.Queue()
            process = Mock(pid=123)
            process.poll.return_value = None
            def launch(args, **kwargs):
                self.assertEqual(Path(args[1]), root / 'launch_ui.pyw')
                self.assertEqual(kwargs['creationflags'], subprocess.CREATE_NO_WINDOW)
                self.assertNotIn('TCL_LIBRARY', kwargs['env'])
                Path(args[-1]).write_text('ready')
                return process
            with patch.object(launcher, 'find_python', side_effect=[root/'base/python.exe', root/'.venv/Scripts/python.exe']) as find, patch.object(
                launcher, 'run_hidden', return_value=0
            ) as hidden, patch.object(launcher.subprocess, 'Popen', side_effect=launch):
                launcher.prepare_and_launch(root, events, threading.Event())
                self.assertEqual(find.call_count, 2)
                self.assertEqual(hidden.call_args_list[0].args[0][1], root/'tools/ensure_gpu.py')
                self.assertEqual(hidden.call_args_list[1].args[0][0], root/'.venv/Scripts/python.exe')
            self.assertEqual(list(events.queue)[-1], ('ready', None))
            self.assertIn('UI ready: pid=123', (root/'launcher.log').read_text())

    def test_bootstrap_failure_is_reported_and_ui_not_launched(self):
        events = queue.Queue()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in launcher.REQUIRED:
                path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.touch()
            with patch.object(launcher, 'find_python', return_value=Path('python.exe')), patch.object(
                launcher, 'run_hidden', return_value=1
            ), patch.object(launcher.subprocess, 'Popen') as spawn:
                launcher.prepare_and_launch(root, events, threading.Event())
                spawn.assert_not_called()
            self.assertEqual(list(events.queue)[-1][0], 'error')

    def test_ui_exit_without_ready_is_not_reported_as_success(self):
        events = queue.Queue()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in launcher.REQUIRED:
                path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.touch()
            process = Mock()
            process.poll.return_value = 1
            with patch.object(launcher, 'find_python', return_value=Path('python.exe')), patch.object(
                launcher, 'run_hidden', return_value=0
            ), patch.object(launcher.subprocess, 'Popen', return_value=process):
                launcher.prepare_and_launch(root, events, threading.Event())
            self.assertEqual(list(events.queue)[-1][0], 'error')


if __name__ == '__main__':
    unittest.main()
