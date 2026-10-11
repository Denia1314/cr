"""Verify project-independent dependencies and actual Windows launcher reuse."""
import hashlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools import ensure_gpu as boot


class SharedEnvironmentTests(unittest.TestCase):
    @unittest.skipUnless(os.name == 'nt', 'Windows shared setup lock')
    def test_concurrent_launchers_wait_for_shared_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            environment = dict(os.environ, CRBOT_PYTHON_ENV=str(Path(tmp) / 'shared'))
            code = ('from tools.ensure_gpu import environment_lock; import time; '
                    'exec("with environment_lock():\\n print(\'locked\',flush=True)\\n time.sleep(1)")')
            first = subprocess.Popen([sys.executable, '-c', code], cwd=boot.ROOT,
                                     env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                self.assertEqual(first.stdout.readline().strip(), 'locked')
                second = subprocess.run([sys.executable, '-c',
                    'from tools.ensure_gpu import environment_lock; exec("with environment_lock():\\n print(\'second locked\')")'],
                    cwd=boot.ROOT, env=environment, capture_output=True, text=True, timeout=10)
                self.assertEqual(second.returncode, 0, second.stderr)
                self.assertIn('waiting', second.stdout)
                self.assertIn('second locked', second.stdout)
                self.assertEqual(first.wait(timeout=10), 0)
            finally:
                if first.poll() is None:
                    first.kill(); first.wait()
                first.stdout.close(); first.stderr.close()

    def test_failed_dependency_install_is_not_marked_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'requirements.txt').write_text('numpy>=2\n')
            with patch.dict(os.environ, {'CRBOT_PYTHON_ENV': str(root)}), patch.object(boot, 'ROOT', root), patch.object(boot, 'run', side_effect=[Mock(returncode=1), Mock(returncode=1)]):
                self.assertEqual(boot.ensure_requirements(), 1)
                self.assertFalse((root / '.requirements-cpu').exists())

    def test_default_path_survives_checkout_change(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'LOCALAPPDATA': tmp}, clear=True):
            with patch.object(boot, 'ROOT', Path(tmp) / 'checkout-a'):
                first = boot.environment_root()
            with patch.object(boot, 'ROOT', Path(tmp) / 'checkout-b'):
                self.assertEqual(boot.environment_root(), first)
            self.assertTrue(first.is_relative_to(Path(tmp) / 'RoyalLab'))
            self.assertIn(f'{sys.version_info.major}.{sys.version_info.minor}', first.name)

    def test_custom_path_must_be_absolute(self):
        with patch.dict(os.environ, {'CRBOT_PYTHON_ENV': 'relative'}):
            with self.assertRaises(ValueError):
                boot.environment_root()

    def test_base_python_not_deleted_local_venv_is_used_to_create_environment(self):
        with patch.object(boot.Path, 'exists', return_value=False), patch.object(boot, 'run', return_value=Mock(returncode=0)) as run:
            boot.create_environment()
            self.assertEqual(run.call_args.args[0][0], sys._base_executable)

    def test_same_requirements_in_new_checkout_do_not_install_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); env = root / 'shared'; env.mkdir()
            first, second = root / 'a', root / 'b'
            first.mkdir(); second.mkdir()
            for checkout in (first, second):
                (checkout / 'requirements.txt').write_text('numpy>=2\n')
            with patch.dict(os.environ, {'CRBOT_PYTHON_ENV': str(env)}), patch.object(boot, 'ROOT', first), patch.object(boot, 'run', return_value=Mock(returncode=0)) as run:
                self.assertEqual(boot.ensure_requirements(), 0)
                self.assertTrue(any('install' in c.args[0] for c in run.call_args_list))
                run.reset_mock()
                with patch.object(boot, 'ROOT', second):
                    self.assertEqual(boot.ensure_requirements(), 0)
                self.assertFalse(any('install' in c.args[0] for c in run.call_args_list))
                run.reset_mock()
                (second / 'requirements.txt').write_text('numpy>=2.1\n')
                with patch.object(boot, 'ROOT', second):
                    self.assertEqual(boot.ensure_requirements(), 0)
                self.assertTrue(any('install' in c.args[0] for c in run.call_args_list))

    @unittest.skipUnless(os.name == 'nt', 'Windows launcher')
    def test_two_checkouts_with_spaces_share_one_environment_and_no_local_venv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); shared = root / 'shared Python'; environment = dict(os.environ, CRBOT_PYTHON_ENV=str(shared))
            for name in ('checkout one', 'checkout two'):
                checkout = root / name; (checkout / 'tools').mkdir(parents=True)
                for file in ('python_env.bat', 'ensure_gpu.py'):
                    shutil.copy2(boot.ROOT / 'tools' / file, checkout / 'tools' / file)
                launcher = checkout / 'check.bat'
                launcher.write_text('@echo off\nsetlocal\ncall "%~dp0tools\\python_env.bat" console\nif errorlevel 1 exit /b 1\n"%BOT_PY%" -c "import sys;print(sys.executable)"\n')
                result = subprocess.run(['cmd.exe', '/d', '/c', str(launcher)], env=environment,
                                        capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(str(shared / 'Scripts/python.exe').replace('/', '\\'), result.stdout)
                self.assertFalse((checkout / '.venv').exists())

    def test_public_asset_cache_is_reused_and_checked(self):
        from crbot import battlefield_assets as assets
        from tools.ensure_battlefield import ensure
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); data = b'fixture public baseline'
            definitions = {name: (len(data), hashlib.sha256(data).hexdigest()) for name in assets.ASSETS}
            fake_ort = SimpleNamespace(__version__='1.23.2', get_available_providers=lambda: ['CPUExecutionProvider'])
            with patch.dict(os.environ, {'LOCALAPPDATA': str(base), 'CRBOT_COMPUTE_DEVICE': 'cpu'}), patch.dict(sys.modules, {'onnxruntime': fake_ort}), patch.object(assets, 'ASSETS', definitions), patch.object(assets, 'urlopen', side_effect=AssertionError('must reuse validated cache')):
                first = base / 'checkout-a'; first.mkdir()
                (first / 'config.json').write_text('{}')
                for name in definitions:
                    path = first / assets.RELATIVE_ROOT / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
                ensure(first)
                second = base / 'checkout-b'; second.mkdir(); (second / 'config.json').write_text('{}')
                ensure(second)
                self.assertTrue(assets.assets_ready(second))
                cache_file = boot.shared_home() / 'public-assets' / assets.RELATIVE_ROOT / next(iter(definitions))
                cache_file.write_bytes(b'corrupted')
                with patch.object(assets, 'urlopen', return_value=io.BytesIO(data)) as download:
                    ensure(second)
                    self.assertEqual(download.call_count, 0)  # Valid project file repairs cache first.
                self.assertTrue(assets.verify_asset(cache_file, next(iter(definitions))))


if __name__ == '__main__':
    unittest.main()
