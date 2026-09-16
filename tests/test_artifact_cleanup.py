"""Exercise the Windows cleanup entry point against disposable project fixtures."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.name == 'nt', 'Windows cleanup tool')
class ArtifactCleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='royal-cleanup-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.put('crbot/__init__.py')
        self.put('tools/cleanup_artifacts.ps1',
                 (Path(__file__).parents[1] / 'tools/cleanup_artifacts.ps1').read_bytes())
        self.put('build/standalone-runtime/library.dll')
        self.put('build/example-release-3.12.6/build/cache.bin')
        self.put('build/example-release-3.12.6/crbot/source.py')
        self.put('reports/gpu_setup/r2/install.whl')
        self.put('reports/gpu_setup/diagnosis.txt')
        self.put('RoyalLab.exe')
        self.put('RoyalLab-3.9.0.exe')
        self.put('RoyalLab-3.12.6.exe')
        for path in ('models/model.bin', 'runs/battle/frame.jpg', '.venv/library.dll',
                     '.training-sync/identity.json', 'config.json', 'training/data.bin'):
            self.put(path)

    def put(self, name, data=b'fixture'):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def run_cleanup(self, preview=False, process=''):
        # Deterministic process inventory; production script still performs its
        # real process/path checks on every target.
        wrapper = self.root / 'invoke.ps1'
        wrapper.write_text("function Get-CimInstance { " + process + " }\n"
                           "& (Join-Path $PSScriptRoot 'tools/cleanup_artifacts.ps1') "
                           + ('-Preview' if preview else '') + '\nexit $LASTEXITCODE',
                           encoding='utf-8')
        result = subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy',
                                 'Bypass', '-File', str(wrapper)], capture_output=True,
                                text=True, timeout=60)
        reports = sorted((self.root / 'reports').glob('artifact-cleanup-*.json'))
        self.assertTrue(reports, result.stdout + result.stderr)
        return result, json.loads(reports[-1].read_text(encoding='utf-8-sig'))

    def test_preview_then_delete_and_repeat_preserves_user_data(self):
        result, report = self.run_cleanup(preview=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertGreater(report['plannedBytes'], 0)
        self.assertEqual(report['freedBytes'], 0)
        self.assertTrue((self.root / 'build/standalone-runtime/library.dll').exists())
        result, report = self.run_cleanup()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertGreater(report['freedBytes'], 0)
        for name in ('build/standalone-runtime', 'reports/gpu_setup/r2/install.whl',
                     'RoyalLab-3.9.0.exe', 'build/example-release-3.12.6/build'):
            self.assertFalse((self.root / name).exists(), name)
        for name in ('RoyalLab.exe', 'RoyalLab-3.12.6.exe', 'models/model.bin',
                     'runs/battle/frame.jpg', '.venv/library.dll', 'config.json',
                     '.training-sync/identity.json', 'training/data.bin',
                     'reports/gpu_setup/diagnosis.txt',
                     'build/example-release-3.12.6/crbot/source.py'):
            self.assertTrue((self.root / name).exists(), name)
        result, report = self.run_cleanup()
        self.assertEqual(result.returncode, 0)
        self.assertEqual(report['freedBytes'], 0)

    def test_busy_build_blocks_all_cleanup(self):
        result, report = self.run_cleanup(process="[pscustomobject]@{Name='python.exe'; CommandLine='python tools/build_standalone.py'; ProcessId=123; ExecutablePath='C:\\python.exe'}")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(report['freedBytes'], 0)
        self.assertTrue((self.root / 'reports/gpu_setup/r2/install.whl').exists())

    def test_running_old_root_executable_is_preserved(self):
        path = str(self.root / 'RoyalLab-3.9.0.exe').replace("'", "''")
        result, report = self.run_cleanup(process="[pscustomobject]@{Name='RoyalLab.exe'; CommandLine='running'; ProcessId=123; ExecutablePath='" + path + "'}")
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((self.root / 'RoyalLab-3.9.0.exe').exists())

    def test_junction_is_not_followed(self):
        outside = self.root / 'protected'
        outside.mkdir()
        (outside / 'keep.bin').write_bytes(b'keep')
        link = self.root / 'build/standalone-runtime/junction'
        command = "New-Item -ItemType Junction -Path '" + str(link) + "' -Target '" + str(outside) + "' | Out-Null"
        subprocess.run(['powershell.exe', '-NoProfile', '-Command', command], check=True,
                       capture_output=True, timeout=20)
        try:
            result = subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy',
                                     'Bypass', '-File', str(self.root / 'tools/cleanup_artifacts.ps1')],
                                    capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue((outside / 'keep.bin').exists())
            self.assertTrue((self.root / 'build/standalone-runtime/library.dll').exists())
        finally:
            # Remove the junction itself, never its destination.
            os.rmdir(link)


if __name__ == '__main__':
    unittest.main()
