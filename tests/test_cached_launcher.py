"""Exercise the real compiled Windows wrapper with a tiny bundled application."""
import base64
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import uuid

from tools.build_bundle import build_bundle, compiler, TRAILER


@unittest.skipUnless(os.name == 'nt' and compiler().exists(), 'Windows C# compiler required')
class CachedLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='royal launcher 中文 ')
        self.root = Path(self.temporary.name)
        self.runtime = self.root / 'runtime'
        self.runtime.mkdir()
        source = self.root / 'fixture.cs'
        source.write_text('''using System; using System.IO; using System.Linq;
class Fixture { static int Main(string[] args) {
    File.WriteAllLines(args[0], args.Skip(1).Concat(new[] {
        Environment.GetEnvironmentVariable("CRBOT_DATA_DIR") ?? "",
        Environment.GetEnvironmentVariable("TCL_LIBRARY") ?? "",
        Environment.GetEnvironmentVariable("PYTHONPATH") ?? ""
    }).Select(x => Convert.ToBase64String(System.Text.Encoding.UTF8.GetBytes(x))));
    return 7;
}}''', encoding='utf-8')
        subprocess.run([str(compiler()), '/nologo', '/target:winexe',
                        '/out:' + str(self.runtime / 'RoyalLab.exe'), str(source)], check=True)
        (self.runtime / 'resource.txt').write_text(str(uuid.uuid4()))
        self.exe = self.root / 'RoyalLab.exe'
        build_bundle(self.runtime, self.exe)
        with self.exe.open('rb') as stream:
            stream.seek(-TRAILER.size, 2)
            _, _, fingerprint, _ = TRAILER.unpack(stream.read())
        self.cache = Path(os.environ['LOCALAPPDATA']) / 'RoyalLab/runtime' / fingerprint.hex()

    def tearDown(self):
        allowed = (Path(os.environ['LOCALAPPDATA']) / 'RoyalLab/runtime').resolve()
        self.assertEqual(self.cache.resolve().parent, allowed)
        if self.cache.exists():
            shutil.rmtree(self.cache)
        self.temporary.cleanup()

    def run_app(self, *args, env=None):
        report = self.root / '参数结果.txt'
        result = subprocess.run([str(self.exe), str(report), *args], env=env, timeout=30)
        self.assertEqual(result.returncode, 7)
        return [base64.b64decode(line).decode('utf-8') for line in report.read_text().splitlines()]

    def test_reuse_preserves_files_and_forwards_exact_arguments(self):
        args = ['', '中文 path', 'a"b', 'D:\\space path\\', '& $(nothing)']
        env = dict(os.environ, TCL_LIBRARY='invalid', PYTHONPATH='invalid')
        result = self.run_app(*args, env=env)
        self.assertEqual(result[:-3], args)
        self.assertEqual(result[-2:], ['', ''])
        marker = (self.cache / '.ready').stat().st_mtime_ns
        self.run_app(*args)
        self.assertEqual((self.cache / '.ready').stat().st_mtime_ns, marker)

    def test_missing_and_modified_cache_are_repaired(self):
        self.run_app()
        cached = self.cache / 'resource.txt'
        cached.unlink()
        self.run_app()
        self.assertEqual(cached.read_bytes(), (self.runtime / 'resource.txt').read_bytes())
        cached.write_text('x' * cached.stat().st_size)
        self.run_app()
        self.assertEqual(cached.read_bytes(), (self.runtime / 'resource.txt').read_bytes())

    def test_portable_data_is_resolved_beside_outer_executable(self):
        config = self.root / 'config.json'
        config.write_text('{"keep":"my settings"}')
        env = os.environ.copy()
        env.pop('CRBOT_DATA_DIR', None)
        self.assertEqual(self.run_app(env=env)[-3], str(self.root))
        self.assertEqual(config.read_text(), '{"keep":"my settings"}')

    def test_concurrent_first_launches_share_complete_cache(self):
        reports = [self.root / ('result%d.txt' % i) for i in range(2)]
        processes = [subprocess.Popen([str(self.exe), str(report)]) for report in reports]
        for process in processes:
            self.assertEqual(process.wait(timeout=30), 7)
        self.assertTrue(all(report.exists() for report in reports))
        self.assertTrue((self.cache / '.ready').exists())


if __name__ == '__main__':
    unittest.main()
