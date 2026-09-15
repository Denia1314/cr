import ctypes
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from crbot.atomic_file import atomic_write
from crbot.replay_learning import _write_json


def sharing_error(code=5):
    error = PermissionError("simulated Windows sharing conflict")
    error.winerror = code
    return error


class AtomicFileTests(unittest.TestCase):
    def test_transient_conflict_retries_and_keeps_original_until_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_bytes(b"old")
            original_replace = os.replace
            sources = []
            def replace(source, target):
                sources.append(source)
                self.assertEqual(path.read_bytes(), b"old")
                if len(sources) < 3:
                    raise sharing_error(32)
                original_replace(source, target)
            with patch("crbot.atomic_file.os.replace", side_effect=replace), patch("crbot.atomic_file.time.sleep"):
                _write_json(path, {"champion": "old", "candidates": ["new"]})
            self.assertEqual(json.loads(path.read_text())["candidates"], ["new"])
            self.assertNotEqual(sources[0].name, "registry.json.tmp")
            self.assertEqual(len(set(sources)), 1)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_permanent_denial_preserves_registry_and_reports_original_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_bytes(b"old")
            error = sharing_error()
            with patch("crbot.atomic_file.os.replace", side_effect=error) as replace, patch("crbot.atomic_file.time.sleep") as sleep:
                with self.assertRaises(PermissionError) as caught:
                    atomic_write(path, b"new")
            self.assertIs(caught.exception, error)
            self.assertEqual(replace.call_count, 9)
            self.assertEqual(sleep.call_count, 8)
            self.assertEqual(path.read_bytes(), b"old")
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_unrelated_errors_are_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("crbot.atomic_file.os.replace", side_effect=OSError("disk failure")), patch("crbot.atomic_file.time.sleep") as sleep:
                with self.assertRaises(OSError):
                    atomic_write(Path(directory) / "file", b"new")
            sleep.assert_not_called()

    @unittest.skipUnless(os.name == "nt", "Windows handle sharing test")
    def test_real_windows_reader_without_delete_share_recovers(self):
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
        kernel.CreateFileW.restype = ctypes.c_void_p
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_bytes(b"old")
            handle = kernel.CreateFileW(str(path), 0x80000000, 1, None, 3, 0, None)
            self.assertNotEqual(handle, ctypes.c_void_p(-1).value)
            timer = threading.Timer(.12, lambda: kernel.CloseHandle(handle))
            try:
                probe = Path(directory) / "probe"
                probe.write_bytes(b"probe")
                with self.assertRaises(PermissionError):
                    os.replace(probe, path)
                timer.start()
                atomic_write(path, b"new")
                self.assertEqual(path.read_bytes(), b"new")
            finally:
                if timer.ident is None:
                    kernel.CloseHandle(handle)
                else:
                    timer.join()
