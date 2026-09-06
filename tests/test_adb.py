from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from crbot.adb import DeviceError, MumuDevice


class MumuConnectionTests(unittest.TestCase):
    def test_auto_detects_mumu_from_environment_variable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            install_dir = Path(directory) / "custom-mumu"
            tool_dir = install_dir / "nx_main"
            tool_dir.mkdir(parents=True)
            (tool_dir / "mumu-cli.exe").touch()
            (tool_dir / "adb.exe").touch()
            config = {
                "mumu": {
                    "install_dir": "auto",
                    "vm_index": 0,
                    "auto_launch": True,
                    "startup_timeout_s": 120,
                }
            }

            with patch.dict(
                os.environ, {"MUMU_INSTALL_DIR": str(install_dir)}, clear=False
            ):
                device = MumuDevice(config)

            self.assertEqual(device.install_dir, install_dir.resolve())
            self.assertEqual(device.cli_path, (tool_dir / "mumu-cli.exe").resolve())
            self.assertEqual(device.adb_path, (tool_dir / "adb.exe").resolve())

    def test_connect_retries_transient_offline_state(self) -> None:
        device = MumuDevice.__new__(MumuDevice)
        device.cli_path = Path("mumu-cli.exe")
        device.adb_path = Path("adb.exe")
        device.vm_index = 0
        device.startup_timeout_s = 10.0
        device.serial = None
        device.ensure_running = lambda: {  # type: ignore[method-assign]
            "adb_host_ip": "127.0.0.1",
            "adb_port": 16384,
        }
        state_attempts = 0

        def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            nonlocal state_attempts
            if command[-1] == "get-state":
                state_attempts += 1
                state = "offline\n" if state_attempts == 1 else "device\n"
                return subprocess.CompletedProcess(command, 0, state, "")
            if command[-2:] == ["getprop", "sys.boot_completed"]:
                return subprocess.CompletedProcess(command, 0, "1\n", "")
            return subprocess.CompletedProcess(command, 0, "connected\n", "")

        device._run = fake_run  # type: ignore[method-assign]
        with patch("crbot.adb.time.sleep", return_value=None):
            serial = device.connect()

        self.assertEqual(serial, "127.0.0.1:16384")
        self.assertEqual(state_attempts, 2)

    def test_foreground_package_reconnects_after_transient_adb_failure(self) -> None:
        device = MumuDevice.__new__(MumuDevice)
        attempts = 0
        reconnects = 0

        def fake_adb(_arguments: list[str], **_kwargs: object) -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise DeviceError("transient adb failure")
            return (
                "topResumedActivity=ActivityRecord{abc u0 "
                "com.tencent.tmgp.supercell.clashroyale/.GameApp}"
            )

        def fake_connect() -> str:
            nonlocal reconnects
            reconnects += 1
            device.serial = "127.0.0.1:16384"
            return device.serial

        device.adb = fake_adb  # type: ignore[method-assign]
        device.connect = fake_connect  # type: ignore[method-assign]

        package = device.foreground_package()

        self.assertEqual(package, "com.tencent.tmgp.supercell.clashroyale")
        self.assertEqual(attempts, 2)
        self.assertEqual(reconnects, 1)


if __name__ == "__main__":
    unittest.main()
