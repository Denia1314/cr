from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from crbot.adb import DeviceError, MumuDevice


class MumuConnectionTests(unittest.TestCase):
    def test_custom_adb_mode_does_not_require_mumu_cli(self) -> None:
        config = {
            "mumu": {
                "connection_mode": "custom_adb",
                "adb_host": "127.0.0.1",
                "adb_port": 7555,
            }
        }
        with (
            patch.object(MumuDevice, "_discover_adb", return_value=Path("adb.exe")),
            patch.object(MumuDevice, "_discover_cli") as discover_cli,
        ):
            device = MumuDevice(config)

        discover_cli.assert_not_called()
        self.assertTrue(device.uses_custom_adb)
        self.assertIsNone(device.cli_path)
        self.assertEqual(device.connection_description, "自定义 ADB 127.0.0.1:7555")

    def test_custom_adb_connects_directly_to_configured_port(self) -> None:
        device = MumuDevice.__new__(MumuDevice)
        device.connection_mode = "custom_adb"
        device.adb_host = "127.0.0.1"
        device.adb_port = 7555
        device.adb_path = Path("adb.exe")
        device.cli_path = None
        device.startup_timeout_s = 10.0
        device.serial = None
        commands: list[list[str]] = []

        def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            commands.append(command)
            if command[-1] == "get-state":
                return subprocess.CompletedProcess(command, 0, "device\n", "")
            if command[-2:] == ["getprop", "sys.boot_completed"]:
                return subprocess.CompletedProcess(command, 0, "1\n", "")
            return subprocess.CompletedProcess(command, 0, "connected\n", "")

        device._run = fake_run  # type: ignore[method-assign]
        serial = device.connect()

        self.assertEqual(serial, "127.0.0.1:7555")
        self.assertIn(["adb.exe", "connect", "127.0.0.1:7555"], commands)
        self.assertFalse(any("mumu-cli" in " ".join(command) for command in commands))

    def test_custom_adb_port_is_validated(self) -> None:
        with self.assertRaisesRegex(DeviceError, "1 到 65535"):
            MumuDevice(
                {
                    "mumu": {
                        "connection_mode": "custom_adb",
                        "adb_port": 70000,
                    }
                }
            )

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
