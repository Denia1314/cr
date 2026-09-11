from __future__ import annotations

import io
import json
import os
import re
import shutil
import string
import subprocess
import time
from pathlib import Path
from typing import Any

from PIL import Image


class DeviceError(RuntimeError):
    pass


class MumuDevice:
    def __init__(self, config: dict[str, Any]):
        mumu = config["mumu"]
        configured_dir = str(mumu.get("install_dir", "auto") or "auto").strip()
        self.install_dir: Path | None = (
            None
            if configured_dir.lower() in {"auto", "detect", ""}
            else Path(os.path.expandvars(configured_dir)).expanduser()
        )
        self.vm_index = int(mumu.get("vm_index", 0))
        self.auto_launch = bool(mumu.get("auto_launch", True))
        self.startup_timeout_s = float(mumu.get("startup_timeout_s", 120))
        self.connection_mode = str(mumu.get("connection_mode", "auto")).strip().lower()
        if self.connection_mode not in {"auto", "custom_adb"}:
            raise DeviceError(
                "mumu.connection_mode 必须为 auto 或 custom_adb"
            )
        self.adb_host = str(mumu.get("adb_host", "127.0.0.1")).strip()
        self.configured_adb_path = str(mumu.get("adb_path", "")).strip()
        if not self.adb_host or any(char.isspace() for char in self.adb_host):
            raise DeviceError("mumu.adb_host 不能为空或包含空格")
        try:
            self.adb_port = int(mumu.get("adb_port", 16384))
        except (TypeError, ValueError) as exc:
            raise DeviceError("mumu.adb_port 必须为 1 到 65535 的整数") from exc
        if not 1 <= self.adb_port <= 65535:
            raise DeviceError("mumu.adb_port 必须为 1 到 65535 的整数")
        self.cli_path = None if self.uses_custom_adb else self._discover_cli()
        self.adb_path = self._discover_adb()
        self.serial: str | None = None

    @property
    def uses_custom_adb(self) -> bool:
        return getattr(self, "connection_mode", "auto") == "custom_adb"

    @property
    def connection_description(self) -> str:
        if self.uses_custom_adb:
            return f"自定义 ADB {self.adb_host}:{self.adb_port}"
        return f"MuMu 自动检测（vmindex={self.vm_index}）"

    @staticmethod
    def _process_kwargs() -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        return kwargs

    @staticmethod
    def _install_root_from_tool(tool: Path) -> Path:
        if tool.parent.name.lower() in {"nx_main", "shell"}:
            return tool.parent.parent
        return tool.parent

    @classmethod
    def _running_install_dirs(cls) -> list[Path]:
        if os.name != "nt":
            return []
        command = (
            "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new();"
            "$ErrorActionPreference='SilentlyContinue';"
            "Get-Process -Name MuMuNxMain,MuMuPlayer,MuMuMultiPlayer "
            "| ForEach-Object {$_.Path}"
        )
        try:
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=6,
                check=False,
                **cls._process_kwargs(),
            )
        except (OSError, subprocess.SubprocessError):
            return []
        roots: list[Path] = []
        for line in result.stdout.splitlines():
            executable = Path(line.strip().strip('"'))
            if executable.name:
                roots.append(cls._install_root_from_tool(executable))
        return roots

    @staticmethod
    def _registry_install_dirs() -> list[Path]:
        if os.name != "nt":
            return []
        try:
            import winreg
        except ImportError:
            return []

        roots: list[Path] = []
        key_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"
        access_modes = (winreg.KEY_READ, winreg.KEY_READ | winreg.KEY_WOW64_32KEY)
        for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            for access in access_modes:
                try:
                    uninstall = winreg.OpenKey(hive, key_path, 0, access)
                except OSError:
                    continue
                with uninstall:
                    for index in range(winreg.QueryInfoKey(uninstall)[0]):
                        try:
                            subkey_name = winreg.EnumKey(uninstall, index)
                            with winreg.OpenKey(uninstall, subkey_name) as subkey:
                                display_name = str(
                                    winreg.QueryValueEx(subkey, "DisplayName")[0]
                                )
                                if "mumu" not in display_name.lower():
                                    continue
                                for value_name in ("InstallLocation", "DisplayIcon"):
                                    try:
                                        raw = str(winreg.QueryValueEx(subkey, value_name)[0])
                                    except OSError:
                                        continue
                                    value = raw.split(",", 1)[0].strip().strip('"')
                                    if not value:
                                        continue
                                    candidate = Path(os.path.expandvars(value))
                                    if candidate.suffix.lower() == ".exe":
                                        candidate = MumuDevice._install_root_from_tool(candidate)
                                    roots.append(candidate)
                        except OSError:
                            continue
        return roots

    def _candidate_install_dirs(self) -> list[Path]:
        candidates: list[Path] = []
        if self.install_dir is not None:
            candidates.append(self.install_dir)
        for variable in ("MUMU_INSTALL_DIR", "MUMU_HOME"):
            value = os.environ.get(variable, "").strip()
            if value:
                candidates.append(Path(os.path.expandvars(value)).expanduser())
        candidates.extend(self._running_install_dirs())
        candidates.extend(self._registry_install_dirs())

        if os.name == "nt":
            drives = [
                Path(f"{letter}:/")
                for letter in string.ascii_uppercase
                if Path(f"{letter}:/").exists()
            ]
        else:
            drives = []
        relative_roots = (
            Path("MuMuPlayer"),
            Path("Netease/MuMuPlayer-12.0"),
            Path("Program Files/Netease/MuMuPlayer-12.0"),
            Path("Program Files/Netease/MuMuPlayer"),
            Path("Program Files (x86)/Netease/MuMuPlayer-12.0"),
        )
        for drive in drives:
            candidates.extend(drive / relative for relative in relative_roots)

        unique: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = os.path.normcase(str(candidate.resolve(strict=False)))
            if normalized not in seen:
                seen.add(normalized)
                unique.append(candidate)
        return unique

    def _discover_cli(self) -> Path:
        for install_dir in self._candidate_install_dirs():
            candidates = (
                install_dir / "nx_main" / "mumu-cli.exe",
                install_dir / "shell" / "mumu-cli.exe",
                install_dir / "mumu-cli.exe",
            )
            for candidate in candidates:
                if candidate.is_file():
                    self.install_dir = install_dir.resolve()
                    return candidate.resolve()
        found = shutil.which("mumu-cli")
        if found:
            candidate = Path(found).resolve()
            self.install_dir = self._install_root_from_tool(candidate)
            return candidate
        raise DeviceError(
            "找不到 mumu-cli.exe；已自动检查运行进程、注册表、各磁盘常见目录和 PATH。"
            "如为便携版，可设置环境变量 MUMU_INSTALL_DIR。"
        )

    def _discover_adb(self) -> Path:
        candidates: list[Path] = []
        if self.configured_adb_path:
            candidates.append(
                Path(os.path.expandvars(self.configured_adb_path)).expanduser()
            )
        for variable in ("ANDROID_SDK_ROOT", "ANDROID_HOME"):
            sdk_root = os.environ.get(variable)
            if sdk_root:
                candidates.append(Path(sdk_root) / "platform-tools" / "adb.exe")
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            candidates.append(
                Path(local_app_data) / "Android" / "Sdk" / "platform-tools" / "adb.exe"
            )
        if self.cli_path is not None:
            candidates.append(self.cli_path.parent / "adb.exe")
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        # Custom ADB mode may have no MuMu install or CLI at all. Explicit
        # platform-tools work independently; MuMu paths are optional fallbacks.
        roots = [self.install_dir] if self.install_dir is not None else self._candidate_install_dirs()
        for root in roots:
            for candidate in (
                root / "nx_main" / "adb.exe",
                root / "shell" / "adb.exe",
                *sorted(root.glob("nx_device/*/shell/adb.exe")),
            ):
                if candidate.is_file():
                    return candidate
        found = shutil.which("adb")
        if found:
            return Path(found)
        raise DeviceError(
            "找不到 adb.exe；请设置 mumu.install_dir 或 mumu.adb_path，"
            "或将 Android platform-tools 加入 PATH"
        )

    def _run(
        self,
        command: list[str],
        *,
        timeout: float = 30,
        binary: bool = False,
        check: bool = True,
    ) -> subprocess.CompletedProcess[Any]:
        text_options: dict[str, Any] = {}
        if not binary:
            # MuMu CLI and Android shell both emit UTF-8. Windows' default GBK
            # decoder can otherwise fail on device names or Android output.
            text_options.update(text=True, encoding="utf-8", errors="replace")
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            **text_options,
            **self._process_kwargs(),
        )
        if check and result.returncode != 0:
            stderr = result.stderr if isinstance(result.stderr, str) else result.stderr.decode(errors="replace")
            raise DeviceError(f"命令失败 ({result.returncode}): {' '.join(command)}\n{stderr.strip()}")
        return result

    def info(self) -> dict[str, Any]:
        if self.uses_custom_adb:
            serial = f"{self.adb_host}:{self.adb_port}"
            result = self._run(
                [str(self.adb_path), "-s", serial, "get-state"],
                timeout=8,
                check=False,
            )
            return {
                "is_android_started": str(result.stdout).strip() == "device",
                "adb_host_ip": self.adb_host,
                "adb_port": self.adb_port,
                "connection_mode": self.connection_mode,
            }
        if self.cli_path is None:  # pragma: no cover - constructor guarantees this
            raise DeviceError("MuMu 自动检测模式缺少 mumu-cli.exe")
        result = self._run(
            [str(self.cli_path), "info", "--vmindex", str(self.vm_index)],
            timeout=15,
        )
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise DeviceError(f"无法解析 MuMu 设备信息：{result.stdout}") from exc

    def ensure_running(self) -> dict[str, Any]:
        info = self.info()
        if info.get("is_android_started"):
            return info
        if self.uses_custom_adb:
            raise DeviceError(
                f"自定义 ADB 设备未上线：{self.adb_host}:{self.adb_port}；"
                "请确认模拟器已启动且 ADB 调试已开启"
            )
        if not self.auto_launch:
            raise DeviceError("MuMu Android 设备未启动，且 auto_launch=false")
        self._run(
            [
                str(self.cli_path),
                "control",
                "--vmindex",
                str(self.vm_index),
                "launch",
            ],
            timeout=20,
        )
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            time.sleep(2)
            info = self.info()
            if info.get("is_android_started"):
                return info
        raise DeviceError(f"MuMu 在 {self.startup_timeout_s:.0f} 秒内未完成启动")

    def connect(self) -> str:
        if self.uses_custom_adb:
            host = self.adb_host
            port = self.adb_port
            use_mumu_cli = False
            timeout_s = max(5.0, min(20.0, self.startup_timeout_s))
        else:
            info = self.ensure_running()
            host = str(info.get("adb_host_ip", "127.0.0.1"))
            port = int(info.get("adb_port", 16384))
            use_mumu_cli = True
            timeout_s = max(10.0, min(45.0, self.startup_timeout_s))
        self.serial = f"{host}:{port}"
        deadline = time.monotonic() + timeout_s
        last_detail = "尚未返回设备状态"
        attempts = 0
        while time.monotonic() < deadline:
            attempts += 1
            try:
                cli_details = ""
                if use_mumu_cli:
                    if self.cli_path is None:  # pragma: no cover - defensive
                        raise DeviceError("MuMu 自动检测模式缺少 mumu-cli.exe")
                    cli_result = self._run(
                        [
                            str(self.cli_path),
                            "adb",
                            "--vmindex",
                            str(self.vm_index),
                            "--cmd",
                            "connect",
                        ],
                        timeout=8,
                        check=False,
                    )
                    cli_details = "；".join(
                        value
                        for value in (
                            str(cli_result.stdout).strip(),
                            str(cli_result.stderr).strip(),
                        )
                        if value
                    )
                direct_result = self._run(
                    [str(self.adb_path), "connect", self.serial],
                    timeout=8,
                    check=False,
                )
                state_result = self._run(
                    [str(self.adb_path), "-s", self.serial, "get-state"],
                    timeout=8,
                    check=False,
                )
                state = str(state_result.stdout).strip()
                if state == "device":
                    boot_result = self._run(
                        [
                            str(self.adb_path),
                            "-s",
                            self.serial,
                            "shell",
                            "getprop",
                            "sys.boot_completed",
                        ],
                        timeout=8,
                        check=False,
                    )
                    if str(boot_result.stdout).strip() == "1":
                        return self.serial
                    last_detail = "ADB 已上线，Android 仍在启动"
                else:
                    details = [
                        state or "无 get-state 输出",
                        str(direct_result.stdout).strip(),
                        str(direct_result.stderr).strip(),
                        cli_details,
                    ]
                    last_detail = "；".join(value for value in details if value)
            except (OSError, subprocess.SubprocessError) as exc:
                last_detail = str(exc)
            if time.monotonic() < deadline:
                time.sleep(1.0)
        raise DeviceError(
            f"ADB 连接超时：{self.serial}，重试 {attempts} 次；"
            f"最后状态：{last_detail}"
        )

    def adb(
        self,
        arguments: list[str],
        *,
        timeout: float = 30,
        binary: bool = False,
        check: bool = True,
    ) -> Any:
        if not self.serial:
            raise DeviceError("ADB 尚未连接")
        result = self._run(
            [str(self.adb_path), "-s", self.serial, *arguments],
            timeout=timeout,
            binary=binary,
            check=check,
        )
        return result.stdout

    def screenshot(self) -> Image.Image:
        arguments = ["exec-out", "screencap", "-p"]
        try:
            raw = self.adb(arguments, timeout=20, binary=True)
        except (DeviceError, subprocess.TimeoutExpired):
            # MuMu's local ADB transport can stall while the emulator and game
            # remain healthy. Screenshots are read-only, so reconnect and retry
            # once before ending a live experiment.
            self.connect()
            try:
                raw = self.adb(arguments, timeout=20, binary=True)
            except subprocess.TimeoutExpired as exc:
                raise DeviceError("ADB 截图重连后仍然超时") from exc
        try:
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            raise DeviceError("ADB 截图解码失败") from exc

    def package_installed(self, package: str) -> bool:
        # Android's package manager returns exit code 1 when a package is
        # absent; that is a normal diagnostic result, not a transport error.
        output = self.adb(["shell", "pm", "path", package], timeout=20, check=False)
        return "package:" in output

    def launch_package(self, package: str) -> None:
        if not self.package_installed(package):
            raise DeviceError(f"未检测到游戏包：{package}")
        self.adb(
            [
                "shell",
                "monkey",
                "-p",
                package,
                "-c",
                "android.intent.category.LAUNCHER",
                "1",
            ],
            timeout=20,
        )

    def foreground_package(self) -> str | None:
        arguments = ["shell", "dumpsys", "activity", "activities"]
        try:
            output = self.adb(arguments, timeout=20)
        except DeviceError:
            # MuMu's local ADB transport can briefly return exit code 1 while the
            # emulator itself remains healthy.  This probe is read-only, so it is
            # safe to reconnect and retry once instead of stopping the whole run.
            self.connect()
            output = self.adb(arguments, timeout=20)
        patterns = [
            r"mResumedActivity:.*? ([A-Za-z0-9_.]+)/",
            r"topResumedActivity=.*? ([A-Za-z0-9_.]+)/",
        ]
        for pattern in patterns:
            match = re.search(pattern, output)
            if match:
                return match.group(1)
        return None

    def tap_pixel(self, x: int, y: int) -> None:
        self.adb(["shell", "input", "tap", str(int(x)), str(int(y))], timeout=10)

    def tap_normalized(self, point: list[float] | tuple[float, float], size: tuple[int, int]) -> tuple[int, int]:
        width, height = size
        x = max(0, min(width - 1, round(float(point[0]) * width)))
        y = max(0, min(height - 1, round(float(point[1]) * height)))
        self.tap_pixel(x, y)
        return x, y
