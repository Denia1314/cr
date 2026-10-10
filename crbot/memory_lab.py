"""Read-only Android access diagnosis and an isolated, non-game memory fixture."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import queue
import re
import subprocess
import threading
import time
from typing import Any

from . import release_label
from .atomic_file import atomic_write

PACKAGE = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)+")
MAX_PACKET = 4096
GAME_PACKAGES = ("com.tencent.tmgp.supercell.clashroyale", "com.supercell.clashroyale")


class LabError(ValueError):
    pass


class ReadOnlyAdb:
    """Never calls connect/launch/root, changes a setting, or injects into a game."""

    def __init__(self, path: Path, serial: str):
        self.path, self.serial = path, serial

    def command(self, *args: str) -> list[str]:
        return [str(self.path), "-s", self.serial, *args]

    def run(self, *args: str, timeout: float = 10) -> subprocess.CompletedProcess[str]:
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if hasattr(subprocess, "CREATE_NO_WINDOW") else {}
        return subprocess.run(self.command(*args), text=True, encoding="utf-8", errors="replace",
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, **options)

    def text(self, *args: str) -> str:
        result = self.run(*args)
        if result.returncode:
            raise LabError("ADB 只读查询失败，请检查已连接设备")
        return result.stdout.strip()


def parse_start_ticks(stat: str) -> int:
    # comm can contain spaces and parentheses; field 3 starts after its final ')'.
    end = stat.rfind(")")
    if end < 0:
        raise LabError("进程身份不可读取")
    tail = stat[end + 1:].split()
    if len(tail) <= 19 or not tail[19].isdigit() or int(tail[19]) <= 0:
        raise LabError("进程启动标识无效")
    return int(tail[19])


def resolve_device(config_path: Path, adb_path: str | None, serial: str | None) -> ReadOnlyAdb:
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    mumu = config.get("mumu", {})
    if adb_path:
        path = Path(adb_path).expanduser().resolve()
    else:
        from .adb import MumuDevice
        path = MumuDevice({"mumu": mumu, "automation": {}}).adb_path
    if not path.is_file():
        raise LabError("未找到 ADB，请指定 --adb 路径")
    reader = ReadOnlyAdb(path, serial or "")
    result = reader.run("devices") if serial else subprocess.run(
        [str(path), "devices"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
        **({"creationflags": subprocess.CREATE_NO_WINDOW} if hasattr(subprocess, "CREATE_NO_WINDOW") else {}),
    )
    if result.returncode:
        raise LabError("无法查询已连接的 ADB 设备")
    available = [parts[0] for line in result.stdout.splitlines()
                 if len(parts := line.split()) >= 2 and parts[1] == "device"]
    preferred = str(mumu.get("adb_host", "127.0.0.1")) + ":" + str(mumu.get("adb_port", 16384))
    selected = serial or (preferred if preferred in available else available[0] if len(available) == 1 else None)
    if not selected or selected not in available:
        raise LabError("目标设备未连接或设备选择不明确；请指定 --serial，不会自动启动或连接模拟器")
    return ReadOnlyAdb(path, selected)


def deploy_fixture(adb: ReadOnlyAdb, binary: Path, abi: str) -> str:
    data = binary.read_bytes()
    machine = {"x86_64": 62, "arm64-v8a": 183}.get(abi)
    if len(data) < 20 or data[:6] != b"\x7fELF\x02\x01" or int.from_bytes(data[18:20], "little") != machine:
        raise LabError("测试采集器不是匹配客体系统架构的 64 位 ELF 文件")
    digest = hashlib.sha256(data).hexdigest()
    directory = "/data/local/tmp/royal-lab-memory"
    remote = directory + "/fixture-" + digest
    for args in (("shell", "mkdir", "-p", directory), ("push", str(binary), remote),
                 ("shell", "chmod", "700", remote)):
        if adb.run(*args, timeout=30).returncode:
            raise LabError("无法准备独立测试采集器；没有修改模拟器权限")
    checked = adb.text("shell", "sha256sum", remote).split()
    if not checked or checked[0] != digest:
        raise LabError("Android 测试采集器摘要核验失败")
    return remote


def package_metadata(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    patterns = {"version_name": r"\bversionName=([^\s]+)", "version_code": r"\bversionCode=(\d+)",
                "declared_primary_abi": r"\bprimaryCpuAbi=([^\s]+)"}
    for name, pattern in patterns.items():
        match = re.search(pattern, text)
        result[name] = match.group(1) if match else None
    flags = re.findall(r"^[ \t]*(?:flags|pkgFlags)=\[([^\]]*)\]", text, flags=re.MULTILINE)
    result["debuggable"] = any("DEBUGGABLE" in row.split() for row in flags) if flags else None
    return result


def diagnose(adb: ReadOnlyAdb, package: str | None, collector: Path | None, sample_header: bool = False) -> dict[str, Any]:
    if sample_header and collector is None:
        raise LabError("限定模块头检查需要显式指定 --collector")
    candidates = (package,) if package else GAME_PACKAGES
    for value in candidates:
        if not PACKAGE.fullmatch(value):
            raise LabError("包名只能包含合法 Android 包名字符")
    installed = set()
    for candidate in candidates:
        listed = adb.text("shell", "pm", "list", "packages", candidate)
        installed.update(line.removeprefix("package:").strip() for line in listed.splitlines() if line.startswith("package:"))
    target = next((candidate for candidate in candidates if candidate in installed), None)
    identity = adb.text("shell", "id")
    abi = adb.text("shell", "getprop", "ro.product.cpu.abi")
    report: dict[str, Any] = {"schema": "memory_lab.diagnosis.v1", "release": release_label(),
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "serial": adb.serial,
        "android": adb.text("shell", "getprop", "ro.build.version.release"), "collector_abi": abi,
        "supported_abis": adb.text("shell", "getprop", "ro.product.cpu.abilist"),
        "shell_identity": identity, "selinux": adb.text("shell", "getenforce"), "package": target,
        "game_memory_read": False, "game_fields_verified": False, "settings_changed": False,
        "status": "package_missing" if not target else "process_missing"}
    if not target:
        return report
    report.update(package_metadata(adb.text("shell", "dumpsys", "package", target)))
    pids = adb.run("shell", "pidof", target).stdout.split()
    if len(pids) != 1 or not pids[0].isdigit() or int(pids[0]) <= 0:
        report["status"] = "process_missing" if not pids else "process_ambiguous"
        return report
    pid = int(pids[0])
    stat = adb.run("shell", "cat", f"/proc/{pid}/stat")
    try:
        started = parse_start_ticks(stat.stdout)
    except LabError:
        report["status"] = "process_identity_unavailable"
        return report
    report.update(pid=pid, process_start_ticks=started)
    if collector:
        remote = deploy_fixture(adb, collector, abi)
        extra = ("sample-platform-header",) if sample_header else ()
        probed = adb.run("exec-out", remote, "probe", str(pid), str(started), *extra)
        try:
            access = json.loads(probed.stdout)
        except (ValueError, TypeError) as exc:
            raise LabError("采集器未返回完整访问诊断") from exc
        if access.get("schema") != "memory_lab.probe.v1" or type(access.get("actual_memory_read")) is not bool:
            raise LabError("访问诊断协议无效")
        if not sample_header and access["actual_memory_read"]:
            raise LabError("默认访问诊断不允许读取内存字节")
        report["access"] = access
        report["game_memory_read"] = access["actual_memory_read"]
        matches = probed.returncode == 0 and access.get("identity_matches") is True
        report["status"] = ("access_open_only" if access.get("maps_opened") and access.get("memory_file_opened")
                            else "access_blocked") if matches else "process_changed"
        if sample_header and matches:
            verified = access.get("platform_header_verified") is True and access.get("sample_bytes_read") == 16 and access["actual_memory_read"]
            report["status"] = "platform_header_verified" if verified else "platform_header_unverified"
    else:
        # Only open read-only descriptors. Never read target memory or print its contents.
        maps = adb.run("shell", "sh", "-c", f"'exec 3< /proc/{pid}/maps'")
        memory = adb.run("shell", "sh", "-c", f"'exec 3< /proc/{pid}/mem'")
        after = adb.run("shell", "cat", f"/proc/{pid}/stat")
        try:
            matches = started == parse_start_ticks(after.stdout)
        except LabError:
            matches = False
        report["access"] = {"maps_opened": maps.returncode == 0, "memory_file_opened": memory.returncode == 0,
                            "identity_matches": matches, "actual_memory_read": False, "probe": "shell_readonly_open"}
        report["status"] = ("access_open_only" if maps.returncode == 0 and memory.returncode == 0
                            else "access_blocked") if matches else "process_changed"
    return report


class FixtureReceiver:
    """Strict test-only protocol: these packets can never become a GameState."""

    def __init__(self, count: int):
        self.count, self.sequence = count, 0
        self.session: str | None = None
        self.last_end = 0
        self.read_ms: list[float] = []
        self.host_arrivals: list[float] = []
        self.summary: dict[str, Any] | None = None

    def accept(self, raw: bytes, received: float) -> None:
        if len(raw) > MAX_PACKET:
            raise LabError("测试数据包超过大小限制")
        try:
            packet = json.loads(raw.decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except (ValueError, UnicodeError) as exc:
            raise LabError("测试数据包损坏") from exc
        if not isinstance(packet, dict) or self.summary is not None:
            raise LabError("测试数据顺序无效")
        if packet.get("schema") == "memory_lab.summary.v1":
            required = ("passed", "bad_address_rejected", "partial_read_rejected", "exited_child_rejected")
            if any(packet.get(key) is not True for key in required) or packet.get("game_memory_read") is not False:
                raise LabError("Android 测试采集器未通过异常读取检查")
            if type(packet.get("samples")) is not int or packet["samples"] != self.count or self.sequence != self.count:
                raise LabError("测试采集缺帧")
            self.summary = packet
            return
        if packet.get("schema") != "memory_lab.fixture.v1" or packet.get("source") != "private_child_fixture":
            raise LabError("不接受游戏观测或未知来源进入测试链路")
        keys = ("sequence", "read_begin_ns", "read_end_ns", "elixir_milli")
        if any(type(packet.get(key)) is not int for key in keys):
            raise LabError("测试字段类型无效")
        seq, begin, end = packet["sequence"], packet["read_begin_ns"], packet["read_end_ns"]
        session = packet.get("session")
        if not isinstance(session, str) or not re.fullmatch(r"\d{1,12}-\d{1,20}", session):
            raise LabError("测试进程会话无效")
        if self.session is not None and self.session != session:
            raise LabError("测试进程会话改变")
        if seq != self.sequence + 1 or seq > self.count or begin < self.last_end or begin <= 0 or end < begin:
            raise LabError("测试序号或时间倒退")
        if packet.get("complete") is not True or packet.get("consistent") is not True:
            raise LabError("拒绝不完整或不一致的测试观测")
        hand = packet.get("hand")
        if not isinstance(hand, list) or len(hand) != 4 or any(type(card) is not int for card in hand):
            raise LabError("测试手牌结构无效")
        if packet["elixir_milli"] != (seq * 137) % 10001 or hand != [(seq + slot) % 8 + 1 for slot in range(4)]:
            raise LabError("读取的已知测试值不匹配")
        self.session, self.sequence, self.last_end = session, seq, end
        self.read_ms.append((end - begin) / 1_000_000)
        self.host_arrivals.append(received)

    def finish(self, returncode: int) -> dict[str, Any]:
        if returncode != 0 or self.summary is None or self.sequence != self.count:
            raise LabError("测试采集进程异常退出或缺少完成报告")
        ordered = sorted(self.read_ms)
        return {"schema": "memory_lab.fixture_report.v1", "status": "fixture_verified", "release": release_label(),
            "samples": self.sequence, "source": "private_child_fixture", "backend": "process_vm_readv",
            "read_ms_p95": ordered[math.ceil(len(ordered) * .95) - 1],
            "host_delivery_span_s": self.host_arrivals[-1] - self.host_arrivals[0],
            "end_to_end_age_verified": False, "game_memory_read": False, "game_fields_verified": False,
            "native_checks": self.summary,
            "note": "仅验证自建子进程的已知测试数据；客体与主机时钟未校准，不宣称游戏读取或端到端延迟"}


def self_test(adb: ReadOnlyAdb, binary: Path, samples: int, interval_ms: int) -> dict[str, Any]:
    if not 1 <= samples <= 600 or not 10 <= interval_ms <= 1000 or samples * interval_ms > 60000:
        raise LabError("测试范围：1–600 个样本、10–1000ms 间隔，总时长不超过 60 秒")
    abi = adb.text("shell", "getprop", "ro.product.cpu.abi")
    remote = deploy_fixture(adb, binary, abi)
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if hasattr(subprocess, "CREATE_NO_WINDOW") else {}
    process = subprocess.Popen(adb.command("exec-out", remote, "fixture", str(samples), str(interval_ms)),
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, **options)
    events: queue.Queue[Any] = queue.Queue(maxsize=samples + 4)

    def pump() -> None:
        try:
            assert process.stdout is not None
            for _ in range(samples + 2):
                raw = process.stdout.readline(MAX_PACKET + 1)
                if not raw:
                    events.put(None)
                    return
                events.put((time.monotonic(), raw))
                if len(raw) > MAX_PACKET:
                    return
            events.put(LabError("测试流包含多余数据包"))
        except (OSError, ValueError) as exc:
            events.put(LabError("测试传输中断：" + type(exc).__name__))

    thread = threading.Thread(target=pump, name="memory-lab-receiver", daemon=True)
    thread.start()
    receiver = FixtureReceiver(samples)
    deadline = time.monotonic() + samples * interval_ms / 1000 + 10
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LabError("测试采集超时")
            try:
                event = events.get(timeout=remaining)
            except queue.Empty as exc:
                raise LabError("测试采集超时") from exc
            if event is None:
                break
            if isinstance(event, Exception):
                raise event
            received, raw = event
            receiver.accept(raw, received)
        return receiver.finish(process.wait(timeout=2))
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        thread.join(timeout=2)
        if process.stdout:
            process.stdout.close()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("action", choices=("diagnose", "self-test"))
    parser.add_argument("--adb", help="指定现有 ADB，不自动安装或启动模拟器")
    parser.add_argument("--serial", help="已连接的设备序列号")
    parser.add_argument("--package", help="诊断包名；默认自动识别腾讯渠道包或国际服")
    parser.add_argument("--collector", type=Path, help="已编译的独立测试采集器 ELF")
    parser.add_argument("--sample-platform-header", action="store_true", help="仅诊断：显式读取平台模块头16字节，不读取游戏字段")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--interval-ms", type=int, default=100)
    parser.add_argument("--output", type=Path, help="报告必须保存在项目 reports 目录内")


def run_arguments(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    report_dir = config_path.parent / "reports"
    output = args.output.resolve() if args.output else report_dir / f"memory-lab-{args.action}.json"
    if not output.is_relative_to(report_dir.resolve()):
        raise LabError("本地诊断报告只能写入项目 reports 目录，避免上传机器信息")
    try:
        adb = resolve_device(config_path, args.adb, args.serial)
        abi = adb.text("shell", "getprop", "ro.product.cpu.abi")
        binary = args.collector or config_path.parent / ".memory-lab/bin" / ("collector-" + abi)
        if args.action == "self-test":
            if args.sample_platform_header:
                raise LabError("限定模块头检查仅适用于 diagnose")
            if not binary.is_file():
                raise LabError("请先使用 tools/build_memory_lab.py 编译测试采集器")
            report = self_test(adb, binary, args.samples, args.interval_ms)
            code = 0
        else:
            if args.collector and not args.collector.is_file():
                raise LabError("指定的测试采集器文件不存在")
            report = diagnose(adb, args.package, args.collector, args.sample_platform_header)
            code = 0 if report["status"] in {"access_open_only", "platform_header_verified"} else 2
    except (LabError, OSError, subprocess.SubprocessError) as exc:
        report = {"schema": "memory_lab.failure.v1", "release": release_label(), "status": "failed",
                  "reason": str(exc), "game_memory_read": False, "game_fields_verified": False}
        code = 2
    atomic_write(output, (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("本地报告：" + str(output))
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="只读内存访问诊断与自建测试进程采集；不读取游戏字段")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--version", action="version", version=release_label())
    add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        return run_arguments(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
