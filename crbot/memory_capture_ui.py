"""Local manual research controls; never feeds the battle decision path."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path
import queue
import subprocess
import threading
import time
import tkinter as tk
from tkinter import ttk

from . import release_label
from .memory_lab import PACKAGE, ReadOnlyAdb, deploy_fixture, parse_start_ticks


class CaptureSession:
    def __init__(self, adb: ReadOnlyAdb, package: str, binary: Path, reports: Path, screenshots: bool = True):
        self.adb, self.package, self.binary, self.reports = adb, package, binary, reports
        self.screenshots = screenshots
        self.events: queue.Queue = queue.Queue()
        self.commands: queue.Queue = queue.Queue()
        self.folder: Path | None = None
        self.thread: threading.Thread | None = None

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def command(self, name: str, cost: int | None = None):
        if name not in {"trigger", "mark", "stop"} or (name == "mark" and cost not in range(1, 11)):
            raise ValueError("无效采集操作")
        self.commands.put((name, cost, time.monotonic()))

    def _run(self):
        process = None
        stop_screens = threading.Event()
        screen_thread = None
        stopping = False
        try:
            if not PACKAGE.fullmatch(self.package):
                raise ValueError("游戏包名格式无效")
            abi = self.adb.text("shell", "getprop", "ro.product.cpu.abi")
            remote = deploy_fixture(self.adb, self.binary, abi)
            pid = self.adb.text("shell", "pidof", self.package)
            if not pid.isdigit():
                raise ValueError("没有找到唯一游戏进程，请检查实例与包名")
            started = parse_start_ticks(self.adb.text("shell", "cat", f"/proc/{pid}/stat"))
            calibrations = []
            for _ in range(3):
                begin = time.monotonic()
                guest = int(self.adb.text("shell", "-T", remote, "clock")) / 1e9
                end = time.monotonic()
                calibrations.append({"offset": (begin + end) / 2 - guest, "uncertainty": (end - begin) / 2})
            self.folder = self.reports / ("manual-memory-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
            self.folder.mkdir(parents=True)
            metadata = dict(pid=pid, start_ticks=started, calibration=min(calibrations, key=lambda c: c["uncertainty"]),
                            release=release_label(), game_fields_verified=False, status="recording")
            (self.folder / "session.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            def screens():
                with (self.folder / "screens.jsonl").open("w", encoding="utf-8") as log:
                    frame = 0
                    while not stop_screens.is_set():
                        begin = time.monotonic()
                        try:
                            with (self.folder / f"screen-{frame:03d}.png").open("wb") as output:
                                result = subprocess.run(self.adb.command("exec-out", "screencap", "-p"), stdout=output,
                                                        stderr=subprocess.DEVNULL, timeout=5)
                            log.write(json.dumps(dict(frame=frame, begin=begin, end=time.monotonic(), ok=result.returncode == 0)) + "\n")
                            log.flush()
                        except (OSError, subprocess.SubprocessError):
                            self.events.put(("notice", "参考截图失败，内存采集仍在继续"))
                        frame += 1
                        stop_screens.wait(.5)
            if self.screenshots:
                screen_thread = threading.Thread(target=screens, daemon=True)
                screen_thread.start()
            process = subprocess.Popen(self.adb.command("shell", "-T", remote, pid, str(started), "triggered"),
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            packets: queue.Queue = queue.Queue(maxsize=8)
            def reader():
                try:
                    for line in iter(lambda: process.stdout.readline(4_000_001), b""):
                        if len(line) > 4_000_000:
                            raise ValueError("采集包超过大小限制")
                        packets.put((line, json.loads(line)))
                except Exception as exc:
                    packets.put(exc)
                finally:
                    packets.put(None)
            threading.Thread(target=reader, daemon=True).start()
            phase, footer, samples = "baseline", None, 0
            deadline = time.monotonic() + 210
            stop_deadline = None
            with (self.folder / "candidates.jsonl").open("wb") as output, (self.folder / "marks.jsonl").open("w", encoding="utf-8") as marks:
                while True:
                    now = time.monotonic()
                    if now > deadline or (stop_deadline and now > stop_deadline):
                        raise TimeoutError("采集结束超时，已保存收到的记录")
                    try:
                        name, cost, clicked = self.commands.get_nowait()
                    except queue.Empty:
                        pass
                    else:
                        if name == "stop":
                            stopping = True
                            process.stdin.write(b"s"); process.stdin.flush()
                            stop_deadline = now + 8
                        elif name == "trigger" and phase == "ready":
                            process.stdin.write(b"\n"); process.stdin.flush()
                            phase = "arming"
                            self.events.put(("arming", "正在启动，请等窗口显示可以出牌"))
                        elif name == "mark" and phase == "active":
                            marks.write(json.dumps(dict(cost=cost, clicked_at=clicked)) + "\n"); marks.flush()
                            self.events.put(("notice", f"已记录 {cost} 费出牌标记，请等待圣水恢复"))
                    try:
                        item = packets.get(timeout=.05)
                    except queue.Empty:
                        continue
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        raise item
                    raw, packet = item
                    output.write(raw); output.flush()
                    event = packet.get("event")
                    if event == "stable_ready":
                        phase = "ready"
                        self.events.put(("ready", "基准就绪：点击“开始出牌采集”（请在75秒内操作）"))
                    elif event == "discovery_started":
                        phase = "active"
                        self.events.put(("active", "现在可以出一张牌；随后选择费用并点击“标记已出牌”"))
                    elif event == "raw_sample":
                        samples += 1
                        self.events.put(("progress", f"已采集 {samples} 点 · 候选 {len(packet['values'])}" + (" · 范围已截断" if packet.get("discovery_truncated") else "")))
                    elif event == "raw_candidates":
                        footer = packet
            code = process.wait(timeout=3)
            if not stopping and (code or footer is None):
                reason = process.stderr.read(2048).decode("utf-8", errors="replace").strip()
                raise ValueError(reason or "采集未正常结束；可能等待触发超时")
            metadata.update(status="stopped" if stopping else "complete", samples=samples,
                            discovery_truncated=footer.get("discovery_truncated") if footer else None)
            (self.folder / "session.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            self.events.put(("done", ("已停止" if stopping else "采集完成") + "，报告已保存在本机"))
        except Exception as exc:
            self.events.put(("error", str(exc)))
            if self.folder:
                (self.folder / "failure.json").write_text(json.dumps(dict(error=str(exc), game_fields_verified=False)), encoding="utf-8")
        finally:
            stop_screens.set()
            if process and process.poll() is None:
                try:
                    process.stdin.write(b"s"); process.stdin.flush(); process.wait(timeout=3)
                except (OSError, subprocess.SubprocessError):
                    process.kill(); process.wait(timeout=3)
            if screen_thread:
                screen_thread.join(timeout=6)
            if process and process.stdin:
                process.stdin.close()


class MemoryCaptureWindow:
    def __init__(self, root: tk.Tk, adb: Path, serial: str, package: str, binary: Path, reports: Path):
        self.root, self.adb_path, self.binary, self.reports = root, adb, binary, reports
        self.session = None
        self.ui_events = queue.Queue()
        root.title(release_label() + " · 手动内存采集")
        root.geometry("760x480"); root.minsize(680, 430)
        frame = ttk.Frame(root, padding=18); frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="手动内存采集", font=("Microsoft YaHei UI", 18, "bold")).pack(anchor="w")
        ttk.Label(frame, text="按钮直接控制本机采集，无需等待聊天。游戏字段尚未验证。", wraplength=620).pack(anchor="w", pady=(6, 14))
        row = ttk.Frame(frame); row.pack(fill="x")
        ttk.Label(row, text="测试设备：").pack(side="left")
        self.device = tk.StringVar(value=serial)
        self.devices = ttk.Combobox(row, textvariable=self.device, width=28, state="readonly"); self.devices.pack(side="left", padx=6)
        self.refresh = ttk.Button(row, text="刷新设备", command=self.refresh_devices); self.refresh.pack(side="left")
        self.package = tk.StringVar(value=package)
        ttk.Entry(frame, textvariable=self.package).pack(fill="x", pady=10)
        self.status = tk.StringVar(value="进入测试对局，满10后点击“记录基准”")
        ttk.Label(frame, textvariable=self.status, wraplength=620, font=("Microsoft YaHei UI", 12)).pack(anchor="w", pady=10)
        self.progress = tk.StringVar(value="尚未开始")
        ttk.Label(frame, textvariable=self.progress).pack(anchor="w")
        actions = ttk.Frame(frame); actions.pack(fill="x", pady=16)
        self.baseline = ttk.Button(actions, text="1. 记录基准", command=self.start); self.baseline.pack(side="left", padx=(0, 10))
        self.trigger = ttk.Button(actions, text="2. 开始出牌采集", state="disabled", command=self.begin); self.trigger.pack(side="left", padx=(0, 10))
        self.stop = ttk.Button(actions, text="停止", state="disabled", command=self.stop_capture); self.stop.pack(side="left")
        marking = ttk.Frame(frame); marking.pack(fill="x")
        ttk.Label(marking, text="出牌费用：").pack(side="left")
        self.cost = tk.IntVar(value=3)
        ttk.Spinbox(marking, from_=1, to=10, textvariable=self.cost, width=4).pack(side="left", padx=8)
        self.mark = ttk.Button(marking, text="3. 标记已出牌", state="disabled", command=self.mark_play); self.mark.pack(side="left")
        self.path = tk.StringVar(value="原始数据仅保存在 reports，不上传代码仓库")
        ttk.Label(frame, textvariable=self.path, wraplength=620).pack(anchor="w", pady=14)
        self.refresh_devices()
        root.after(50, self.poll)
        root.protocol("WM_DELETE_WINDOW", self.close)

    def refresh_devices(self):
        self.refresh["state"] = "disabled"
        def worker():
            try:
                result = subprocess.run([str(self.adb_path), "devices"], capture_output=True, text=True, timeout=10)
                if result.returncode:raise OSError("设备查询失败")
                devices = [parts[0] for line in result.stdout.splitlines() if len(parts := line.split()) == 2 and parts[1] == "device"]
                self.ui_events.put(("devices", devices))
            except (OSError, subprocess.SubprocessError):
                self.ui_events.put(("device_error", "设备查询失败或超时，请检查测试实例后刷新"))
        self.refresh_thread=threading.Thread(target=worker,daemon=True)
        self.refresh_thread.start()

    def start(self):
        if not self.device.get():
            self.status.set("请先选择隔离测试设备"); return
        if not self.binary.is_file():
            self.status.set("采集器未编译，请按说明构建 observer"); return
        self.session = CaptureSession(ReadOnlyAdb(self.adb_path, self.device.get()), self.package.get(), self.binary, self.reports)
        self.baseline["state"] = self.refresh["state"] = self.devices["state"] = "disabled"
        self.stop["state"] = "normal"
        self.status.set("正在记录两次满水基准，请先不出牌")
        self.session.start()

    def begin(self):
        self.trigger["state"] = "disabled"
        self.status.set("正在启动采集，请先不出牌")
        self.session.command("trigger")

    def mark_play(self):
        try:
            self.session.command("mark", self.cost.get())
        except (ValueError, tk.TclError):
            self.status.set("费用需填写1到10之间的整数")

    def stop_capture(self):
        if self.session:
            self.session.command("stop")
            self.status.set("正在停止并保存记录")
            self.stop["state"] = self.trigger["state"] = self.mark["state"] = "disabled"

    def poll(self):
        while True:
            try:event,value=self.ui_events.get_nowait()
            except queue.Empty:break
            if not self.session or not self.session.thread or not self.session.thread.is_alive():
                self.refresh["state"]="normal"
            if event=='devices':
                self.devices['values']=value
                if self.device.get() not in value:self.device.set('')
                self.status.set('进入测试对局，满10后点击“记录基准”' if value else '没有已连接设备，请先连接隔离测试实例再刷新')
            else:self.status.set(value)
        if self.session:
            while True:
                try: event, message = self.session.events.get_nowait()
                except queue.Empty: break
                if event == "progress": self.progress.set(message)
                else: self.status.set(message)
                if event == "ready": self.trigger["state"] = "normal"
                if event == "active": self.mark["state"] = "normal"
                if event in {"done", "error"}:
                    self.baseline["state"] = self.refresh["state"] = "normal"
                    self.devices["state"] = "readonly"
                    self.stop["state"] = self.trigger["state"] = self.mark["state"] = "disabled"
                if self.session.folder: self.path.set(str(self.session.folder))
        self.root.after(50, self.poll)

    def close(self):
        if self.session and self.session.thread and self.session.thread.is_alive():
            self.session.command("stop")
            self.root.withdraw()
            self.root.after(100, self.close_when_stopped)
        else:
            self.root.destroy()

    def close_when_stopped(self):
        if self.session.thread.is_alive():
            self.root.after(100, self.close_when_stopped)
        else:
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", type=Path)
    parser.add_argument("--serial", default="")
    parser.add_argument("--package")
    parser.add_argument("--collector", type=Path, default=Path(".memory-lab/bin/observer-x86_64"))
    args = parser.parse_args()
    config = json.loads(Path("config.json").read_text(encoding="utf-8")) if Path("config.json").is_file() else {}
    if args.adb is None:
        from .adb import MumuDevice
        args.adb = MumuDevice(config).adb_path
    args.package = args.package or config.get("game", {}).get("package", "com.supercell.clashroyale")
    root = tk.Tk()
    MemoryCaptureWindow(root, args.adb, args.serial, args.package, args.collector, Path.cwd() / "reports")
    root.mainloop()

if __name__ == "__main__":
    main()
