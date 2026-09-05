from __future__ import annotations

import json
import queue
import re
import subprocess
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from .adb import DeviceError, MumuDevice
from .battle_perception import UniversalHandRecognizer, detect_lane_threats
from .cards import CardCatalog
from .vision import WorkflowRecognizer, battle_ui_score, estimate_elixir


@dataclass(frozen=True)
class TouchscreenInfo:
    path: str
    name: str
    maximum_x: int
    maximum_y: int


@dataclass(frozen=True)
class TouchGesture:
    started_at: float
    ended_at: float
    start_x: int
    start_y: int
    end_x: int
    end_y: int

    def normalized(self, info: TouchscreenInfo) -> tuple[list[float], list[float]]:
        def point(x: int, y: int) -> list[float]:
            return [
                round(min(1.0, max(0.0, x / max(1, info.maximum_x))), 6),
                round(min(1.0, max(0.0, y / max(1, info.maximum_y))), 6),
            ]

        return point(self.start_x, self.start_y), point(self.end_x, self.end_y)


class GetEventTouchParser:
    """Parse Linux multitouch protocol-B events emitted by `getevent -lt`."""

    LINE = re.compile(
        r"\[\s*(?P<time>[0-9.]+)\]\s+(?:\S+:\s+)?"
        r"(?P<type>\S+)\s+(?P<code>\S+)\s+(?P<value>[0-9a-fA-F]+)"
    )

    def __init__(self) -> None:
        self.active = False
        self.started_at = 0.0
        self.start_x: int | None = None
        self.start_y: int | None = None
        self.x: int | None = None
        self.y: int | None = None

    @staticmethod
    def _value(raw: str) -> int:
        value = int(raw, 16)
        return value - (1 << 32) if value >= (1 << 31) else value

    def feed(self, line: str) -> TouchGesture | None:
        match = self.LINE.search(line)
        if match is None:
            return None
        event_time = float(match.group("time"))
        code = match.group("code")
        value = self._value(match.group("value"))
        if code == "ABS_MT_TRACKING_ID":
            if value >= 0:
                self.active = True
                self.started_at = event_time
                self.start_x = self.start_y = self.x = self.y = None
                return None
            if not self.active:
                return None
            self.active = False
            if None in {self.start_x, self.start_y, self.x, self.y}:
                return None
            return TouchGesture(
                started_at=self.started_at,
                ended_at=event_time,
                start_x=int(self.start_x),
                start_y=int(self.start_y),
                end_x=int(self.x),
                end_y=int(self.y),
            )
        if not self.active:
            return None
        if code == "ABS_MT_POSITION_X":
            self.x = value
            if self.start_x is None:
                self.start_x = value
        elif code == "ABS_MT_POSITION_Y":
            self.y = value
            if self.start_y is None:
                self.start_y = value
        return None


def discover_touchscreen(device: MumuDevice) -> TouchscreenInfo:
    output = str(device.adb(["shell", "getevent", "-lp"], timeout=15))
    blocks = re.split(r"(?=add device \d+:)", output)
    for block in blocks:
        if "Touchscreen" not in block or "ABS_MT_POSITION_X" not in block:
            continue
        path_match = re.search(r"add device \d+:\s+(\S+)", block)
        name_match = re.search(r'name:\s+"([^"]+)"', block)
        x_match = re.search(r"ABS_MT_POSITION_X.*?max\s+(-?\d+)", block)
        y_match = re.search(r"ABS_MT_POSITION_Y.*?max\s+(-?\d+)", block)
        if path_match and x_match and y_match:
            return TouchscreenInfo(
                path=path_match.group(1),
                name=name_match.group(1) if name_match else "Touchscreen",
                maximum_x=int(x_match.group(1)),
                maximum_y=int(y_match.group(1)),
            )
    raise DeviceError("未找到可读取的 MuMu 触摸屏输入设备")


@dataclass
class _PendingSelection:
    slot_index: int
    selected_at: float
    before: Image.Image


@dataclass
class _ActionJob:
    slot_index: int
    deploy_point: list[float]
    before: Image.Image
    gesture_type: str
    observed_at: float
    battle_index: int


class DemonstrationRecorder:
    """Observe manual offline battles and record human card-play demonstrations."""

    def __init__(
        self,
        device: MumuDevice,
        config: dict[str, Any],
        config_path: Path,
        *,
        stop_event: threading.Event | None = None,
    ):
        self.device = device
        self.config = config
        self.config_path = config_path.resolve()
        self.project_root = self.config_path.parent
        self.stop_event = stop_event or threading.Event()
        self.touchscreen = discover_touchscreen(device)
        catalog_value = str(config.get("dataset", {}).get("card_catalog", "data/cards.json"))
        catalog_path = Path(catalog_value)
        if not catalog_path.is_absolute():
            catalog_path = self.project_root / catalog_path
        self.catalog = CardCatalog.load(catalog_path.resolve())
        self.hand_recognizer = UniversalHandRecognizer(self.catalog, config["vision"])
        self.workflow = WorkflowRecognizer(config, self.config_path)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.project_root / "demonstrations" / stamp
        collision = 1
        while self.run_dir.exists():
            self.run_dir = self.project_root / "demonstrations" / f"{stamp}_{collision:02d}"
            collision += 1
        self.frames_dir = self.run_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.actions_path = self.run_dir / "actions.jsonl"
        self.raw_touches_path = self.run_dir / "raw_touches.jsonl"
        self.latest_frame: Image.Image | None = None
        self.snapshots: deque[tuple[float, Image.Image]] = deque(maxlen=12)
        self.state_lock = threading.Lock()
        self.file_lock = threading.Lock()
        self.jobs: queue.Queue[_ActionJob | None] = queue.Queue()
        self.in_battle = False
        self.offline_verified = False
        self.offline_streak = 0
        self.battle_absent_streak = 0
        self.battle_index = 0
        self.completed_battles = 0
        self.action_count = 0
        self.raw_touch_count = 0
        self.pending: _PendingSelection | None = None
        self.touch_process: subprocess.Popen[str] | None = None
        self._write_manifest()

    def _write_manifest(self) -> None:
        payload = {
            "schema_version": 1,
            "created_at_unix": time.time(),
            "source": "human_demonstration",
            "allowed_mode": self.config.get("game", {}).get("allowed_mode"),
            "package": self.config.get("game", {}).get("package"),
            "touchscreen": asdict(self.touchscreen),
            "human_actions_are_ground_truth": True,
            "machine_perception_is_ground_truth": False,
        }
        (self.run_dir / "manifest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def request_stop(self) -> None:
        self.stop_event.set()
        if self.touch_process is not None and self.touch_process.poll() is None:
            self.touch_process.terminate()

    def _screen_loop(self) -> None:
        interval = float(self.config.get("demonstration", {}).get("screen_interval_s", 0.35))
        required_gate = int(
            self.config.get("safety", {}).get("offline_gate_consecutive_frames", 3)
        )
        end_frames = int(
            self.config.get("automation", {}).get("battle_end_consecutive_frames", 4)
        )
        package = str(self.config["game"]["package"])
        last_foreground_check = 0.0
        while not self.stop_event.is_set():
            started = time.monotonic()
            try:
                if started - last_foreground_check >= 2.0:
                    last_foreground_check = started
                    if self.device.foreground_package() != package:
                        with self.state_lock:
                            self.offline_verified = False
                            self.offline_streak = 0
                            self.in_battle = False
                        self.stop_event.wait(interval)
                        continue
                image = self.device.screenshot()
                matches = self.workflow.match_all(image)
                gate_visible = any(match.kind == "offline_gate" for match in matches)
                battle_visible = battle_ui_score(
                    image,
                    self.config["automation"]["battle_ui_roi"],
                ) >= float(self.config["automation"].get("battle_ui_threshold", 0.72))
                with self.state_lock:
                    self.latest_frame = image
                    self.snapshots.append((time.monotonic(), image.copy()))
                    if gate_visible:
                        self.offline_streak += 1
                        if self.offline_streak >= required_gate and not self.offline_verified:
                            self.offline_verified = True
                            print("[示范] 离线人机入口验证通过；进入战斗后开始记录操作。")
                    else:
                        self.offline_streak = 0
                    if battle_visible and self.offline_verified:
                        self.battle_absent_streak = 0
                        if not self.in_battle:
                            self.in_battle = True
                            self.battle_index += 1
                            print(f"[示范] 开始记录第 {self.battle_index} 局手动操作。")
                    elif self.in_battle:
                        self.battle_absent_streak += 1
                        if self.battle_absent_streak >= end_frames:
                            self.in_battle = False
                            self.offline_verified = False
                            self.completed_battles += 1
                            self.pending = None
                            print(
                                f"[示范] 第 {self.completed_battles} 局记录结束，"
                                f"累计有效出牌 {self.action_count} 次。"
                            )
            except (DeviceError, OSError, ValueError) as exc:
                print(f"[示范] 截图监控错误：{exc}")
                self.stop_event.set()
                return
            elapsed = time.monotonic() - started
            self.stop_event.wait(max(0.02, interval - elapsed))

    def _latest_snapshot(self) -> Image.Image | None:
        with self.state_lock:
            return self.snapshots[-1][1].copy() if self.snapshots else None

    def _slot_at(self, point: list[float]) -> int | None:
        if point[1] < float(self.config.get("demonstration", {}).get("card_region_top", 0.82)):
            return None
        centers = self.config["vision"]["card_slot_centers"]
        distances = [abs(float(center[0]) - point[0]) for center in centers]
        index = min(range(len(distances)), key=distances.__getitem__)
        return index if distances[index] <= 0.105 else None

    @staticmethod
    def _is_deploy(point: list[float]) -> bool:
        return 0.10 <= point[0] <= 0.90 and 0.30 <= point[1] <= 0.80

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        with self.file_lock, path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _handle_gesture(self, gesture: TouchGesture) -> None:
        received_at = time.monotonic()
        start, end = gesture.normalized(self.touchscreen)
        with self.state_lock:
            in_battle = self.in_battle
            offline_verified = self.offline_verified
            battle_index = self.battle_index
        if not (in_battle and offline_verified):
            return
        before = self._latest_snapshot()
        if before is None:
            return
        slot = self._slot_at(start)
        selection_timeout = float(
            self.config.get("demonstration", {}).get(
                "pending_selection_timeout_s", 5.0
            )
        )
        category = "other"
        completed_job: _ActionJob | None = None
        if slot is not None and self._is_deploy(end):
            category = "card_drag"
            completed_job = _ActionJob(
                slot, end, before, "drag", received_at, battle_index
            )
            self.pending = None
        elif slot is not None:
            category = "card_select"
            self.pending = _PendingSelection(slot, received_at, before)
        elif (
            self.pending is not None
            and received_at - self.pending.selected_at <= selection_timeout
            and self._is_deploy(end)
        ):
            category = "card_deploy"
            completed_job = _ActionJob(
                self.pending.slot_index,
                end,
                self.pending.before,
                "tap_then_tap",
                received_at,
                battle_index,
            )
            self.pending = None
        elif (
            self.pending is not None
            and received_at - self.pending.selected_at > selection_timeout
        ):
            self.pending = None
        self.raw_touch_count += 1
        self._append_jsonl(
            self.raw_touches_path,
            {
                "sequence": self.raw_touch_count,
                "timestamp_unix": time.time(),
                "battle_index": battle_index,
                "category": category,
                "start": start,
                "end": end,
                "duration_s": round(max(0.0, gesture.ended_at - gesture.started_at), 4),
                "offline_verified": True,
            },
        )
        if completed_job is not None:
            self.jobs.put(completed_job)

    def _touch_loop(self) -> None:
        if not self.device.serial:
            raise DeviceError("ADB 尚未连接")
        command = [
            str(self.device.adb_path),
            "-s",
            self.device.serial,
            "shell",
            "getevent",
            "-lt",
            self.touchscreen.path,
        ]
        self.touch_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **self.device._process_kwargs(),
        )
        parser = GetEventTouchParser()
        assert self.touch_process.stdout is not None
        for line in self.touch_process.stdout:
            if self.stop_event.is_set():
                break
            gesture = parser.feed(line)
            if gesture is not None:
                self._handle_gesture(gesture)

    def _action_worker(self) -> None:
        delay = float(self.config.get("demonstration", {}).get("after_action_delay_s", 0.45))
        while True:
            job = self.jobs.get()
            if job is None:
                return
            self.stop_event.wait(delay)
            after = self._latest_snapshot() or job.before.copy()
            matches = self.hand_recognizer.recognize(job.before)
            selected = matches[job.slot_index]
            threats = detect_lane_threats(job.before)
            elixir, elixir_confidence = estimate_elixir(
                job.before, self.config["vision"]["elixir_roi"]
            )
            self.action_count += 1
            stem = f"b{job.battle_index:03d}_a{self.action_count:05d}"
            before_path = self.frames_dir / f"{stem}_before.jpg"
            after_path = self.frames_dir / f"{stem}_after.jpg"
            job.before.save(before_path, format="JPEG", quality=92, optimize=True)
            after.save(after_path, format="JPEG", quality=92, optimize=True)
            payload = {
                "schema_version": 1,
                "action_id": f"{self.run_dir.name}-{stem}",
                "timestamp_unix": time.time(),
                "source": "human_demonstration",
                "human_action_ground_truth": True,
                "machine_perception_ground_truth": False,
                "offline_verified": True,
                "battle_index": job.battle_index,
                "gesture_type": job.gesture_type,
                "slot_index": job.slot_index,
                "deploy_point": job.deploy_point,
                "selected_card_id": selected.card_id,
                "selected_card_confidence": selected.confidence,
                "hand": [match.card_id for match in matches],
                "hand_matches": [match.to_dict() for match in matches],
                "elixir": elixir,
                "elixir_confidence": round(float(elixir_confidence), 5),
                "threats": {lane: value.to_dict() for lane, value in threats.items()},
                "before_frame": str(before_path.relative_to(self.run_dir)).replace("\\", "/"),
                "after_frame": str(after_path.relative_to(self.run_dir)).replace("\\", "/"),
                "screen_size": [job.before.width, job.before.height],
            }
            self._append_jsonl(self.actions_path, payload)
            card_text = selected.card_id or "unknown"
            print(
                f"[示范] 已学习出牌 #{self.action_count}："
                f"slot={job.slot_index + 1} card={card_text} "
                f"deploy=({job.deploy_point[0]:.3f},{job.deploy_point[1]:.3f})"
            )

    def run(self) -> None:
        print(
            f"[示范] 触摸监控已连接：{self.touchscreen.name} "
            f"{self.touchscreen.maximum_x}x{self.touchscreen.maximum_y}"
        )
        print("[示范] 请停留在已标定的离线人机入口并手动开始对局。")
        screen_thread = threading.Thread(
            target=self._screen_loop, name="demo-screen", daemon=True
        )
        touch_thread = threading.Thread(
            target=self._touch_loop, name="demo-touch", daemon=True
        )
        action_thread = threading.Thread(
            target=self._action_worker, name="demo-actions", daemon=True
        )
        screen_thread.start()
        touch_thread.start()
        action_thread.start()
        try:
            while not self.stop_event.wait(0.25):
                if not touch_thread.is_alive():
                    raise DeviceError("触摸事件监控意外停止")
        finally:
            self.request_stop()
            self.jobs.put(None)
            touch_thread.join(timeout=2.0)
            screen_thread.join(timeout=2.0)
            action_thread.join(timeout=10.0)
            print(
                f"[示范] 录制结束：{self.completed_battles} 局，"
                f"{self.action_count} 次有效出牌，目录={self.run_dir}"
            )
            if bool(
                self.config.get("demonstration", {}).get(
                    "auto_train_after_recording", True
                )
            ):
                from .imitation import audit_demonstrations, train_imitation_policy

                imitation_config = dict(self.config.get("demonstration", {}))
                audit = audit_demonstrations(
                    self.project_root, self.catalog, imitation_config
                )
                if audit.ready:
                    result = train_imitation_policy(
                        self.project_root, self.catalog, imitation_config
                    )
                    candidate = result["candidate"]
                    if candidate.get("promoted"):
                        print(
                            f"[示范] 新模仿策略已通过验证并晋级："
                            f"{candidate['version']}"
                            f"（{candidate.get('maturity', 'full')}）"
                        )
                    else:
                        print(
                            "[示范] 候选模仿策略未通过验证，继续保留原策略："
                            + "；".join(candidate.get("rejection_reasons", []))
                        )
                else:
                    print(
                        "[示范] 模仿训练数据尚未达标："
                        + "；".join(audit.blocking_reasons)
                    )
