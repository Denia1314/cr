from __future__ import annotations

import time
from pathlib import Path
from threading import Event
from typing import Any

from PIL import Image

from .adb import DeviceError, MumuDevice
from .policy import BattlePolicy
from .recorder import TrainingRecorder
from .replay import ExperienceReplayRecorder
from .vision import (
    ProbeMatch,
    WorkflowRecognizer,
    battle_ui_score,
    detect_battle_result,
    find_chest_open_screen,
    find_result_confirm_button,
)


class BotEngine:
    def __init__(
        self,
        device: MumuDevice,
        config: dict[str, Any],
        config_path: Path,
        *,
        dry_run: bool = False,
        max_battles: int = 0,
        stop_event: Event | None = None,
    ):
        self.device = device
        self.config = config
        self.config_path = config_path
        self.project_root = config_path.parent
        self.dry_run = dry_run
        self.max_battles = max(0, int(max_battles))
        self.stop_event = stop_event or Event()
        self.recognizer = WorkflowRecognizer(config, config_path)
        self.policy = BattlePolicy(config, config_path)
        if self.policy.mode == "reactive_catalog":
            if self.policy.reactive_ready:
                print(
                    f"[策略] 通用响应策略已加载，"
                    f"卡牌模板={len(self.policy.hand_recognizer.templates)}"
                )
                if (
                    self.policy.learned_detector is not None
                    and self.policy.learned_detector.available
                ):
                    version = self.policy.learned_detector.champion.get("version", "unknown")
                    print(f"[策略] 已加载验证通过的战场模型：{version}")
                else:
                    print("[策略] 尚无验证通过的战场模型，使用通用压路威胁检测。")
                if (
                    self.policy.imitation_model is not None
                    and self.policy.imitation_model.available
                ):
                    version = self.policy.imitation_model.champion.get(
                        "version", "unknown"
                    )
                    maturity = self.policy.imitation_model.champion.get(
                        "maturity", "full"
                    )
                    print(
                        f"[策略] 已加载你示范操作训练出的模仿策略："
                        f"{version}（{maturity}）"
                    )
                if (
                    self.policy.replay_model is not None
                    and self.policy.replay_model.available
                ):
                    version = (self.policy.replay_model.champion or {}).get(
                        "version", "unknown"
                    )
                    print(f"[策略] 已加载验证晋升的回放自学习策略：{version}")
            else:
                print(f"[策略] 通用响应策略未就绪，将停止盲目出牌：{self.policy.perception_error}")
        self.recorder = TrainingRecorder(
            self.project_root,
            config.get("dataset", {}),
        )
        replay_config = dict(config.get("replay", {}))
        if self.dry_run:
            replay_config["enabled"] = False
        policy_metadata: dict[str, Any] = {
            "mode": self.policy.mode,
            "rule_version": self.policy.policy.get("version", "unversioned"),
        }
        if (
            self.policy.imitation_model is not None
            and self.policy.imitation_model.available
        ):
            champion = self.policy.imitation_model.champion or {}
            policy_metadata.update(
                {
                    "imitation_version": champion.get("version"),
                    "imitation_maturity": champion.get("maturity"),
                }
            )
        if self.policy.replay_model is not None and self.policy.replay_model.available:
            champion = self.policy.replay_model.champion or {}
            policy_metadata.update(
                {
                    "replay_policy_version": champion.get("version"),
                    "replay_policy_status": champion.get("status"),
                }
            )
        self.replay = ExperienceReplayRecorder(
            self.recorder.run_dir,
            replay_config,
            policy_metadata=policy_metadata,
        )
        self.offline_verified = False
        self.offline_gate_streak = 0
        self.offline_gate_last_seen = 0.0
        self.battle_seen = False
        self.in_battle = False
        self.completed_battles = 0
        self.previous_battle_frame: Image.Image | None = None
        self.latest_frame: Image.Image | None = None
        self.last_known_at = time.monotonic()

    def request_stop(self) -> None:
        """Ask the engine to stop at the next safe boundary."""
        self.stop_event.set()

    def _stop_requested(self) -> bool:
        return self.stop_event.is_set()

    def _sleep(self, seconds: float) -> bool:
        """Wait interruptibly and return True when a stop was requested."""
        return self.stop_event.wait(max(0.0, float(seconds)))

    def _tap(self, point: list[float], image: Image.Image) -> list[int] | None:
        if self.dry_run:
            return None
        return list(self.device.tap_normalized(point, image.size))

    def _update_offline_gate(
        self, image: Image.Image, matches: list[ProbeMatch]
    ) -> ProbeMatch | None:
        gate = next((match for match in matches if match.kind == "offline_gate"), None)
        if gate is not None:
            self.offline_gate_streak += 1
            self.offline_gate_last_seen = time.monotonic()
            required_frames = int(
                self.config.get("safety", {}).get("offline_gate_consecutive_frames", 3)
            )
            first_verification = not self.offline_verified and self.offline_gate_streak >= required_frames
            if first_verification:
                self.offline_verified = True
                print(f"[安全门] 已识别离线人机标志，分数 {gate.score:.3f}")
                self.recorder.record(
                    "offline_gate_verified",
                    image,
                    {
                        "probe": gate.name,
                        "score": round(gate.score, 4),
                        "consecutive_frames": self.offline_gate_streak,
                    },
                )
        else:
            self.offline_gate_streak = 0
        return gate

    def _run_single_marker(self, package: str) -> None:
        automation = self.config["automation"]
        poll_interval = float(self.config["timing"].get("poll_interval_s", 0.7))
        unknown_timeout = float(self.config["timing"].get("unknown_timeout_s", 45))
        start_point = list(automation["start_battle_point"])
        start_retry_s = float(automation.get("start_retry_s", 8.0))
        matchmaking_notice_s = max(
            10.0,
            float(automation.get("matchmaking_status_interval_s", 60.0)),
        )
        battle_threshold = float(automation.get("battle_ui_threshold", 0.72))
        end_frames = int(automation.get("battle_end_consecutive_frames", 4))
        confirm_interval = float(
            automation.get("post_battle_confirm_interval_s", 0.75)
        )
        post_timeout = float(automation.get("post_battle_timeout_s", 35.0))
        chest_tap_interval = max(
            0.5, float(automation.get("chest_screen_tap_interval_s", 0.8))
        )
        chest_taps_per_detection = max(
            1, min(8, int(automation.get("chest_screen_taps_per_detection", 5)))
        )
        chest_tap_burst_delay = max(
            0.12, float(automation.get("chest_screen_tap_burst_delay_s", 0.28))
        )
        chest_followup_tap_delay = max(
            0.0,
            float(automation.get("chest_screen_followup_tap_delay_s", 3.0)),
        )

        awaiting_battle = False
        last_start_at = 0.0
        last_matchmaking_notice_at = 0.0
        battle_absent_frames = 0
        post_battle = False
        post_battle_since = 0.0
        last_confirm_at = 0.0
        last_chest_tap_at = 0.0
        result_confirmed_at = 0.0
        result_reward_recorded = False

        print(
            "离线模式：全局检测确定与宝箱，确认离线标志后自动开局。"
            " 宝箱识别=整体布局-v5（选择型/四星直接开启型）"
        )

        while True:
            if self._stop_requested():
                print("[停止] 已收到停止请求，自动训练已安全结束。")
                return
            if (
                self.max_battles
                and self.completed_battles >= self.max_battles
                and not post_battle
            ):
                print(f"已达到 max_battles={self.max_battles}，停止。")
                return

            foreground = self.device.foreground_package()
            if foreground and foreground != package:
                print(f"前台应用已离开游戏（{foreground}），为避免误点，停止。")
                return

            now = time.monotonic()
            image = self.device.screenshot()
            self.latest_frame = image
            matches = self.recognizer.match_all(image)
            gate = self._update_offline_gate(image, matches)
            ui_score = battle_ui_score(image, automation["battle_ui_roi"])
            battle_detected = ui_score >= battle_threshold

            # These two high-specificity controls are checked before the battle
            # color heuristic. Purple chest themes otherwise resemble the
            # purple elixir strip and can be mistaken for a battle frame.
            confirm_point, confirm_score = find_result_confirm_button(image)
            chest_point: list[float] | None = None
            chest_score = 0.0
            if (
                confirm_point is None
                and bool(automation.get("chest_screen_enabled", True))
            ):
                chest_point, chest_score = find_chest_open_screen(image)

            if (
                confirm_point is not None
                and now - last_confirm_at >= confirm_interval
            ):
                if self.in_battle:
                    self.in_battle = False
                    self.battle_seen = False
                    self.previous_battle_frame = None
                    self.completed_battles += 1
                    post_battle = True
                    post_battle_since = now
                    result_reward_recorded = False
                    print(f"[结算] 第 {self.completed_battles} 局结束，发现确定按钮")
                    self.recorder.record(
                        "battle_ended_by_confirm",
                        image,
                        {
                            "battle_ui_score": round(ui_score, 4),
                            "completed_battles": self.completed_battles,
                        },
                    )
                elif not post_battle:
                    # A confirmation can also appear on a reward screen opened
                    # before startup.  Enter the same bounded reward flow.
                    post_battle = True
                    post_battle_since = now

                result = detect_battle_result(image)
                pixel = self._tap(confirm_point, image)
                event = self.recorder.record(
                    "global_confirm",
                    image,
                    {
                        "click_normalized": confirm_point,
                        "click_pixel": pixel,
                        "confidence": round(confirm_score, 4),
                        "dry_run": self.dry_run,
                        "result": result.to_dict(),
                    },
                )
                if (
                    self.completed_battles > 0
                    and not result_reward_recorded
                    and not self.dry_run
                ):
                    episode = self.replay.finish_battle(
                        self.completed_battles,
                        result,
                        event.get("frame"),
                    )
                    result_reward_recorded = episode is not None
                print(
                    f"[确定] {'试运行：' if self.dry_run else ''}"
                    f"全局识别并点击 {confirm_point} score={confirm_score:.3f}"
                )
                self.last_known_at = now
                last_confirm_at = now
                result_confirmed_at = now
                awaiting_battle = False
                self.offline_verified = False
                self.offline_gate_streak = 0
                if self._sleep(poll_interval):
                    print("[停止] 已收到停止请求，自动训练已安全结束。")
                    return
                continue

            if (
                chest_point is not None
                and now - last_chest_tap_at >= chest_tap_interval
            ):
                if self.in_battle:
                    self.in_battle = False
                    self.battle_seen = False
                    self.previous_battle_frame = None
                    self.completed_battles += 1
                    result_reward_recorded = False
                    self.recorder.record(
                        "battle_ended_by_chest",
                        image,
                        {"completed_battles": self.completed_battles},
                    )
                    print(
                        f"[结算] 第 {self.completed_battles} 局结束，"
                        "进入宝箱画面"
                    )
                awaiting_battle = False
                post_battle = True
                post_battle_since = now
                battle_absent_frames = 0
                self.offline_verified = False
                self.offline_gate_streak = 0
                self.last_known_at = now
                last_chest_tap_at = now
                pixels: list[list[int] | None] = []
                burst_tap_total = 1 if self.dry_run else chest_taps_per_detection
                for tap_index in range(burst_tap_total):
                    pixels.append(self._tap(chest_point, image))
                    if (
                        tap_index + 1 < burst_tap_total
                        and self._sleep(chest_tap_burst_delay)
                    ):
                        print("[停止] 已收到停止请求，自动训练已安全结束。")
                        return
                followup_tapped = False
                if not self.dry_run:
                    if self._sleep(chest_followup_tap_delay):
                        print("[停止] 已收到停止请求，自动训练已安全结束。")
                        return
                    pixels.append(self._tap(chest_point, image))
                    followup_tapped = True
                self.recorder.record(
                    "chest_open_screen_tap",
                    image,
                    {
                        "click_normalized": chest_point,
                        "click_pixel": pixels[0],
                        "tap_count": len(pixels),
                        "burst_tap_count": burst_tap_total,
                        "followup_tapped": followup_tapped,
                        "followup_delay_s": chest_followup_tap_delay,
                        "confidence": round(chest_score, 4),
                        "dry_run": self.dry_run,
                    },
                )
                if self.dry_run:
                    action_text = "试运行：已模拟宝箱点击"
                else:
                    action_text = (
                        f"连续点击宝箱 {burst_tap_total} 次，"
                        f"等待 {chest_followup_tap_delay:g} 秒后补点 1 次"
                    )
                print(
                    f"[宝箱] 整体画面识别成功；{action_text} "
                    f"{chest_point} score={chest_score:.3f}"
                )
                if self._sleep(poll_interval):
                    print("[停止] 已收到停止请求，自动训练已安全结束。")
                    return
                continue

            if battle_detected and (self.in_battle or awaiting_battle or self.offline_verified):
                self.last_known_at = now
                battle_absent_frames = 0
                awaiting_battle = False
                post_battle = False
                if not self.in_battle:
                    self.in_battle = True
                    self.battle_seen = True
                    self.previous_battle_frame = None
                    self.policy.reset_battle(now)
                    # Each new match must return to the marked offline screen
                    # before the following match can be started.
                    self.offline_verified = False
                    self.offline_gate_streak = 0
                    print(f"[战斗] 已自动识别战斗界面，分数 {ui_score:.3f}")
                    self.recorder.record(
                        "battle_started_auto",
                        image,
                        {"battle_ui_score": round(ui_score, 4)},
                    )
                    self.replay.start_battle(self.completed_battles + 1)
                self._play_battle(image)

            elif self.in_battle:
                battle_absent_frames += 1
                if battle_absent_frames >= end_frames:
                    self.in_battle = False
                    self.battle_seen = False
                    self.previous_battle_frame = None
                    self.completed_battles += 1
                    post_battle = True
                    post_battle_since = now
                    last_confirm_at = 0.0
                    result_confirmed_at = 0.0
                    result_reward_recorded = False
                    print(f"[结算] 自动判断第 {self.completed_battles} 局结束")
                    self.recorder.record(
                        "battle_ended_auto",
                        image,
                        {
                            "battle_ui_score": round(ui_score, 4),
                            "completed_battles": self.completed_battles,
                        },
                    )

            elif post_battle:
                self.last_known_at = now
                elapsed = now - post_battle_since
                if elapsed > post_timeout:
                    self.recorder.record(
                        "post_battle_timeout",
                        image,
                        {"seconds": round(elapsed, 2)},
                    )
                    raise DeviceError("结算页面处理超时，已停止")
                if confirm_point is not None:
                    # The literal 确定 text is visible; wait only for debounce.
                    pass
                elif (
                    result_confirmed_at > 0.0
                    and now - result_confirmed_at >= 1.5
                    and gate is not None
                    and self.offline_verified
                ):
                    post_battle = False
                    result_confirmed_at = 0.0
                    print("[结算] 已点击确定并返回大厅")

            elif gate is not None and self.offline_verified and not awaiting_battle:
                self.last_known_at = now
                if now - last_start_at >= start_retry_s:
                    pixel = self._tap(start_point, image)
                    print(
                        f"[开局] {'试运行：' if self.dry_run else ''}已确认离线模式，"
                        f"点击对战 {start_point}"
                    )
                    self.recorder.record(
                        "auto_start_battle",
                        image,
                        {
                            "click_normalized": start_point,
                            "click_pixel": pixel,
                            "dry_run": self.dry_run,
                            "gate_score": round(gate.score, 4),
                        },
                    )
                    last_start_at = now
                    last_matchmaking_notice_at = now
                    if not self.dry_run:
                        awaiting_battle = True

            elif awaiting_battle:
                self.last_known_at = now
                elapsed = now - last_start_at
                if now - last_matchmaking_notice_at >= matchmaking_notice_s:
                    self.recorder.record(
                        "matchmaking_waiting",
                        None,
                        {"seconds": round(elapsed, 2)},
                    )
                    print(
                        f"[匹配] 已等待 {elapsed:.0f} 秒，继续等待进入战斗；"
                        "不会按超时错误停止"
                    )
                    last_matchmaking_notice_at = now

            elif now - self.last_known_at > unknown_timeout:
                self.recorder.record(
                    "unknown_timeout",
                    image,
                    {"seconds": round(now - self.last_known_at, 2)},
                )
                raise DeviceError(
                    f"未知页面持续超过 {unknown_timeout:.0f} 秒，已安全停止；请重新标定离线标志"
                )

            if self._sleep(poll_interval):
                print("[停止] 已收到停止请求，自动训练已安全结束。")
                return

    def _play_battle(self, image: Image.Image) -> None:
        self.recorder.record_battle_sample(
            image,
            battle_index=self.completed_battles + 1,
        )
        now = time.monotonic()
        observation = self.policy.observe_replay_state(image, self.previous_battle_frame, now=now)
        if not self.dry_run:
            self.replay.observe_action_effect(self.completed_battles + 1, observation)
        decision = self.policy.decide(image, self.previous_battle_frame, now=now)
        if decision is not None:
            card_pixel = None
            deploy_pixel = None
            if not self.dry_run:
                card_pixel = list(self.device.tap_normalized(decision.card_point, image.size))
                if self._sleep(0.09):
                    return
                deploy_pixel = list(self.device.tap_normalized(decision.deploy_point, image.size))
            print(
                f"[下牌] {'试运行：' if self.dry_run else ''}"
                f"slot={decision.slot_index + 1} card={decision.card_id or 'unknown'} "
                f"lane={decision.lane} "
                f"reason={decision.reason} elixir={decision.elixir:.1f}"
                + (
                    f" role={decision.card_formation_role}"
                    if decision.card_formation_role
                    else ""
                )
                + (
                    f" enemy={','.join(decision.enemy_cards)}"
                    if decision.enemy_cards
                    else ""
                )
                + (" imitation=on" if decision.imitation_used else "")
                + (" replay-learning=on" if decision.replay_learning_used else "")
            )
            payload = decision.to_dict()
            payload["allies_observed"] = observation["allies_observed"]
            payload["observed_allies"] = observation["observed_allies"]
            payload.update(
                {
                    "card_pixel": card_pixel,
                    "deploy_pixel": deploy_pixel,
                    "dry_run": self.dry_run,
                }
            )
            event = self.recorder.record("battle_action", image, payload)
            if not self.dry_run:
                self.replay.record_action(
                    self.completed_battles + 1,
                    image,
                    payload,
                    event.get("frame"),
                )
        self.previous_battle_frame = image.copy()

    def run(self) -> None:
        if self._stop_requested():
            print("[停止] 启动已取消，未打开游戏。")
            return
        calibrated = self.recognizer.calibrated_names()
        required = {"offline_ai_marker"}
        if not required.issubset(calibrated):
            missing = ", ".join(sorted(required - set(calibrated)))
            raise DeviceError(f"标定不完整，缺少：{missing}。请先运行 calibrate.bat")

        package = self.config["game"]["package"]
        if self._stop_requested():
            print("[停止] 启动已取消，未打开游戏。")
            return
        self.device.launch_package(package)
        if self._sleep(3):
            print("[停止] 已在游戏启动后安全停止。")
            return
        print(f"训练数据目录：{self.recorder.run_dir}")
        print("离线安全门已启用；可随时从控制台安全停止。")
        self._run_single_marker(package)
