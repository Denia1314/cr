from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import Any

from PIL import Image

from .adb import DeviceError, MumuDevice
from .action_confirmation import (
    ActionConfirmation,
    ActionConfirmationTracker,
    assess_action_evidence,
)
from .controlled_experiment import (
    ControlledExperiment,
    load_controlled_experiment_episodes,
)
from .decision_router import create_policy
from . import implemented_stages_label, release_label
from .runtime_model import runtime_model_label, apply_decision_engine
from .recorder import TrainingRecorder
from .replay import ExperienceReplayRecorder
from .temporal import TimingStats
from .training_sync import ReplaySync
from .vision import (
    ProbeMatch,
    WorkflowRecognizer,
    battle_ui_score,
    detect_battle_result,
    find_chest_open_screen,
    find_result_confirm_button,
    estimate_elixir,
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
        config = apply_decision_engine(config, config.get("policy", {}).get("decision_engine", "legacy"))
        self.device = device
        self.config = config
        self.config_path = config_path
        self.project_root = config_path.parent
        self.dry_run = dry_run
        self.max_battles = max(0, int(max_battles))
        self.stop_event = stop_event or Event()
        self.self_learning = None
        if not dry_run and config.get("self_learning", {}).get("enabled", False):
            from .self_learning import SelfLearningService
            self.self_learning = SelfLearningService(self.project_root, config, self._stop_requested)
        self.recognizer = WorkflowRecognizer(config, config_path)
        self.policy = create_policy(config, config_path)
        print(f"[决策引擎] {config.get('policy', {}).get('decision_engine', 'legacy')}")
        print(f"[版本] {release_label()} · {implemented_stages_label()}")
        print(f"[运行版本] 当前运行：{runtime_model_label(self.policy.runtime_model)}")
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
            "decision_engine": config.get("policy", {}).get("decision_engine", "legacy"),
            "mode": self.policy.mode,
            "runtime_model": self.policy.runtime_model,
            "rule_version": self.policy.policy.get("version", "unversioned"),
            "temporal_observation_schema": "hand_elixir_formation_v1",
            "tactical_observation_schema": "threat_layers_v1",
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
        replay_version = None
        if self.policy.replay_model is not None:
            replay_version = (self.policy.replay_model.champion or {}).get("version")
        prior_experiment_episodes = load_controlled_experiment_episodes(
            self.project_root, replay_version
        )
        self.experiment = ControlledExperiment(
            dict(config.get("controlled_experiment", {})),
            self.policy.replay_model,
            prior_episodes=prior_experiment_episodes,
        )
        if self.experiment.enabled:
            print(
                f"[实验] 已恢复当前模型的有效批次进度："
                f"{self.experiment.battle_offset} 局"
            )
        automation_config = dict(config.get("automation", {}))
        self.action_confirmation = ActionConfirmationTracker(
            timeout_s=float(
                automation_config.get("action_confirmation_timeout_s", 1.35)
            ),
            poll_interval_s=float(
                automation_config.get("action_confirmation_poll_interval_s", 0.18)
            ),
            stable_frames=int(
                automation_config.get("action_confirmation_stable_frames", 2)
            ),
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
        self._last_screenshot_elapsed_s = 0.0
        self.response_timing = TimingStats(
            max_samples=int(automation_config.get("timing_max_samples", 512))
        )
        self._write_runtime_status()

    def _write_runtime_status(self) -> None:
        replay_model = self.policy.replay_model
        hand = getattr(self.policy, "hand_recognizer", None)
        hand_status = hand.compute_status() if hand is not None and hasattr(hand, "compute_status") else None
        ReplaySync(self.project_root).write_runtime_status({
            "prediction": self.policy.prediction_status() if hasattr(self.policy, "prediction_status") else {"selected_engine": "legacy", "actual_engine": "legacy"},
            "rule_version": self.policy.policy.get("version", "unversioned"),
            "hand_matching": hand_status,
            "capture": {**self.frame_stream.status(),"backend":getattr(self.device,"capture_backend","unknown")} if getattr(self,"frame_stream",None) else {"backend":"synchronous"},
            "replay_model": {
                "version": ((replay_model.champion or {}).get("version") if replay_model else None),
                "loaded": bool(replay_model is not None and replay_model.available),
                "load_error": (getattr(replay_model, "load_error", None) if replay_model else "disabled"),
            },
        })

    def request_stop(self) -> None:
        """Ask the engine to stop at the next safe boundary."""
        self.stop_event.set()

    def _stop_requested(self) -> bool:
        return self.stop_event.is_set()

    def response_timing_summary(self) -> dict[str, dict[str, float | int | None]]:
        """Return M2 stage latency percentiles collected in this run."""
        return self.response_timing.summary()

    def _record_response_timing(self, image: Image.Image | None, reason: str) -> None:
        summary = self.response_timing_summary()
        if not summary:
            return
        self.recorder.record(
            "battle_timing_summary",
            image,
            {"reason": str(reason), "summary": summary},
        )
        self._write_runtime_status()

    def _sleep(self, seconds: float) -> bool:
        """Wait interruptibly and return True when a stop was requested."""
        return self.stop_event.wait(max(0.0, float(seconds)))

    def _capture_frame(self, *, after=0., timeout=8.):
        stream=getattr(self,'frame_stream',None)
        if stream is None:
            started=time.monotonic()
            image=self.device.screenshot()
            finished=time.monotonic()
        else:
            frame=stream.get(sequence=getattr(self,'_frame_sequence',0),after=after,timeout=timeout)
            self._frame_sequence=frame.sequence
            image,started,finished=frame.image,frame.started,frame.finished
        self._last_battle_capture=(image,started)
        self._last_screenshot_elapsed_s=finished-started
        return image

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
        from .battle_transition import BattleTransitionGuard
        transition = BattleTransitionGuard.from_config(automation)
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
            screenshot_started = time.perf_counter()
            image = self._capture_frame()
            self.response_timing.record("screenshot", self._last_screenshot_elapsed_s)
            self.latest_frame = image
            matches = self.recognizer.match_all(image)
            gate = self._update_offline_gate(image, matches)
            ui_score = battle_ui_score(image, automation["battle_ui_roi"])
            battle_detected = ui_score >= battle_threshold
            transition.observe(battle_detected, now)

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
                    transition.reset()
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
                    if episode is not None:
                        if self.self_learning is not None:
                            self.self_learning.on_battle_completed(episode)
                        stop_reason = self.experiment.observe(episode)
                        if stop_reason:
                            print(f"[实验护栏] {stop_reason}，将在局间停止。")
                            rollback = self.experiment.rollback_to_baseline()
                            self.replay.policy_metadata.update(rollback)
                            ReplaySync(self.project_root).write_runtime_status({
                                "rule_version": self.policy.policy.get("version", "unversioned"),
                                "experiment": rollback,
                            })
                            self.request_stop()
                print(
                    f"[确定] {'试运行：' if self.dry_run else ''}"
                    f"全局识别并点击 {confirm_point} score={confirm_score:.3f}"
                )
                self._record_response_timing(image, "confirm")
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
                    transition.reset()
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
                self._record_response_timing(image, "chest")
                if self._sleep(poll_interval):
                    print("[停止] 已收到停止请求，自动训练已安全结束。")
                    return
                continue

            if battle_detected and (self.in_battle or awaiting_battle or self.offline_verified):
                self.last_known_at = now
                if (not self.in_battle or post_battle) and not transition.entry_ready:
                    if self._sleep(poll_interval):
                        return
                    continue
                if self.in_battle and post_battle:
                    self.recorder.record("battle_resumed_after_ui_gap", image,
                                         {"battle_ui_score": round(ui_score, 4),
                                          "completed_battles": self.completed_battles})
                    print("[战斗] 战斗界面恢复，撤销待确认结算并继续当前对局")
                awaiting_battle = False
                post_battle = False
                if not self.in_battle:
                    self.in_battle = True
                    transition.start(now)
                    self.battle_seen = True
                    self.previous_battle_frame = None
                    self.policy.reset_battle(now)
                    self.replay.policy_metadata.update(
                        self.experiment.activate(self.completed_battles + 1)
                    )
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
                # Missing elixir UI can be loading, animation or occlusion. It is
                # not positive evidence of a completed match, and must not clear
                # the current policy/replay or revoke its right to resume.
                self.last_known_at = now
                if transition.end_pending and not post_battle:
                    post_battle = True
                    post_battle_since = now
                    self.recorder.record("battle_end_pending", image,
                                         {"battle_ui_score": round(ui_score, 4),
                                          "absent_frames": transition.absent_frames,
                                          "completed_battles": self.completed_battles})
                    print("[战斗] 界面暂时消失，等待结算证据或战斗画面恢复")
                if post_battle and now - post_battle_since > post_timeout:
                    self.recorder.record("post_battle_timeout", image,
                                         {"seconds": round(now-post_battle_since, 2),
                                          "end_confirmed": False})
                    raise DeviceError("战斗界面持续缺失且未确认结算，已停止；本局未计为完成")

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
                if self.self_learning is not None:
                    try:
                        if self.self_learning.boundary():
                            # Training can take minutes. Never click using the pre-training frame.
                            self.last_known_at = time.monotonic()
                            self.offline_verified = False
                            self.offline_gate_streak = 0
                            continue
                    except (OSError, ValueError) as exc:
                        print(f"[自主学习] 暂停本次任务，现役保持不变：{exc}")
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
        captured = getattr(self, "_last_battle_capture", None)
        self.policy.prediction_frame_age_s = max(0, now - captured[1]) if captured and captured[0] is image else 0
        cycle_started = time.perf_counter()
        timing: dict[str, float] = {}
        timing["screenshot_s"] = self._last_screenshot_elapsed_s
        perception_started = time.perf_counter()
        observation = self.policy.observe_replay_state(image, self.previous_battle_frame, now=now)
        timing["perception_s"] = time.perf_counter() - perception_started
        if not self.dry_run:
            self.replay.observe_action_effect(self.completed_battles + 1, observation)
        policy_snapshot = self.policy.snapshot_state()
        decision_started = time.perf_counter()
        decision = self.policy.decide(image, self.previous_battle_frame, now=now)
        timing["decision_s"] = time.perf_counter() - decision_started
        if hasattr(self.policy, "prediction_status"):
            prediction = self.policy.prediction_status()
            plan = prediction.get("plan") or {}
            key = (self.completed_battles, prediction["world"]["revision"], prediction["fallback_count"])
            if key != getattr(self, "_last_prediction_log_key", None):
                self._last_prediction_log_key = key
                self.recorder.record("battle_prediction", None, prediction)
                print(f"[推演] 执行={prediction['actual_engine']} "
                      f"感知={plan.get('perception_mode', 'unavailable')} "
                      f"学习={plan.get('learning', {}).get('reason', 'unavailable')} "
                      f"改选={plan.get('learning', {}).get('changed_selection', False)} "
                      f"{plan.get('fallback_reason') or plan.get('reason') or '等待观察'}")
        self._write_runtime_status()
        self.response_timing.record("perception", timing["perception_s"])
        self.response_timing.record("decision", timing["decision_s"])
        if decision is not None:
            action_hand_metadata = self.policy.hand_metadata(now)
            action_formation_metadata = self.policy.formation_metadata(now)
            decision, card_pixel, deploy_pixel, confirmation, post_image, send_error = (
                self._execute_action(image, decision, policy_snapshot, timing=timing)
            )
            timing["total_s"] = time.perf_counter() - cycle_started
            self.response_timing.record("total", timing["total_s"])
            if decision.threat_type != "none":
                timing["threat_to_confirmation_s"] = timing["total_s"]
                self.response_timing.record(
                    "threat_to_confirmation", timing["total_s"]
                )
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
                + f" status={decision.action_status}"
                + (
                    f" confirm={confirmation.confidence:.2f}"
                    if confirmation is not None
                    else ""
                )
                + (f" error={send_error}" if send_error else "")
            )
            payload = decision.to_dict()
            payload["allies_observed"] = observation["allies_observed"]
            payload["observed_allies"] = observation["observed_allies"]
            payload["hand_metadata"] = action_hand_metadata
            payload["formation_metadata"] = action_formation_metadata
            payload.update(
                {
                    "card_pixel": card_pixel,
                    "deploy_pixel": deploy_pixel,
                    "dry_run": self.dry_run,
                    "action_confirmation": (
                        confirmation.to_dict() if confirmation is not None else None
                    ),
                    "action_send_error": send_error,
                    "timing_s": dict(timing),
                    "response_timing_summary": self.response_timing_summary(),
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
            if post_image is not None:
                self.previous_battle_frame = post_image.copy()
        else:
            self.previous_battle_frame = image.copy()

    def _revalidate_prediction(self, image, decision):
        stream=getattr(self,'frame_stream',None)
        if stream is None or getattr(decision,'decision_engine','legacy') != 'predictive':
            return image,None
        frame=stream.latest()
        if frame is None or frame.sequence <= getattr(self,'_frame_sequence',0):
            return image,None
        if time.monotonic()-frame.started > float(self.config.get('prediction',{}).get('max_plan_age_s',2.5)):
            return image,'latest_frame_stale'
        current=frame.image
        matches=self.policy.hand_recognizer.recognize(current)
        match=next((m for m in matches if m.slot_index==decision.slot_index),None)
        if match is None or match.card_id != decision.card_id or match.confidence < float(getattr(self.policy,'policy',{}).get('hand_min_confidence',.4)):
            return current,'hand_changed_before_send'
        elixir,confidence=estimate_elixir(current,self.config['vision']['elixir_roi'])
        if confidence >= .08 and elixir is not None and elixir < (decision.card_cost or 0):
            return current,'elixir_changed_before_send'
        threats=self.policy._perceive_threats(current,image,time.monotonic())
        old=decision.left_threat if decision.lane=='left' else decision.right_threat
        threat=threats[decision.lane]
        if old >= .17 and threat.score < .05 and threat.unit_count == 0:
            return current,'threat_disappeared_before_send'
        return current,None

    def _execute_action(
        self,
        image: Image.Image,
        decision,
        policy_snapshot: dict[str, Any],
        *,
        timing: dict[str, float] | None = None,
    ) -> tuple[Any, list[int] | None, list[int] | None, ActionConfirmation, Image.Image | None, str | None]:
        """Send one action and resolve it from bounded post-click evidence."""
        decision = self.policy.prepare_action(decision, policy_snapshot)
        action_id = decision.action_id
        recheck_started=time.perf_counter()
        image,rejected=self._revalidate_prediction(image,decision)
        if hasattr(self,'response_timing'):
            self.response_timing.record('pre_send_recheck',time.perf_counter()-recheck_started)
        if rejected:
            confirmation=ActionConfirmation(action_id,'rejected',1.,rejected,{'pre_send_recheck':rejected},0.,0)
            self.policy.resolve_action(action_id,'rejected',now=time.monotonic())
            return replace(decision,action_status='rejected'),None,None,confirmation,image,rejected
        self.recorder.record(
            "battle_action_proposed",
            image,
            decision.to_dict(),
        )

        # Capture pre-action observations once.  If either recognizer is
        # unavailable, the confirmation state machine simply demands stronger
        # remaining evidence instead of assuming the tap worked.
        pre_matches = None
        if self.policy.last_hand_image is image and self.policy.last_hand_matches:
            pre_matches = self.policy.last_hand_matches
        elif self.policy.hand_recognizer is not None:
            try:
                pre_matches = self.policy.hand_recognizer.recognize(image)
            except Exception:
                pre_matches = None
        if (
            self.policy.last_elixir_image is image
            and self.policy.last_elixir_estimate_value is not None
        ):
            pre_elixir = self.policy.last_elixir_estimate_value
            pre_elixir_confidence = (
                self.policy.last_elixir_confidence
                if self.policy.last_elixir_estimate_source.startswith("vision")
                else 0.0
            )
        else:
            pre_elixir, pre_elixir_confidence = estimate_elixir(
                image, self.config["vision"]["elixir_roi"]
            )
        if pre_elixir is None or pre_elixir_confidence < 0.08:
            pre_elixir = float(decision.elixir)

        card_pixel: list[int] | None = None
        deploy_pixel: list[int] | None = None
        send_error: str | None = None
        if getattr(decision, "decision_engine", "legacy") == "predictive" and time.monotonic() > decision.plan_valid_until:
            confirmation = ActionConfirmation(action_id, "rejected", 1.0,
                "推演快照已过期，未发送点击", {"plan_expired": True}, 0.0, 0)
            self.policy.resolve_action(action_id, "rejected", now=time.monotonic())
            return replace(decision, action_status="rejected"), None, None, confirmation, None, "plan_expired"
        if self.dry_run:
            dry_run_elapsed = 0.0
            if timing is not None:
                timing["send_s"] = dry_run_elapsed
                timing["adb_s"] = dry_run_elapsed
                timing["confirmation_s"] = 0.0
            self.response_timing.record("send", dry_run_elapsed)
            self.response_timing.record("adb", dry_run_elapsed)
            self.response_timing.record("confirmation", 0.0)
            confirmation = ActionConfirmation(
                action_id,
                "unknown",
                0.0,
                "试运行未发送真实点击，未将动作视为确认成功",
                {"dry_run": True},
                0.0,
                0,
            )
            self.policy.resolve_action(action_id, "unknown", now=time.monotonic())
            return (
                replace(decision, action_status="unknown"),
                card_pixel,
                deploy_pixel,
                confirmation,
                None,
                None,
            )

        send_started = time.perf_counter()
        captured=getattr(self,'_last_battle_capture',None)
        if captured:
            self.response_timing.record('capture_to_send',max(0,time.monotonic()-captured[1]))
        try:
            card_pixel = list(self.device.tap_normalized(decision.card_point, image.size))
            if self._sleep(0.09):
                raise DeviceError("停止请求发生在选牌点击后")
            deploy_pixel = list(
                self.device.tap_normalized(decision.deploy_point, image.size)
            )
        except Exception as exc:  # ADB timeout can still be a real game action.
            send_error = str(exc)
        send_elapsed = time.perf_counter() - send_started
        if timing is not None:
            timing["send_s"] = send_elapsed
            timing["adb_s"] = send_elapsed
        self.response_timing.record("send", send_elapsed)
        self.response_timing.record("adb", send_elapsed)

        # The evidence budget starts only after the deployment tap attempt has
        # completed.  Pre-action recognition and ADB latency must not consume
        # the short confirmation window.  Keep event-image encoding outside
        # the critical loop as well so slow disks cannot steal observation
        # frames.
        confirmation_window_started_at = time.monotonic()
        self.action_confirmation.register(
            action_id,
            sent_at=confirmation_window_started_at,
        )
        timeout_s = self.action_confirmation.timeout_s
        deadline = confirmation_window_started_at + timeout_s
        post_image: Image.Image | None = None
        confirmation: ActionConfirmation | None = None
        last_evidence: dict[str, Any] = {}
        confirmation_error = False
        confirmation_started = time.perf_counter()
        observation_timings: list[dict[str, Any]] = []
        while time.monotonic() < deadline and not self._stop_requested():
            capture_started = time.perf_counter()
            try:
                post_image = self._capture_frame(after=confirmation_window_started_at,timeout=max(.01,deadline-time.monotonic()))
            except Exception as exc:
                send_error = send_error or f"确认截图失败：{exc}"
                confirmation_error = True
                last_evidence = {**last_evidence, "capture_error": True}
                break
            capture_elapsed = time.perf_counter() - capture_started
            recognition_started = time.perf_counter()
            post_matches = None
            if self.policy.hand_recognizer is not None:
                try:
                    post_matches = self.policy.hand_recognizer.recognize(post_image)
                except Exception:
                    post_matches = None
            post_elixir, post_elixir_confidence = estimate_elixir(
                post_image, self.config["vision"]["elixir_roi"]
            )
            if post_elixir_confidence < 0.08:
                post_elixir = None
            recognition_elapsed = time.perf_counter() - recognition_started
            self.response_timing.record("confirmation_capture", capture_elapsed)
            self.response_timing.record("confirmation_recognition", recognition_elapsed)
            center = None
            centers = self.config["vision"].get("card_slot_centers", [])
            if 0 <= int(decision.slot_index) < len(centers):
                center = list(centers[int(decision.slot_index)])
            last_evidence = assess_action_evidence(
                pre_image=image,
                post_image=post_image,
                pre_matches=pre_matches,
                post_matches=post_matches,
                slot_index=decision.slot_index,
                pre_elixir=pre_elixir,
                post_elixir=post_elixir,
                card_cost=decision.card_cost,
                slot_center=center,
                vision_config=self.config["vision"],
            )
            if send_error:
                last_evidence["click_error"] = True
            observation_timings.append({
                "index": len(observation_timings) + 1,
                "observed_after_send_s": round(max(0.0, time.monotonic() - confirmation_window_started_at), 6),
                "capture_s": round(capture_elapsed, 6),
                "recognition_s": round(recognition_elapsed, 6),
            })
            confirmation = self.action_confirmation.observe(
                action_id, last_evidence, now=time.monotonic()
            )
            if confirmation.status in {"confirmed", "rejected", "unknown"}:
                break
            remaining = max(0.0, deadline - time.monotonic())
            if self._sleep(min(self.action_confirmation.poll_interval_s, remaining)):
                break

        if confirmation is None or confirmation.status == "sent":
            if self._stop_requested() or confirmation_error:
                confirmation = self.action_confirmation.force_unknown(
                    action_id,
                    reason=(
                        "停止请求中断确认，保留为未知"
                        if self._stop_requested()
                        else "确认截图失败，保留为未知"
                    ),
                    evidence={**last_evidence, "click_error": bool(send_error)},
                    now=time.monotonic(),
                )
            else:
                confirmation = self.action_confirmation.timeout(
                    action_id, now=max(time.monotonic(), deadline)
                )
        self.policy.resolve_action(
            action_id,
            confirmation.status,
            now=time.monotonic(),
        )
        confirmation_elapsed = time.perf_counter() - confirmation_started
        if timing is not None:
            timing["confirmation_s"] = confirmation_elapsed
        self.response_timing.record("confirmation", confirmation_elapsed)
        self.recorder.record(
            "battle_action_sent",
            post_image,
            {
                "action_id": action_id,
                "card_pixel": card_pixel,
                "deploy_pixel": deploy_pixel,
                "send_error": send_error,
                "confirmation_timeout_s": timeout_s,
                "confirmation_elapsed_s": round(confirmation_elapsed, 6),
                "confirmation_status": confirmation.status,
                "confirmation_failure_code": confirmation.failure_code,
                "confirmation_observations": observation_timings,
                "confirmation_summary": self.action_confirmation.summary(),
                "confirmation_frame_role": (
                    "final_observation" if post_image is not None else "unavailable"
                ),
                "confirmation_window_starts_after_send": True,
            },
        )
        return (
            replace(decision, action_status=confirmation.status),
            card_pixel,
            deploy_pixel,
            confirmation,
            post_image,
            send_error,
        )

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
        self.frame_stream=None
        if self.config.get('automation',{}).get('latest_frame_capture',False):
            from .frame_stream import LatestFrameStream
            self.frame_stream=LatestFrameStream(getattr(self.device,'screenshot_fast',self.device.screenshot)).start()
        try:
            self._run_single_marker(package)
        finally:
            if self.frame_stream is not None:
                self.frame_stream.close()
