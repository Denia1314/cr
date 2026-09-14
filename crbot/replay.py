from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from .imitation import battlefield_features
from .vision import BattleResult, detect_battle_result


REPLAY_SCHEMA_VERSION = 2
ACTION_CONFIRMATION_STATUSES = frozenset(
    {"proposed", "sent", "confirmed", "rejected", "unknown", "legacy_unknown"}
)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def state_from_action(image: Image.Image, payload: dict[str, Any]) -> dict[str, Any]:
    visual = battlefield_features(image)
    return {
        "schema": "screen_state_v1",
        "temporal_schema": "hand_elixir_formation_v1",
        "elixir": float(payload.get("elixir", 0.0)),
        "elixir_source": str(payload.get("elixir_source", "unknown")),
        "elixir_confidence": float(payload.get("elixir_confidence", 0.0)),
        "elixir_age_s": float(payload.get("elixir_age_s", -1.0)),
        "elixir_phase": str(payload.get("elixir_phase", "unknown")),
        "hand": [value if value else None for value in payload.get("hand", [])],
        "hand_confidence": float(payload.get("hand_confidence", 0.0)),
        "hand_age_s": float(payload.get("hand_age_s", -1.0)),
        "hand_metadata": dict(payload.get("hand_metadata", {}))
        if isinstance(payload.get("hand_metadata", {}), dict)
        else {},
        "formation_metadata": dict(payload.get("formation_metadata", {}))
        if isinstance(payload.get("formation_metadata", {}), dict)
        else {},
        "timing_s": dict(payload.get("timing_s", {}))
        if isinstance(payload.get("timing_s", {}), dict)
        else {},
        "left_threat": float(payload.get("left_threat", 0.0)),
        "right_threat": float(payload.get("right_threat", 0.0)),
        "left_threat_type": str(payload.get("left_threat_type", "none")),
        "right_threat_type": str(payload.get("right_threat_type", "none")),
        "left_threat_proximity": float(
            payload.get("left_threat_proximity", 0.0)
        ),
        "right_threat_proximity": float(
            payload.get("right_threat_proximity", 0.0)
        ),
        "left_threat_approach_rate": float(
            payload.get("left_threat_approach_rate", 0.0)
        ),
        "right_threat_approach_rate": float(
            payload.get("right_threat_approach_rate", 0.0)
        ),
        "left_unit_count": int(payload.get("left_unit_count", 0)),
        "right_unit_count": int(payload.get("right_unit_count", 0)),
        "battle_elapsed_s": float(payload.get("battle_elapsed_s", 0.0)),
        "threat_type": str(payload.get("threat_type", "none")),
        "threat_proximity": float(payload.get("threat_proximity", 0.0)),
        "threat_approach_rate": float(
            payload.get("threat_approach_rate", 0.0)
        ),
        "enemy_cards": [str(value) for value in payload.get("enemy_cards", [])],
        "left_threat_unit_layers": [
            str(value) for value in payload.get("left_threat_unit_layers", [])
        ],
        "right_threat_unit_layers": [
            str(value) for value in payload.get("right_threat_unit_layers", [])
        ],
        "threat_unit_layers": [
            str(value) for value in payload.get("threat_unit_layers", [])
        ],
        "left_threat_layer_confidence": float(
            payload.get("left_threat_layer_confidence", 0.0)
        ),
        "right_threat_layer_confidence": float(
            payload.get("right_threat_layer_confidence", 0.0)
        ),
        "threat_layer_confidence": float(
            payload.get("threat_layer_confidence", 0.0)
        ),
        "battlefield_edges": [round(float(value), 6) for value in visual],
        "allies_observed": bool(payload.get("allies_observed", False)),
        "observed_allies": list(payload.get("observed_allies", [])),
        "action_confirmation_status": str(
            payload.get("action_status") or "legacy_unknown"
        ),
    }


def action_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    deploy = list(payload.get("deploy_point", []))
    status = str(payload.get("action_status") or "legacy_unknown").strip()
    if status not in ACTION_CONFIRMATION_STATUSES:
        status = "unknown"
    confirmation = payload.get("action_confirmation", {})
    if not isinstance(confirmation, dict):
        confirmation = {}
    return {
        "action_id": str(payload.get("action_id", "")).strip(),
        "action_status": status,
        "action_confirmation": dict(confirmation),
        "slot_index": int(payload.get("slot_index", -1)),
        "card_id": payload.get("card_id"),
        "deploy_point": [float(value) for value in deploy[:2]],
        "lane": str(payload.get("lane", "unknown")),
        "reason": str(payload.get("reason", "unknown")),
        "formation_phase": str(payload.get("formation_phase", "")),
        "card_formation_role": str(payload.get("card_formation_role", "")),
        "desired_formation_role": str(payload.get("desired_formation_role", "")),
        "learned_action_value": dict(payload.get("learned_action_value", {}))
        if isinstance(payload.get("learned_action_value"), dict) else {},
        "card_attack_targets": [
            str(value) for value in payload.get("card_attack_targets", [])
        ],
        "card_targeting_source": str(payload.get("card_targeting_source", "unknown")),
    }


class ExperienceReplayRecorder:
    """Buffer one offline episode and write verified transitions at its end."""

    def __init__(
        self,
        run_dir: Path,
        config: dict[str, Any] | None = None,
        *,
        policy_metadata: dict[str, Any] | None = None,
        source: str = "offline_ai_bot",
    ):
        self.run_dir = run_dir.resolve()
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", True))
        self.allow_bot_training = bool(
            self.config.get("allow_bot_training", False)
        )
        self.gamma = float(self.config.get("gamma", 0.99))
        self.minimum_result_confidence = float(
            self.config.get("minimum_result_confidence", 0.75)
        )
        self.win_reward = float(self.config.get("terminal_win_reward", 1.0))
        self.loss_reward = float(self.config.get("terminal_loss_reward", -1.0))
        self.draw_reward = float(self.config.get("terminal_draw_reward", 0.0))
        self.crown_reward = float(self.config.get("crown_difference_reward", 0.1))
        self.policy_metadata = dict(policy_metadata or {})
        self.source = source
        self.transitions_path = self.run_dir / "replay_transitions.jsonl"
        self.episodes_path = self.run_dir / "replay_episodes.jsonl"
        self.manifest_path = self.run_dir / "replay_manifest.json"
        self.transition_sequence = _line_count(self.transitions_path)
        self.episode_sequence = _line_count(self.episodes_path)
        self.current_battle: int | None = None
        self.pending: list[dict[str, Any]] = []
        if self.enabled and not self.manifest_path.is_file():
            self._write_manifest()

    def _write_manifest(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        value = {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "created_at_unix": time.time(),
            "source": self.source,
            "policy_actions_are_ground_truth": False,
            "automatic_result_reward": True,
            "action_observation_schema": "short_horizon_visual_proxy_v1",
            "action_observations_are_causal_ground_truth": False,
            "action_confirmation_schema": "stable_hand_elixir_visual_v1",
            "temporal_observation_schema": "hand_elixir_formation_v1",
            "training_requires_confirmed_action": True,
            "allow_bot_training": self.allow_bot_training,
            "training_policy": (
                "collection_only_until_explicitly_enabled"
                if not self.allow_bot_training
                else "verified_results_only"
            ),
            "policy": self.policy_metadata,
        }
        self.manifest_path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def start_battle(self, battle_index: int) -> None:
        if not self.enabled:
            return
        self.current_battle = max(1, int(battle_index))
        self.pending = []

    def discard_current(self) -> None:
        self.current_battle = None
        self.pending = []

    def record_action(
        self,
        battle_index: int,
        image: Image.Image,
        payload: dict[str, Any],
        frame: str | None,
    ) -> None:
        if not self.enabled:
            return
        battle_index = max(1, int(battle_index))
        if self.current_battle != battle_index:
            self.start_battle(battle_index)
        state = state_from_action(image, payload)
        if self.pending:
            self.pending[-1]["next_state"] = state
            self.pending[-1]["next_frame"] = frame
            # A later action can alter the same hand/elixir evidence.  Close
            # the prior short-horizon window instead of attributing joint
            # effects to the earlier action.
            self.pending[-1]["feedback_window_closed"] = True
            self.pending[-1]["feedback_window_closed_reason"] = "next_action_sent"
        self.pending.append(
            {
                "state": state,
                "state_frame": frame,
                "action": action_from_payload(payload),
                "next_state": None,
                "next_frame": None,
                "action_observations": [],
            }
        )

    def observe_action_effect(self, battle_index: int, state: dict[str, Any]) -> None:
        """Capture compact observed states even while the policy holds its cards."""
        if not self.enabled or battle_index != self.current_battle or not self.pending:
            return
        current = self.pending[-1]
        elapsed = float(state.get("battle_elapsed_s", 0)) - float(current["state"].get("battle_elapsed_s", 0))
        if not 0.5 <= elapsed <= 8.0:
            return
        observations = current.setdefault("action_observations", [])
        if observations and float(state["battle_elapsed_s"]) - float(observations[-1]["battle_elapsed_s"]) < 0.6:
            return
        if len(observations) < 12:
            observations.append(dict(state))

    def _terminal_reward(self, result: BattleResult) -> float:
        base = {
            "win": self.win_reward,
            "loss": self.loss_reward,
            "draw": self.draw_reward,
        }.get(result.outcome, 0.0)
        crown_difference = 0
        if result.player_crowns is not None and result.opponent_crowns is not None:
            crown_difference = result.player_crowns - result.opponent_crowns
        return base + self.crown_reward * crown_difference

    def finish_battle(
        self,
        battle_index: int,
        result: BattleResult,
        frame: str | None,
        *, trial_runtime: dict | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        battle_index = max(1, int(battle_index))
        if self.current_battle != battle_index:
            self.current_battle = battle_index
        verified = (
            result.outcome in {"win", "loss", "draw"}
            and result.confidence >= self.minimum_result_confidence
        )
        reward = self._terminal_reward(result) if verified else 0.0
        terminal_state = {
            "schema": "terminal_state_v1",
            **result.to_dict(),
        }
        action_count = len(self.pending)
        confirmed_action_count = sum(
            str(item.get("action", {}).get("action_status", "legacy_unknown"))
            == "confirmed"
            for item in self.pending
        )
        action_statuses = [
            str(item.get("action", {}).get("action_status", "legacy_unknown"))
            for item in self.pending
        ]
        for index, transition in enumerate(self.pending):
            self.transition_sequence += 1
            is_last = index == action_count - 1
            if is_last:
                transition["next_state"] = terminal_state
                transition["next_frame"] = frame
            remaining = action_count - 1 - index
            row = {
                "schema_version": REPLAY_SCHEMA_VERSION,
                "transition_id": (
                    f"{self.run_dir.name}-b{battle_index:03d}-"
                    f"t{self.transition_sequence:06d}"
                ),
                "timestamp_unix": time.time(),
                "source": self.source,
                "battle_index": battle_index,
                **transition,
                "reward": round(reward if is_last else 0.0, 6),
                "return_to_go": round(reward * (self.gamma**remaining), 6),
                "done": is_last,
                "outcome": result.outcome,
                "reward_verified": verified,
                "action_status": str(
                    transition.get("action", {}).get("action_status", "legacy_unknown")
                ),
                "eligible_for_training": (
                    verified
                    and not self.policy_metadata.get("sl3_evaluation_only", False)
                    and self.allow_bot_training
                    and str(
                        transition.get("action", {}).get(
                            "action_status", "legacy_unknown"
                        )
                    )
                    == "confirmed"
                ),
                "policy": self.policy_metadata,
            }
            _append_jsonl(self.transitions_path, row)

        self.episode_sequence += 1
        episode = {
            "sl4_runtime": dict(trial_runtime or {}) if self.policy_metadata.get("sl4_deployment_id") else {},
            "sl3_runtime": dict(trial_runtime or {}),
            "schema_version": REPLAY_SCHEMA_VERSION,
            "episode_id": f"{self.run_dir.name}-b{battle_index:03d}",
            "timestamp_unix": time.time(),
            "source": self.source,
            "battle_index": battle_index,
            "action_count": action_count,
            **result.to_dict(),
            "terminal_reward": round(reward, 6),
            "reward_verified": verified,
            "confirmed_action_count": confirmed_action_count,
            "action_status_counts": {
                status: action_statuses.count(status)
                for status in sorted(set(action_statuses))
            },
            "eligible_for_training": (
                verified
                and not self.policy_metadata.get("sl3_evaluation_only", False)
                and self.allow_bot_training
                and action_count > 0
                and confirmed_action_count == action_count
            ),
            "result_frame": frame,
            "policy": self.policy_metadata,
        }
        _append_jsonl(self.episodes_path, episode)
        self.discard_current()
        return episode


@dataclass(frozen=True)
class ReplayAudit:
    runs: int
    episodes: int
    verified_episodes: int
    wins: int
    losses: int
    draws: int
    unknown_results: int
    transitions: int
    verified_transitions: int
    training_eligible_transitions: int
    confirmed_transitions: int
    unconfirmed_transitions: int
    action_status_counts: dict[str, int]
    win_rate: float
    collection_ready: bool
    bot_training_enabled: bool
    policy_breakdown: dict[str, dict[str, Any]]
    blocking_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["blocking_reasons"] = list(self.blocking_reasons)
        return value


def audit_replay(project_root: Path, config: dict[str, Any]) -> ReplayAudit:
    run_dirs = sorted((project_root.resolve() / "runs").glob("*"))
    episodes: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    used_runs = 0
    for run_dir in run_dirs:
        episode_rows = list(_read_jsonl(run_dir / "replay_episodes.jsonl"))
        transition_rows = list(_read_jsonl(run_dir / "replay_transitions.jsonl"))
        if episode_rows or transition_rows:
            used_runs += 1
        episodes.extend(episode_rows)
        transitions.extend(transition_rows)
    wins = sum(row.get("outcome") == "win" for row in episodes)
    losses = sum(row.get("outcome") == "loss" for row in episodes)
    draws = sum(row.get("outcome") == "draw" for row in episodes)
    unknown = len(episodes) - wins - losses - draws
    verified_episodes = sum(bool(row.get("reward_verified")) for row in episodes)
    verified_transitions = sum(
        bool(row.get("reward_verified")) for row in transitions
    )
    require_confirmation = bool(config.get("require_action_confirmation", False))
    status_counts: dict[str, int] = {}
    for row in transitions:
        action = row.get("action", {})
        status = str(
            row.get("action_status")
            or (action.get("action_status") if isinstance(action, dict) else "")
            or "legacy_unknown"
        )
        if status not in ACTION_CONFIRMATION_STATUSES:
            status = "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1
    confirmed = status_counts.get("confirmed", 0)
    eligible = sum(
        bool(row.get("eligible_for_training"))
        and (
            not require_confirmation
            or str(
                row.get("action_status")
                or (
                    row.get("action", {}).get("action_status")
                    if isinstance(row.get("action", {}), dict)
                    else ""
                )
                or "legacy_unknown"
            )
            == "confirmed"
        )
        for row in transitions
    )
    policy_breakdown: dict[str, dict[str, Any]] = {}

    def policy_name(row: dict[str, Any]) -> str:
        policy = row.get("policy", {})
        if not isinstance(policy, dict):
            return "unknown"
        return str(
            policy.get("rule_version")
            or policy.get("version")
            or policy.get("mode")
            or "unknown"
        )

    for row in episodes:
        name = policy_name(row)
        value = policy_breakdown.setdefault(
            name,
            {
                "episodes": 0,
                "verified_episodes": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "unknown_results": 0,
                "transitions": 0,
                "verified_transitions": 0,
                "action_count": 0,
                "terminal_reward_sum": 0.0,
            },
        )
        value["episodes"] += 1
        value["action_count"] += int(row.get("action_count", 0))
        value["terminal_reward_sum"] += float(row.get("terminal_reward", 0.0))
        if row.get("reward_verified"):
            value["verified_episodes"] += 1
        outcome = str(row.get("outcome", "unknown"))
        key = {
            "win": "wins",
            "loss": "losses",
            "draw": "draws",
        }.get(outcome, "unknown_results")
        value[key] += 1
    for row in transitions:
        value = policy_breakdown.setdefault(
            policy_name(row),
            {
                "episodes": 0,
                "verified_episodes": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "unknown_results": 0,
                "transitions": 0,
                "verified_transitions": 0,
                "action_count": 0,
                "terminal_reward_sum": 0.0,
            },
        )
        value["transitions"] += 1
        if row.get("reward_verified"):
            value["verified_transitions"] += 1
    for value in policy_breakdown.values():
        decided = value["wins"] + value["losses"] + value["draws"]
        episodes_count = value["episodes"]
        value["win_rate"] = round(value["wins"] / max(1, decided), 6)
        value["average_actions"] = round(
            value.pop("action_count") / max(1, episodes_count), 3
        )
        value["average_terminal_reward"] = round(
            value.pop("terminal_reward_sum") / max(1, episodes_count), 6
        )
    minimum_episodes = int(config.get("minimum_verified_episodes", 50))
    minimum_wins = int(config.get("minimum_wins", 10))
    minimum_losses = int(config.get("minimum_losses", 10))
    minimum_transitions = int(config.get("minimum_verified_transitions", 300))
    reasons: list[str] = []
    if verified_episodes < minimum_episodes:
        reasons.append(f"可信结算 {verified_episodes}/{minimum_episodes} 局")
    if wins < minimum_wins:
        reasons.append(f"胜局 {wins}/{minimum_wins}")
    if losses < minimum_losses:
        reasons.append(f"负局 {losses}/{minimum_losses}")
    if verified_transitions < minimum_transitions:
        reasons.append(f"可信经验 {verified_transitions}/{minimum_transitions} 条")
    decided = wins + losses + draws
    return ReplayAudit(
        runs=used_runs,
        episodes=len(episodes),
        verified_episodes=verified_episodes,
        wins=wins,
        losses=losses,
        draws=draws,
        unknown_results=unknown,
        transitions=len(transitions),
        verified_transitions=verified_transitions,
        training_eligible_transitions=eligible,
        confirmed_transitions=confirmed,
        unconfirmed_transitions=len(transitions) - confirmed,
        action_status_counts=status_counts,
        win_rate=round(wins / max(1, decided), 6),
        collection_ready=not reasons,
        bot_training_enabled=bool(config.get("allow_bot_training", False)),
        policy_breakdown=policy_breakdown,
        blocking_reasons=tuple(reasons),
    )


def backfill_replay_history(
    project_root: Path,
    config: dict[str, Any],
) -> dict[str, int]:
    scanned = 0
    written_runs = 0
    written_episodes = 0
    written_transitions = 0
    for run_dir in sorted((project_root.resolve() / "runs").glob("*")):
        events_path = run_dir / "events.jsonl"
        if not events_path.is_file() or (run_dir / "replay_manifest.json").is_file():
            continue
        events = list(_read_jsonl(events_path))
        if not any(row.get("event") == "battle_action" for row in events):
            continue
        scanned += 1
        recorder = ExperienceReplayRecorder(
            run_dir,
            config,
            policy_metadata={"version": "historical_unknown"},
            source="historical_offline_ai_bot",
        )
        battle_index = 0
        before_transitions = recorder.transition_sequence
        before_episodes = recorder.episode_sequence
        for event in events:
            event_name = str(event.get("event", ""))
            if event_name in {"battle_started", "battle_started_auto"}:
                battle_index += 1
                recorder.start_battle(battle_index)
                continue
            frame_value = event.get("frame")
            frame_path = run_dir / str(frame_value) if frame_value else None
            if event_name == "battle_action" and frame_path and frame_path.is_file():
                if battle_index <= 0:
                    battle_index = 1
                    recorder.start_battle(battle_index)
                with Image.open(frame_path) as loaded:
                    image = loaded.convert("RGB")
                recorder.record_action(
                    battle_index,
                    image,
                    event,
                    str(frame_value),
                )
                continue
            if event_name in {"battle_ended_auto", "navigation_result_continue"}:
                if frame_path and frame_path.is_file():
                    with Image.open(frame_path) as loaded:
                        image = loaded.convert("RGB")
                    result = detect_battle_result(image)
                else:
                    result = BattleResult(
                        "unknown", None, None, 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
                    )
                recorder.finish_battle(battle_index or 1, result, str(frame_value or ""))
        episode_delta = recorder.episode_sequence - before_episodes
        transition_delta = recorder.transition_sequence - before_transitions
        if episode_delta:
            written_runs += 1
            written_episodes += episode_delta
            written_transitions += transition_delta
    return {
        "scanned_runs": scanned,
        "written_runs": written_runs,
        "written_episodes": written_episodes,
        "written_transitions": written_transitions,
    }
