"""Conservative, battle-boundary control for baseline/candidate experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExperimentAssignment:
    arm: str
    batch_index: int
    battle_index: int


class ControlledExperiment:
    def __init__(self, config: dict[str, Any], replay_model: Any | None):
        self.config = dict(config)
        self.enabled = bool(self.config.get("enabled", False))
        self.batch_size = max(1, int(self.config.get("batch_size", 10)))
        self.replay_model = replay_model
        self.candidate_scale = float(getattr(replay_model, "influence_scale", 0.0))
        self.episodes: list[dict[str, Any]] = []
        self.stop_reason = ""

    def assignment(self, battle_index: int) -> ExperimentAssignment:
        index = max(1, int(battle_index))
        batch = (index - 1) // self.batch_size
        arm = "baseline" if batch % 2 == 0 else "candidate"
        return ExperimentAssignment(arm=arm, batch_index=batch, battle_index=index)

    def activate(self, battle_index: int) -> dict[str, Any]:
        assigned = self.assignment(battle_index)
        if self.replay_model is not None:
            if not self.enabled or assigned.arm == "candidate":
                self.replay_model.influence_scale = self.candidate_scale
            else:
                self.replay_model.influence_scale = 0.0
        champion = getattr(self.replay_model, "champion", None) or {}
        return {
            "experiment_enabled": self.enabled,
            "experiment_arm": assigned.arm if self.enabled else "normal",
            "experiment_batch_index": assigned.batch_index if self.enabled else None,
            "runtime_replay_version": champion.get("version"),
            "runtime_replay_loaded": bool(self.replay_model is not None and getattr(self.replay_model, "available", False)),
            "runtime_replay_influence_scale": round(float(getattr(self.replay_model, "influence_scale", 0.0)), 6),
        }

    def observe(self, episode: dict[str, Any]) -> str:
        if not self.enabled:
            return ""
        self.episodes.append(dict(episode))
        minimum = max(1, int(self.config.get("minimum_battles_before_guardrail", 10)))
        if len(self.episodes) < minimum:
            return ""
        unknown = sum(not bool(row.get("reward_verified")) for row in self.episodes)
        actions = sum(int(row.get("action_count", 0)) for row in self.episodes)
        confirmed = sum(int(row.get("confirmed_action_count", 0)) for row in self.episodes)
        unknown_rate = unknown / len(self.episodes)
        unconfirmed_rate = (actions - confirmed) / actions if actions else 0.0
        if unknown_rate > float(self.config.get("maximum_unknown_result_rate", 0.10)):
            self.stop_reason = f"unknown_result_rate={unknown_rate:.3f}"
        elif unconfirmed_rate > float(self.config.get("maximum_unconfirmed_action_rate", 0.10)):
            self.stop_reason = f"unconfirmed_action_rate={unconfirmed_rate:.3f}"
        return self.stop_reason
