"""Conservative, battle-boundary control for baseline/candidate experiments."""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentAssignment:
    arm: str
    batch_index: int
    battle_index: int


class ControlledExperiment:
    def __init__(
        self,
        config: dict[str, Any],
        replay_model: Any | None,
        *,
        prior_episodes: list[dict[str, Any]] | None = None,
    ):
        self.config = dict(config)
        self.enabled = bool(self.config.get("enabled", False))
        self.batch_size = max(1, int(self.config.get("batch_size", 10)))
        self.replay_model = replay_model
        self.candidate_scale = float(getattr(replay_model, "influence_scale", 0.0))
        self.episodes = [deepcopy(row) for row in (prior_episodes or [])]
        self.battle_offset = self._restored_battle_offset()
        self.stop_reason = ""

    def _restored_battle_offset(self) -> int:
        recorded_indices = [
            int(row.get("policy", {}).get("experiment_battle_index"))
            for row in self.episodes
            if isinstance(row.get("policy"), dict)
            and row["policy"].get("experiment_battle_index") is not None
        ]
        if recorded_indices:
            return max(recorded_indices)

        # Older runs did not persist an absolute experiment index. Collapse
        # excess repetitions caused by process restarts so each arm still gets
        # a complete batch before alternation continues.
        baseline = sum(
            row.get("policy", {}).get("experiment_arm") == "baseline"
            for row in self.episodes
            if isinstance(row.get("policy"), dict)
        )
        candidate = sum(
            row.get("policy", {}).get("experiment_arm") == "candidate"
            for row in self.episodes
            if isinstance(row.get("policy"), dict)
        )
        cycles = min(baseline // self.batch_size, candidate // self.batch_size)
        baseline_remainder = max(0, baseline - cycles * self.batch_size)
        candidate_remainder = max(0, candidate - cycles * self.batch_size)
        offset = cycles * self.batch_size * 2
        if baseline_remainder < self.batch_size:
            return offset + baseline_remainder
        return offset + self.batch_size + min(candidate_remainder, self.batch_size)

    def assignment(self, battle_index: int) -> ExperimentAssignment:
        index = self.battle_offset + max(1, int(battle_index))
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
            "experiment_battle_index": assigned.battle_index if self.enabled else None,
            "experiment_guardrail_version": (
                self.config.get("guardrail_version") if self.enabled else None
            ),
            "runtime_replay_version": champion.get("version"),
            "runtime_replay_loaded": bool(self.replay_model is not None and getattr(self.replay_model, "available", False)),
            "runtime_replay_influence_scale": round(float(getattr(self.replay_model, "influence_scale", 0.0)), 6),
        }

    def observe(self, episode: dict[str, Any]) -> str:
        if not self.enabled:
            return ""
        # The recorder reuses policy_metadata across battles. Freeze each
        # completed episode before the next assignment mutates that mapping.
        self.episodes.append(deepcopy(episode))
        policy = episode.get("policy", {})
        if not isinstance(policy, dict) or policy.get("experiment_arm") != "candidate":
            return ""
        current_batch = policy.get("experiment_batch_index")
        candidate_episodes = self._unique_batch_episodes(
            [
            row
            for row in self.episodes
            if isinstance(row.get("policy"), dict)
            and row["policy"].get("experiment_arm") == "candidate"
            and row["policy"].get("experiment_batch_index") == current_batch
            ]
        )
        minimum = max(1, int(self.config.get("minimum_battles_before_guardrail", 10)))
        if len(candidate_episodes) < minimum:
            return ""
        unknown = sum(not bool(row.get("reward_verified")) for row in candidate_episodes)
        actions = sum(int(row.get("action_count", 0)) for row in candidate_episodes)
        confirmed = sum(int(row.get("confirmed_action_count", 0)) for row in candidate_episodes)
        unknown_rate = unknown / len(candidate_episodes)
        unconfirmed_rate = (actions - confirmed) / actions if actions else 0.0
        if unknown_rate > float(self.config.get("maximum_unknown_result_rate", 0.10)):
            self.stop_reason = f"unknown_result_rate={unknown_rate:.3f}"
        else:
            absolute_limit = self.config.get("maximum_unconfirmed_action_rate")
            if absolute_limit is not None and unconfirmed_rate > float(absolute_limit):
                self.stop_reason = f"unconfirmed_action_rate={unconfirmed_rate:.3f}"
            relative_limit = self.config.get(
                "maximum_confirmation_rate_drop_vs_baseline"
            )
            baseline_episodes = self._unique_batch_episodes(
                [
                    row
                    for row in self.episodes
                    if isinstance(row.get("policy"), dict)
                    and row["policy"].get("experiment_arm") == "baseline"
                    and row["policy"].get("experiment_batch_index")
                    == int(current_batch) - 1
                ]
            ) if current_batch is not None else []
            baseline_actions = sum(
                int(row.get("action_count", 0)) for row in baseline_episodes
            )
            baseline_confirmed = sum(
                int(row.get("confirmed_action_count", 0))
                for row in baseline_episodes
            )
            if (
                not self.stop_reason
                and relative_limit is not None
                and len(baseline_episodes) >= minimum
                and baseline_actions > 0
                and actions > 0
            ):
                baseline_rate = baseline_confirmed / baseline_actions
                candidate_rate = confirmed / actions
                rate_drop = baseline_rate - candidate_rate
                if rate_drop > float(relative_limit):
                    self.stop_reason = (
                        f"confirmation_rate_drop_vs_baseline={rate_drop:.3f}"
                    )
        return self.stop_reason

    @staticmethod
    def _unique_batch_episodes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        unique: dict[object, dict[str, Any]] = {}
        for row in sorted(rows, key=lambda item: float(item.get("timestamp_unix", 0.0))):
            policy = row.get("policy", {})
            index = policy.get("experiment_battle_index") if isinstance(policy, dict) else None
            key: object = index if index is not None else row.get("episode_id", id(row))
            unique.setdefault(key, row)
        return list(unique.values())

    def rollback_to_baseline(self) -> dict[str, Any]:
        """Disable candidate influence without replacing or deleting its model."""
        if self.replay_model is not None:
            self.replay_model.influence_scale = 0.0
        return {
            "experiment_arm": "baseline",
            "runtime_replay_influence_scale": 0.0,
            "rollback_reason": self.stop_reason or "manual_boundary_rollback",
        }


def load_controlled_experiment_episodes(
    project_root: Path, replay_version: str | None = None
) -> list[dict[str, Any]]:
    """Load completed experiment episodes, optionally for one replay champion."""
    rows: list[dict[str, Any]] = []
    for path in sorted((project_root.resolve() / "runs").glob("*/replay_episodes.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            policy = row.get("policy", {})
            if (
                isinstance(policy, dict)
                and policy.get("experiment_enabled")
                and (
                    replay_version is None
                    or policy.get("runtime_replay_version") == replay_version
                )
            ):
                rows.append(row)
    return rows


def audit_controlled_experiment(project_root: Path) -> dict[str, Any]:
    """Summarize completed experiment episodes and detect arm contamination."""
    raw_rows = load_controlled_experiment_episodes(project_root)
    rows: list[dict[str, Any]] = []
    seen: dict[tuple[str, int], str] = {}
    duplicates: list[str] = []
    for row in sorted(raw_rows, key=lambda item: float(item.get("timestamp_unix", 0.0))):
        policy = row.get("policy", {})
        index = policy.get("experiment_battle_index") if isinstance(policy, dict) else None
        version = str(policy.get("runtime_replay_version", "")) if isinstance(policy, dict) else ""
        if index is not None:
            key = (version, int(index))
            if key in seen:
                duplicates.append(
                    f"{version}:{index}:{seen[key]}:{row.get('episode_id', 'unknown')}"
                )
                continue
            seen[key] = str(row.get("episode_id", "unknown"))
        rows.append(row)
    arms: dict[str, dict[str, Any]] = {}
    contamination: list[str] = []
    for row in rows:
        policy = row.get("policy", {})
        arm = str(policy.get("experiment_arm", "unknown"))
        summary = arms.setdefault(arm, {
            "episodes": 0, "verified": 0, "wins": 0, "losses": 0,
            "draws": 0, "unknown": 0, "actions": 0, "confirmed_actions": 0,
        })
        summary["episodes"] += 1
        verified = bool(row.get("reward_verified"))
        summary["verified"] += int(verified)
        outcome = str(row.get("outcome", "unknown")) if verified else "unknown"
        summary[outcome if outcome in {"wins", "losses", "draws"} else {
            "win": "wins", "loss": "losses", "draw": "draws"
        }.get(outcome, "unknown")] += 1
        summary["actions"] += int(row.get("action_count", 0))
        summary["confirmed_actions"] += int(row.get("confirmed_action_count", 0))
        scale = float(policy.get("runtime_replay_influence_scale", 0.0))
        loaded = bool(policy.get("runtime_replay_loaded", False))
        episode_id = str(row.get("episode_id", "unknown"))
        if arm == "baseline" and scale != 0.0:
            contamination.append(f"{episode_id}:baseline_nonzero_scale")
        if arm == "candidate" and (not loaded or scale <= 0.0):
            contamination.append(f"{episode_id}:candidate_not_active")
    for summary in arms.values():
        verified = int(summary["verified"])
        actions = int(summary["actions"])
        summary["win_rate"] = round(summary["wins"] / verified, 6) if verified else None
        summary["confirmation_rate"] = round(summary["confirmed_actions"] / actions, 6) if actions else None
    comparison = _audit_comparison_groups(raw_rows)
    return {
        "schema": "controlled_experiment_audit_v3",
        "raw_episodes": len(raw_rows),
        "episodes": len(rows),
        "arms": arms,
        "contamination": contamination,
        "duplicate_experiment_indices": duplicates,
        "comparison": comparison,
        "ready_for_comparison": comparison["ready_for_comparison"],
    }


def _audit_comparison_groups(raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep exploratory cohorts visible without claiming missing identity is controlled."""
    required = (
        "runtime_replay_version", "experiment_guardrail_version", "mode",
        "runtime_model", "rule_version", "imitation_version", "code_commit",
        "config_sha256", "replay_model_sha256", "deck_id", "battle_mode",
        "environment_id",
    )
    by_index: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in raw_rows:
        policy = row.get("policy", {})
        if not isinstance(policy, dict):
            continue
        index = policy.get("experiment_battle_index")
        if index is not None:
            by_index.setdefault((str(policy.get("runtime_replay_version")), int(index)), []).append(row)

    conflicts = {
        key for key, items in by_index.items()
        if len({json.dumps(item, sort_keys=True, ensure_ascii=False) for item in items}) > 1
    }
    exclusions: list[dict[str, Any]] = []
    groups: dict[str, dict[str, Any]] = {}
    seen: set[tuple[str, int]] = set()
    for row in sorted(raw_rows, key=lambda item: float(item.get("timestamp_unix", 0.0))):
        policy = row.get("policy", {})
        if not isinstance(policy, dict):
            continue
        episode_id = str(row.get("episode_id", "unknown"))
        index = policy.get("experiment_battle_index")
        key = (str(policy.get("runtime_replay_version")), int(index)) if index is not None else None
        if key in conflicts:
            exclusions.append({"episode_id": episode_id, "reason": "conflicting_experiment_index"})
            continue
        if key is not None and key in seen:
            exclusions.append({"episode_id": episode_id, "reason": "identical_duplicate"})
            continue
        if key is not None:
            seen.add(key)
        missing = [field for field in required if not policy.get(field)]
        if index is None:
            missing.append("experiment_battle_index")
        if policy.get("experiment_batch_index") is None:
            missing.append("experiment_batch_index")
        arm = policy.get("experiment_arm")
        scale = float(policy.get("runtime_replay_influence_scale", 0.0))
        if arm not in {"baseline", "candidate"}:
            exclusions.append({"episode_id": episode_id, "reason": "invalid_arm"})
            continue
        if arm == "baseline" and scale != 0.0 or arm == "candidate" and (
            not policy.get("runtime_replay_loaded") or scale <= 0.0
        ):
            exclusions.append({"episode_id": episode_id, "reason": "arm_contamination"})
            continue
        identity = {field: policy.get(field) for field in required}
        # Baseline and candidate must share a cohort. Candidate scale is reported
        # separately because the baseline deliberately sets it to zero.
        group_key = json.dumps(identity, sort_keys=True, ensure_ascii=False)
        group = groups.setdefault(group_key, {"identity": identity, "arms": {}, "batches": {}, "missing_identity_fields": missing})
        group["missing_identity_fields"] = sorted(set(group["missing_identity_fields"]) | set(missing))
        for target in (group["arms"], group["batches"].setdefault(str(policy.get("experiment_batch_index")), {})):
            summary = target.setdefault(arm, {"episodes": 0, "verified": 0, "wins": 0, "losses": 0, "draws": 0, "unknown": 0})
            summary["episodes"] += 1
            verified = bool(row.get("reward_verified"))
            summary["verified"] += int(verified)
            outcome = str(row.get("outcome", "unknown")) if verified else "unknown"
            bucket = {"win": "wins", "loss": "losses", "draw": "draws"}.get(outcome, outcome)
            summary[bucket if bucket in {"wins", "losses", "draws"} else "unknown"] += 1
        if missing:
            exclusions.append({"episode_id": episode_id, "reason": "missing_identity", "fields": missing})
    result = list(groups.values())
    for group in result:
        batches = group["batches"]
        complete_pairs = [
            [batch, batch + 1] for batch in sorted(int(value) for value in batches if value != "None")
            if batch % 2 == 0
            and batches[str(batch)].get("baseline", {}).get("episodes", 0) >= 10
            and batches.get(str(batch + 1), {}).get("candidate", {}).get("episodes", 0) >= 10
        ]
        group["complete_batch_pairs"] = complete_pairs
        group["ready_for_comparison"] = not group["missing_identity_fields"] and bool(complete_pairs)
    return {"groups": result, "exclusions": exclusions,
            "conflicting_indices": [f"{version}:{index}" for version, index in sorted(conflicts)],
            "ready_for_comparison": any(group["ready_for_comparison"] for group in result)}
