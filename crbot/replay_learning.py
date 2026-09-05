from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from .battle_perception import LaneThreat
from .cards import CardCatalog, CardDefinition
from .imitation import (
    BATTLEFIELD_FEATURE_COUNT,
    battlefield_features,
    candidate_features,
)


REPLAY_POLICY_SCHEMA_VERSION = 1
CONTEXT_FEATURE_COUNT = 6


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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _policy_version(row: dict[str, Any]) -> str:
    policy = row.get("policy", {})
    if not isinstance(policy, dict):
        return "unknown"
    return str(
        policy.get("rule_version")
        or policy.get("version")
        or policy.get("mode")
        or "unknown"
    )


@dataclass(frozen=True)
class ReplayLearningAction:
    group_id: str
    transition_id: str
    card_id: str
    hand: tuple[str | None, ...]
    deploy_point: tuple[float, float]
    elixir: float
    left_threat: float
    right_threat: float
    left_threat_type: str
    right_threat_type: str
    left_threat_proximity: float
    right_threat_proximity: float
    left_threat_approach_rate: float
    right_threat_approach_rate: float
    left_unit_count: int
    right_unit_count: int
    threat_type: str
    threat_proximity: float
    threat_approach_rate: float
    battlefield_edges: tuple[float, ...]
    outcome: str
    return_to_go: float
    policy_version: str
    formation_phase: str
    desired_role: str
    battle_elapsed_s: float

    @property
    def target(self) -> float:
        return 1.0 if self.outcome == "win" else 0.0

    def threats(self) -> dict[str, LaneThreat]:
        def build(lane: str, score: float) -> LaneThreat:
            is_left = lane == "left"
            return LaneThreat(
                lane=lane,
                score=score,
                unit_count=self.left_unit_count if is_left else self.right_unit_count,
                proximity=(
                    self.left_threat_proximity
                    if is_left
                    else self.right_threat_proximity
                ),
                threat=self.left_threat_type if is_left else self.right_threat_type,
                centers=(),
                approach_rate=(
                    self.left_threat_approach_rate
                    if is_left
                    else self.right_threat_approach_rate
                ),
            )

        return {
            "left": build("left", self.left_threat),
            "right": build("right", self.right_threat),
        }


@dataclass(frozen=True)
class ReplayLearningAudit:
    runs: int
    verified_episodes: int
    wins: int
    losses: int
    actions: int
    distinct_cards: int
    target_policy_version: str
    ready: bool
    blocking_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["blocking_reasons"] = list(self.blocking_reasons)
        return value


def collect_replay_learning_actions(
    project_root: Path,
    catalog: CardCatalog,
    config: dict[str, Any],
) -> list[ReplayLearningAction]:
    target_version = str(config.get("training_policy_version", "")).strip()
    actions: list[ReplayLearningAction] = []
    seen: set[str] = set()
    for run_dir in sorted((project_root.resolve() / "runs").glob("*")):
        if not run_dir.is_dir():
            continue
        for row in _read_jsonl(run_dir / "replay_transitions.jsonl"):
            transition_id = str(row.get("transition_id", "")).strip()
            if not transition_id or transition_id in seen:
                continue
            if not bool(row.get("reward_verified")):
                continue
            outcome = str(row.get("outcome", "unknown"))
            if outcome not in {"win", "loss"}:
                continue
            policy_version = _policy_version(row)
            if target_version and policy_version != target_version:
                continue
            action = row.get("action", {})
            state = row.get("state", {})
            if not isinstance(action, dict) or not isinstance(state, dict):
                continue
            card_id = str(action.get("card_id") or "").strip()
            if catalog.get(card_id) is None:
                continue
            raw_hand = state.get("hand", [])
            if not isinstance(raw_hand, list):
                continue
            hand = tuple(
                str(value) if value and catalog.get(str(value)) is not None else None
                for value in raw_hand[:4]
            )
            if card_id not in hand:
                continue
            raw_deploy = action.get("deploy_point", [])
            if not isinstance(raw_deploy, list) or len(raw_deploy) < 2:
                continue
            deploy = (float(raw_deploy[0]), float(raw_deploy[1]))
            if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in deploy):
                continue
            raw_edges = state.get("battlefield_edges", [])
            if not isinstance(raw_edges, list) or len(raw_edges) != BATTLEFIELD_FEATURE_COUNT:
                edges = (0.0,) * BATTLEFIELD_FEATURE_COUNT
            else:
                edges = tuple(float(value) for value in raw_edges)
            battle_index = max(1, int(row.get("battle_index", 1)))
            group_id = f"{run_dir.name}:battle-{battle_index}"
            left_score = float(state.get("left_threat", 0.0))
            right_score = float(state.get("right_threat", 0.0))
            strongest = "left" if left_score >= right_score else "right"
            legacy_type = str(state.get("threat_type", "none"))
            legacy_proximity = float(state.get("threat_proximity", 0.0))
            legacy_approach = float(state.get("threat_approach_rate", 0.0))
            reason = str(action.get("reason", ""))
            formation_phase = str(action.get("formation_phase", "")).strip()
            if not formation_phase:
                formation_phase = reason.rsplit("_", 1)[0] if reason else "unknown"
            desired_role = str(action.get("desired_formation_role", "")).strip()
            if not desired_role:
                if formation_phase in {
                    "form_frontline",
                    "complete_frontline",
                    "protect_surviving_backline",
                }:
                    desired_role = "frontline"
                elif formation_phase in {
                    "support_frontline",
                    "support_counterpush",
                    "reinforce_push",
                    "stage_backline",
                }:
                    desired_role = "backline"
            actions.append(
                ReplayLearningAction(
                    group_id=group_id,
                    transition_id=transition_id,
                    card_id=card_id,
                    hand=hand,
                    deploy_point=deploy,
                    elixir=float(state.get("elixir", 0.0)),
                    left_threat=left_score,
                    right_threat=right_score,
                    left_threat_type=str(
                        state.get(
                            "left_threat_type",
                            legacy_type if strongest == "left" else "none",
                        )
                    ),
                    right_threat_type=str(
                        state.get(
                            "right_threat_type",
                            legacy_type if strongest == "right" else "none",
                        )
                    ),
                    left_threat_proximity=float(
                        state.get(
                            "left_threat_proximity",
                            legacy_proximity if strongest == "left" else 0.0,
                        )
                    ),
                    right_threat_proximity=float(
                        state.get(
                            "right_threat_proximity",
                            legacy_proximity if strongest == "right" else 0.0,
                        )
                    ),
                    left_threat_approach_rate=float(
                        state.get(
                            "left_threat_approach_rate",
                            legacy_approach if strongest == "left" else 0.0,
                        )
                    ),
                    right_threat_approach_rate=float(
                        state.get(
                            "right_threat_approach_rate",
                            legacy_approach if strongest == "right" else 0.0,
                        )
                    ),
                    left_unit_count=int(state.get("left_unit_count", 0)),
                    right_unit_count=int(state.get("right_unit_count", 0)),
                    threat_type=str(state.get("threat_type", "none")),
                    threat_proximity=float(state.get("threat_proximity", 0.0)),
                    threat_approach_rate=float(
                        state.get("threat_approach_rate", 0.0)
                    ),
                    battlefield_edges=edges,
                    outcome=outcome,
                    return_to_go=float(row.get("return_to_go", 0.0)),
                    policy_version=policy_version,
                    formation_phase=formation_phase,
                    desired_role=desired_role,
                    battle_elapsed_s=float(state.get("battle_elapsed_s", 0.0)),
                )
            )
            seen.add(transition_id)
    return actions


def audit_replay_learning(
    project_root: Path,
    catalog: CardCatalog,
    config: dict[str, Any],
) -> ReplayLearningAudit:
    actions = collect_replay_learning_actions(project_root, catalog, config)
    outcomes: dict[str, str] = {}
    runs: set[str] = set()
    cards: set[str] = set()
    for action in actions:
        outcomes[action.group_id] = action.outcome
        runs.add(action.group_id.split(":battle-", 1)[0])
        cards.add(action.card_id)
    wins = sum(value == "win" for value in outcomes.values())
    losses = sum(value == "loss" for value in outcomes.values())
    minimum_episodes = int(config.get("minimum_verified_episodes", 50))
    minimum_wins = int(config.get("minimum_wins", 10))
    minimum_losses = int(config.get("minimum_losses", 10))
    minimum_actions = int(config.get("minimum_verified_transitions", 300))
    minimum_cards = int(config.get("minimum_distinct_cards_for_training", 8))
    reasons: list[str] = []
    if len(outcomes) < minimum_episodes:
        reasons.append(f"同版本可信对局 {len(outcomes)}/{minimum_episodes} 局")
    if wins < minimum_wins:
        reasons.append(f"同版本胜局 {wins}/{minimum_wins}")
    if losses < minimum_losses:
        reasons.append(f"同版本负局 {losses}/{minimum_losses}")
    if len(actions) < minimum_actions:
        reasons.append(f"同版本可信动作 {len(actions)}/{minimum_actions} 条")
    if len(cards) < minimum_cards:
        reasons.append(f"同版本已用卡牌 {len(cards)}/{minimum_cards} 种")
    return ReplayLearningAudit(
        runs=len(runs),
        verified_episodes=len(outcomes),
        wins=wins,
        losses=losses,
        actions=len(actions),
        distinct_cards=len(cards),
        target_policy_version=str(config.get("training_policy_version", "")).strip(),
        ready=not reasons,
        blocking_reasons=tuple(reasons),
    )


def _feature(
    action: ReplayLearningAction,
    card: CardDefinition,
    visual_weight: float,
) -> np.ndarray:
    values = candidate_features(
        card,
        action.elixir,
        action.threats(),
        np.asarray(action.battlefield_edges, dtype=np.float32),
        visual_weight,
    )
    values.extend(
        _context_features(
            action.formation_phase,
            action.desired_role,
            action.battle_elapsed_s,
        )
    )
    return np.asarray(values, dtype=np.float32)


def _context_features(
    formation_phase: str,
    desired_role: str,
    battle_elapsed_s: float,
) -> list[float]:
    phase = formation_phase.casefold()
    role = desired_role.casefold()
    return [
        1.0 if "defense" in phase or phase.startswith("counter_") else 0.0,
        1.0 if role == "frontline" else 0.0,
        1.0 if role == "backline" else 0.0,
        1.0 if "counterpush" in phase else 0.0,
        1.0 if "stage" in phase else 0.0,
        min(1.0, max(0.0, float(battle_elapsed_s) / 180.0)),
    ]


def _stratified_group_split(
    actions: list[ReplayLearningAction],
    validation_fraction: float,
    seed: int,
) -> tuple[list[ReplayLearningAction], list[ReplayLearningAction]]:
    group_outcomes = {action.group_id: action.outcome for action in actions}
    validation_groups: set[str] = set()
    for outcome in ("win", "loss"):
        groups = sorted(
            group for group, value in group_outcomes.items() if value == outcome
        )
        if len(groups) < 2:
            raise ValueError(f"至少需要两局{outcome}对局才能划分训练集和验证集")
        ranked = sorted(
            groups,
            key=lambda value: hashlib.sha256(
                f"{seed}:{outcome}:{value}".encode()
            ).hexdigest(),
        )
        count = max(
            1,
            min(len(groups) - 1, math.ceil(len(groups) * validation_fraction)),
        )
        validation_groups.update(ranked[:count])
    return (
        [action for action in actions if action.group_id not in validation_groups],
        [action for action in actions if action.group_id in validation_groups],
    )


def _temporal_group_split(
    actions: list[ReplayLearningAction],
    validation_fraction: float,
) -> tuple[list[ReplayLearningAction], list[ReplayLearningAction]]:
    """Hold out the newest whole battles to detect policy/data drift."""
    groups = sorted({action.group_id for action in actions})
    if len(groups) < 4:
        raise ValueError("至少需要四局对局才能进行时间外推验证")
    count = max(2, min(len(groups) - 2, math.ceil(len(groups) * validation_fraction)))
    validation_groups = set(groups[-count:])
    train = [action for action in actions if action.group_id not in validation_groups]
    validation = [action for action in actions if action.group_id in validation_groups]
    train_outcomes = {action.outcome for action in train}
    validation_outcomes = {action.outcome for action in validation}
    if train_outcomes != {"win", "loss"} or validation_outcomes != {"win", "loss"}:
        raise ValueError("最新时间窗必须同时包含胜局和负局")
    return train, validation


def _class_weights(targets: np.ndarray) -> tuple[float, float]:
    positives = max(1, int(np.sum(targets >= 0.5)))
    negatives = max(1, len(targets) - positives)
    return len(targets) / (2.0 * positives), len(targets) / (2.0 * negatives)


def _balanced_knn_predict(
    training_x: np.ndarray,
    training_y: np.ndarray,
    sample: np.ndarray,
    neighbors: int,
    positive_weight: float,
    negative_weight: float,
) -> float:
    if len(training_x) == 0:
        return 0.5
    distances = np.sum((training_x - sample.reshape(1, -1)) ** 2, axis=1)
    count = min(max(1, neighbors), len(distances))
    indices = np.argpartition(distances, count - 1)[:count]
    values = training_y[indices]
    weights = np.where(values >= 0.5, positive_weight, negative_weight)
    denominator = float(np.sum(weights))
    if denominator <= 0.0:
        return 0.5
    return float(np.sum(values * weights) / denominator)


def _plain_knn_predict(
    training_x: np.ndarray,
    training_y: np.ndarray,
    sample: np.ndarray,
    neighbors: int,
) -> float:
    if len(training_x) == 0:
        return 0.5
    distances = np.sum((training_x - sample.reshape(1, -1)) ** 2, axis=1)
    count = min(max(1, neighbors), len(distances))
    indices = np.argpartition(distances, count - 1)[:count]
    return float(np.mean(training_y[indices]))


def _auc(targets: np.ndarray, predictions: np.ndarray) -> float:
    positive = predictions[targets >= 0.5]
    negative = predictions[targets < 0.5]
    if len(positive) == 0 or len(negative) == 0:
        return 0.5
    comparisons = positive.reshape(-1, 1) - negative.reshape(1, -1)
    return float(
        (np.sum(comparisons > 0.0) + 0.5 * np.sum(comparisons == 0.0))
        / comparisons.size
    )


def _evaluate_candidate(
    train: list[ReplayLearningAction],
    validation: list[ReplayLearningAction],
    catalog: CardCatalog,
    neighbors: int,
    visual_weight: float,
) -> dict[str, float]:
    training_x = np.asarray(
        [_feature(action, catalog.by_id[action.card_id], visual_weight) for action in train],
        dtype=np.float32,
    )
    training_y = np.asarray([action.target for action in train], dtype=np.float32)
    positive_weight, negative_weight = _class_weights(training_y)
    predictions = np.asarray(
        [
            _balanced_knn_predict(
                training_x,
                training_y,
                _feature(action, catalog.by_id[action.card_id], visual_weight),
                neighbors,
                positive_weight,
                negative_weight,
            )
            for action in validation
        ],
        dtype=np.float32,
    )
    targets = np.asarray([action.target for action in validation], dtype=np.float32)
    positive_mask = targets >= 0.5
    negative_mask = ~positive_mask
    guesses = predictions >= 0.5
    true_positive_rate = float(np.mean(guesses[positive_mask]))
    true_negative_rate = float(np.mean(~guesses[negative_mask]))
    balanced_accuracy = (true_positive_rate + true_negative_rate) / 2.0
    prior = float(np.mean(training_y))
    # The KNN probabilities are class-balanced for decision ranking. Restore
    # the observed training prior before measuring probability calibration.
    numerator = predictions * prior
    denominator = numerator + (1.0 - predictions) * (1.0 - prior)
    calibrated = np.divide(
        numerator,
        denominator,
        out=np.full_like(predictions, prior),
        where=denominator > 1e-8,
    )
    brier = float(np.mean((calibrated - targets) ** 2))
    baseline_brier = float(np.mean((prior - targets) ** 2))

    win_top_two: list[float] = []
    loss_bottom_two: list[float] = []
    for action in validation:
        cards = [
            catalog.get(card_id)
            for card_id in dict.fromkeys(value for value in action.hand if value)
        ]
        cards = [card for card in cards if card is not None]
        if len(cards) < 2:
            continue
        ranked = sorted(
            cards,
            key=lambda card: _balanced_knn_predict(
                training_x,
                training_y,
                _feature(action, card, visual_weight),
                neighbors,
                positive_weight,
                negative_weight,
            ),
            reverse=True,
        )
        actual_rank = next(
            (index for index, card in enumerate(ranked) if card.card_id == action.card_id),
            len(ranked) - 1,
        )
        top_count = min(2, len(ranked))
        if action.outcome == "win":
            win_top_two.append(float(actual_rank < top_count))
        else:
            loss_bottom_two.append(float(actual_rank >= len(ranked) - top_count))

    deploy_train = [action for action in train if action.outcome == "win"]
    deploy_validation = [action for action in validation if action.outcome == "win"]
    deploy_x = np.asarray(
        [
            _feature(action, catalog.by_id[action.card_id], visual_weight)
            for action in deploy_train
        ],
        dtype=np.float32,
    )
    deploy_y = np.asarray(
        [action.deploy_point for action in deploy_train], dtype=np.float32
    )
    deploy_errors: list[float] = []
    for action in deploy_validation:
        sample = _feature(action, catalog.by_id[action.card_id], visual_weight)
        predicted_x = _plain_knn_predict(
            deploy_x, deploy_y[:, 0], sample, neighbors
        )
        predicted_y = _plain_knn_predict(
            deploy_x, deploy_y[:, 1], sample, neighbors
        )
        deploy_errors.append(
            (abs(predicted_x - action.deploy_point[0]) + abs(predicted_y - action.deploy_point[1]))
            / 2.0
        )

    groups = {action.group_id: action.outcome for action in validation}
    win_groups = sum(value == "win" for value in groups.values())
    loss_groups = sum(value == "loss" for value in groups.values())
    win_top_two_rate = float(np.mean(win_top_two)) if win_top_two else 0.0
    loss_bottom_two_rate = float(np.mean(loss_bottom_two)) if loss_bottom_two else 0.0
    return {
        "validation_actions": float(len(validation)),
        "validation_episodes": float(len(groups)),
        "validation_wins": float(win_groups),
        "validation_losses": float(loss_groups),
        "balanced_accuracy": round(balanced_accuracy, 6),
        "auc": round(_auc(targets, predictions), 6),
        "brier_score": round(brier, 6),
        "baseline_brier_score": round(baseline_brier, 6),
        "brier_improvement": round(baseline_brier - brier, 6),
        "value_margin": round(
            float(np.mean(predictions[positive_mask]) - np.mean(predictions[negative_mask])),
            6,
        ),
        "win_selected_top2_rate": round(win_top_two_rate, 6),
        "loss_selected_bottom2_rate": round(loss_bottom_two_rate, 6),
        "rank_separation": round((win_top_two_rate + loss_bottom_two_rate) / 2.0, 6),
        "win_deploy_mae": round(float(np.mean(deploy_errors)), 6),
    }


class ReplayPolicyRegistry:
    def __init__(self, project_root: Path):
        self.root = project_root.resolve() / "models" / "replay_policy"
        self.path = self.root / "registry.json"

    def load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {
                "schema_version": REPLAY_POLICY_SCHEMA_VERSION,
                "champion": None,
                "candidates": [],
            }
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {
                "schema_version": REPLAY_POLICY_SCHEMA_VERSION,
                "champion": None,
                "candidates": [],
            }
        return value if isinstance(value, dict) else {}

    def champion(self) -> dict[str, Any] | None:
        value = self.load().get("champion")
        return value if isinstance(value, dict) else None

    def champion_path(self) -> Path | None:
        champion = self.champion()
        if champion is None:
            return None
        path = self.root / str(champion.get("model_path", ""))
        return path if path.is_file() else None

    def register(
        self,
        model_path: Path,
        metrics: dict[str, float],
        config: dict[str, Any],
        manifest: dict[str, Any],
    ) -> dict[str, Any]:
        version = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}"
        relative = Path("candidates") / f"replay_policy_{version}.npz"
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, target)
        reasons: list[str] = []
        checks = (
            ("validation_episodes", "minimum_validation_episodes", 12, "验证对局不足"),
            ("validation_wins", "minimum_validation_wins", 3, "验证胜局不足"),
            ("validation_losses", "minimum_validation_losses", 6, "验证负局不足"),
        )
        for metric, setting, default, message in checks:
            if int(metrics.get(metric, 0)) < int(config.get(setting, default)):
                reasons.append(message)
        if float(metrics.get("balanced_accuracy", 0.0)) < float(
            config.get("minimum_balanced_accuracy", 0.55)
        ):
            reasons.append("胜负平衡准确率未达标")
        if float(metrics.get("auc", 0.0)) < float(config.get("minimum_auc", 0.58)):
            reasons.append("胜负排序 AUC 未达标")
        if float(metrics.get("brier_improvement", -1.0)) < float(
            config.get("minimum_brier_improvement", 0.005)
        ):
            reasons.append("概率误差没有超过常数基线")
        if float(metrics.get("rank_separation", 0.0)) < float(
            config.get("minimum_rank_separation", 0.53)
        ):
            reasons.append("胜局选牌与败局选牌的影子排序分离不足")
        if float(metrics.get("win_deploy_mae", 1.0)) > float(
            config.get("maximum_win_deploy_mae", 0.14)
        ):
            reasons.append("胜局落点复现误差未达标")
        if bool(config.get("require_temporal_validation", False)):
            temporal_checks = (
                (
                    "temporal_balanced_accuracy",
                    "minimum_temporal_balanced_accuracy",
                    0.55,
                    "时间外推胜负平衡准确率未达标",
                ),
                (
                    "temporal_auc",
                    "minimum_temporal_auc",
                    0.58,
                    "时间外推 AUC 未达标",
                ),
                (
                    "temporal_brier_improvement",
                    "minimum_temporal_brier_improvement",
                    0.0,
                    "时间外推概率误差没有超过基线",
                ),
                (
                    "temporal_rank_separation",
                    "minimum_temporal_rank_separation",
                    0.53,
                    "时间外推选牌排序分离不足",
                ),
            )
            for metric, setting, default, message in temporal_checks:
                if float(metrics.get(metric, -1.0)) < float(
                    config.get(setting, default)
                ):
                    reasons.append(message)

        score = (
            float(metrics.get("auc", 0.0))
            + float(metrics.get("balanced_accuracy", 0.0))
            + 0.5 * float(metrics.get("rank_separation", 0.0))
            + float(metrics.get("brier_improvement", 0.0))
            - 0.25 * float(metrics.get("win_deploy_mae", 1.0))
        )
        registry = self.load()
        promotion_blockers: list[str] = []
        quality_passed = not reasons
        if not bool(config.get("allow_bot_training", False)):
            promotion_blockers.append("配置为影子模式，未授权机器人经验接管实战")
        champion = registry.get("champion")
        if quality_passed and not promotion_blockers and isinstance(champion, dict):
            improvement = float(config.get("minimum_score_improvement", 0.01))
            if score < float(champion.get("score", 0.0)) + improvement:
                promotion_blockers.append("综合分未超过当前回放策略")
        promoted = quality_passed and not promotion_blockers
        status = "champion" if promoted else ("shadow_pass" if quality_passed else "rejected")
        candidate = {
            "version": version,
            "created_at_unix": time.time(),
            "model_path": relative.as_posix(),
            "metrics": metrics,
            "score": round(score, 6),
            "manifest": manifest,
            "quality_passed": quality_passed,
            "status": status,
            "influence_scale": round(
                float(config.get("assisted_policy_weight_scale", 0.10)), 6
            ),
            "promoted": promoted,
            "rejection_reasons": reasons,
            "promotion_blockers": promotion_blockers,
        }
        if promoted:
            registry["champion"] = dict(candidate)
        registry.setdefault("candidates", []).append(candidate)
        _write_json(self.path, registry)
        return candidate


def train_replay_policy(
    project_root: Path,
    catalog: CardCatalog,
    config: dict[str, Any],
) -> dict[str, Any]:
    audit = audit_replay_learning(project_root, catalog, config)
    if not audit.ready:
        raise ValueError("回放数据尚未达标：" + "；".join(audit.blocking_reasons))
    actions = collect_replay_learning_actions(project_root, catalog, config)
    validation_fraction = float(config.get("validation_fraction", 0.25))
    seed = int(config.get("training_seed", config.get("seed", 20260903)))
    neighbors = int(config.get("training_neighbors", 31))
    visual_weight = float(config.get("battlefield_visual_weight", 0.08))
    train, validation = _stratified_group_split(
        actions, validation_fraction, seed
    )
    metrics = _evaluate_candidate(
        train, validation, catalog, neighbors, visual_weight
    )
    temporal_train: list[ReplayLearningAction] = []
    temporal_validation: list[ReplayLearningAction] = []
    if bool(config.get("require_temporal_validation", False)):
        temporal_train, temporal_validation = _temporal_group_split(
            actions, float(config.get("temporal_validation_fraction", 0.25))
        )
        temporal_metrics = _evaluate_candidate(
            temporal_train,
            temporal_validation,
            catalog,
            neighbors,
            visual_weight,
        )
        metrics.update(
            {f"temporal_{key}": value for key, value in temporal_metrics.items()}
        )

    value_x = np.asarray(
        [_feature(action, catalog.by_id[action.card_id], visual_weight) for action in actions],
        dtype=np.float32,
    )
    value_y = np.asarray([action.target for action in actions], dtype=np.float32)
    positive_weight, negative_weight = _class_weights(value_y)
    winning_actions = [action for action in actions if action.outcome == "win"]
    deploy_x = np.asarray(
        [
            _feature(action, catalog.by_id[action.card_id], visual_weight)
            for action in winning_actions
        ],
        dtype=np.float32,
    )
    deploy_y = np.asarray(
        [action.deploy_point for action in winning_actions], dtype=np.float32
    )
    temporary_dir = project_root.resolve() / "models" / "replay_policy" / "temporary"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    temporary_model = temporary_dir / "candidate.npz"
    np.savez_compressed(
        temporary_model,
        value_x=value_x,
        value_y=value_y,
        deploy_x=deploy_x,
        deploy_y=deploy_y,
        neighbors=np.asarray([neighbors], dtype=np.int32),
        positive_weight=np.asarray([positive_weight], dtype=np.float32),
        negative_weight=np.asarray([negative_weight], dtype=np.float32),
        visual_weight=np.asarray([visual_weight], dtype=np.float32),
        visual_feature_count=np.asarray([BATTLEFIELD_FEATURE_COUNT], dtype=np.int32),
        context_feature_count=np.asarray([CONTEXT_FEATURE_COUNT], dtype=np.int32),
    )
    train_groups = sorted({action.group_id for action in train})
    validation_groups = sorted({action.group_id for action in validation})
    manifest = {
        "source": "verified_offline_ai_replay",
        "policy_actions_are_ground_truth": False,
        "target_policy_version": audit.target_policy_version,
        "total_actions": len(actions),
        "train_actions": len(train),
        "validation_actions": len(validation),
        "train_battles": train_groups,
        "validation_battles": validation_groups,
        "validation_groups_are_disjoint": not bool(
            set(train_groups).intersection(validation_groups)
        ),
        "validation_method": "deterministic_stratified_whole_battle_holdout",
        "temporal_validation_battles": sorted(
            {action.group_id for action in temporal_validation}
        ),
        "temporal_validation_method": (
            "newest_whole_battle_holdout"
            if temporal_validation
            else "disabled"
        ),
        "features": (
            "battlefield_edges_elixir_lane_threat_card_roles_formation_phase_"
            "desired_rank_and_battle_time_without_card_identity"
        ),
        "deployment_training_source": "winning_episodes_only",
        "final_model_refit_on_all_verified_actions": True,
        "runtime_requires_explicit_allow_bot_training": True,
    }
    candidate = ReplayPolicyRegistry(project_root).register(
        temporary_model, metrics, config, manifest
    )
    return {
        "audit": audit.to_dict(),
        "metrics": metrics,
        "candidate": candidate,
    }


class ReplayPolicyModel:
    def __init__(self, project_root: Path):
        self.registry = ReplayPolicyRegistry(project_root)
        self.champion = self.registry.champion()
        self.available = False
        self.influence_scale = 0.0
        self.visual_weight = 0.0
        self.visual_feature_count = 0
        self.context_feature_count = 0
        self._cached_image_id: int | None = None
        self._cached_battlefield: np.ndarray | None = None
        path = self.registry.champion_path()
        if path is None:
            return
        try:
            with np.load(path) as data:
                self.value_x = data["value_x"].copy()
                self.value_y = data["value_y"].copy()
                self.deploy_x = data["deploy_x"].copy()
                self.deploy_y = data["deploy_y"].copy()
                self.neighbors = int(data["neighbors"][0])
                self.positive_weight = float(data["positive_weight"][0])
                self.negative_weight = float(data["negative_weight"][0])
                self.visual_weight = float(data["visual_weight"][0])
                self.visual_feature_count = int(data["visual_feature_count"][0])
                if "context_feature_count" in data:
                    self.context_feature_count = int(data["context_feature_count"][0])
            self.influence_scale = float(
                (self.champion or {}).get("influence_scale", 0.10)
            )
            self.available = bool(len(self.value_x) and len(self.deploy_x))
        except (OSError, ValueError, KeyError, IndexError):
            self.available = False

    def _battlefield(self, image: Image.Image | None) -> np.ndarray | None:
        if self.visual_feature_count <= 0:
            return None
        if image is None:
            if self._cached_battlefield is not None:
                return self._cached_battlefield
            return np.zeros(self.visual_feature_count, dtype=np.float32)
        image_id = id(image)
        if self._cached_image_id != image_id or self._cached_battlefield is None:
            self._cached_image_id = image_id
            self._cached_battlefield = battlefield_features(image)
        return self._cached_battlefield

    def prepare_frame(self, image: Image.Image) -> None:
        self._battlefield(image)

    def _feature(
        self,
        card: CardDefinition,
        elixir: float,
        threats: dict[str, LaneThreat],
        image: Image.Image | None,
        formation_phase: str,
        desired_role: str,
        battle_elapsed_s: float,
    ) -> np.ndarray:
        values = candidate_features(
            card,
            elixir,
            threats,
            self._battlefield(image),
            self.visual_weight,
        )
        if self.context_feature_count:
            values.extend(
                _context_features(
                    formation_phase,
                    desired_role,
                    battle_elapsed_s,
                )[: self.context_feature_count]
            )
        return np.asarray(values, dtype=np.float32)

    def card_score(
        self,
        card: CardDefinition,
        elixir: float,
        threats: dict[str, LaneThreat],
        image: Image.Image | None = None,
        formation_phase: str = "",
        desired_role: str = "",
        battle_elapsed_s: float = 0.0,
    ) -> float:
        if not self.available:
            return 0.5
        return _balanced_knn_predict(
            self.value_x,
            self.value_y,
            self._feature(
                card,
                elixir,
                threats,
                image,
                formation_phase,
                desired_role,
                battle_elapsed_s,
            ),
            self.neighbors,
            self.positive_weight,
            self.negative_weight,
        )

    def deploy_point(
        self,
        card: CardDefinition,
        elixir: float,
        threats: dict[str, LaneThreat],
        image: Image.Image | None = None,
        formation_phase: str = "",
        desired_role: str = "",
        battle_elapsed_s: float = 0.0,
    ) -> list[float] | None:
        if not self.available or len(self.deploy_x) == 0:
            return None
        sample = self._feature(
            card,
            elixir,
            threats,
            image,
            formation_phase,
            desired_role,
            battle_elapsed_s,
        )
        return [
            _plain_knn_predict(
                self.deploy_x, self.deploy_y[:, 0], sample, self.neighbors
            ),
            _plain_knn_predict(
                self.deploy_x, self.deploy_y[:, 1], sample, self.neighbors
            ),
        ]
