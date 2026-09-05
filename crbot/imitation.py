from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image

from .battle_perception import LaneThreat
from .cards import CardCatalog, CardDefinition


ROLE_FEATURES = (
    "cheap",
    "tank",
    "win_condition",
    "splash",
    "support",
    "tank_killer",
    "swarm",
    "building",
    "spell",
    "ranged",
    "air",
    "spawner",
)
THREAT_TYPES = ("single", "heavy", "swarm")
BATTLEFIELD_GRID = (8, 12)
BATTLEFIELD_FEATURE_COUNT = BATTLEFIELD_GRID[0] * BATTLEFIELD_GRID[1]
STRATEGIC_ROLES = frozenset(
    {
        "tank",
        "win_condition",
        "splash",
        "support",
        "tank_killer",
        "swarm",
        "building",
        "spell",
        "ranged",
        "air",
        "spawner",
    }
)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_number} 行不是有效 JSON") from exc
            if isinstance(value, dict):
                yield value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def list_demonstration_runs(project_root: Path) -> list[Path]:
    root = project_root.resolve() / "demonstrations"
    if not root.is_dir():
        return []
    return sorted(
        [path for path in root.iterdir() if (path / "actions.jsonl").is_file()],
        key=lambda path: path.name,
    )


@dataclass(frozen=True)
class ExpertAction:
    run_name: str
    action_id: str
    battle_index: int
    slot_index: int
    deploy_point: tuple[float, float]
    hand: tuple[str | None, ...]
    elixir: float
    threats: dict[str, LaneThreat]
    before_frame: Path | None = None
    selected_card_source: str = "recognizer"

    @property
    def group_id(self) -> str:
        return f"{self.run_name}:battle-{self.battle_index}"


@dataclass(frozen=True)
class ImitationAudit:
    runs: int
    raw_actions: int
    usable_actions: int
    battles: int
    distinct_selected_cards: int
    selected_card_counts: dict[str, int]
    rejected_actions: int
    automatically_reconstructed_actions: int
    ready: bool
    blocking_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["blocking_reasons"] = list(self.blocking_reasons)
        return value


def _lane_threat(value: dict[str, Any], lane: str) -> LaneThreat:
    raw = value.get(lane, {}) if isinstance(value, dict) else {}
    centers = tuple(
        (float(point[0]), float(point[1]))
        for point in raw.get("centers", [])
        if isinstance(point, list) and len(point) == 2
    )
    return LaneThreat(
        lane=lane,
        score=float(raw.get("score", 0.0)),
        unit_count=int(raw.get("unit_count", 0)),
        proximity=float(raw.get("proximity", 0.0)),
        threat=str(raw.get("threat", "none")),
        centers=centers,
        enemy_cards=tuple(str(card_id) for card_id in raw.get("enemy_cards", [])),
    )


def collect_expert_actions(
    project_root: Path,
    catalog: CardCatalog,
    config: dict[str, Any],
) -> tuple[list[ExpertAction], int, int]:
    actions: list[ExpertAction] = []
    raw_count = 0
    rejected = 0
    minimum_confidence = float(config.get("minimum_selected_card_confidence", 0.45))
    for run_dir in list_demonstration_runs(project_root):
        tracked_hands: dict[int, list[str | None]] = {}
        for raw in _read_jsonl(run_dir / "actions.jsonl"):
            raw_count += 1
            slot = int(raw.get("slot_index", -1))
            hand_raw = list(raw.get("hand", []))
            deploy = list(raw.get("deploy_point", []))
            confidence = float(raw.get("selected_card_confidence", 0.0))
            battle_index = int(raw.get("battle_index", 0))
            if (
                raw.get("source") != "human_demonstration"
                or not bool(raw.get("human_action_ground_truth"))
                or not bool(raw.get("offline_verified"))
                or not 0 <= slot < 4
                or len(hand_raw) != 4
                or len(deploy) != 2
                or not 0.0 <= float(deploy[0]) <= 1.0
                or not 0.0 <= float(deploy[1]) <= 1.0
            ):
                rejected += 1
                continue

            # A selected or dragged card is briefly lifted out of its slot, which can
            # make that slot look empty in the screenshot.  Reconstruct only slots
            # that have not been invalidated by a previous play in the same battle.
            tracked = tracked_hands.setdefault(battle_index, [None, None, None, None])
            matches = list(raw.get("hand_matches", []))
            hand: list[str | None] = []
            for index, value in enumerate(hand_raw):
                card_id = str(value) if value else None
                match_confidence = 1.0
                if index < len(matches) and isinstance(matches[index], dict):
                    match_confidence = float(matches[index].get("confidence", 0.0))
                elif index == slot:
                    match_confidence = confidence
                reliable = (
                    card_id is not None
                    and catalog.get(card_id) is not None
                    and match_confidence >= minimum_confidence
                )
                hand.append(card_id if reliable else tracked[index])

            recognized_selected = str(raw.get("selected_card_id") or "") or None
            selected_source = "recognizer"
            if (
                recognized_selected is None
                or catalog.get(recognized_selected) is None
                or confidence < minimum_confidence
            ):
                recognized_selected = hand[slot]
                selected_source = "temporal_reconstruction"
            if recognized_selected is not None:
                hand[slot] = recognized_selected

            # Preserve the current observations for the next action, but invalidate
            # the played slot because Clash Royale will replace it with a new card.
            tracked_hands[battle_index] = list(hand)
            tracked_hands[battle_index][slot] = None
            selected_card = hand[slot]
            if selected_card is None or catalog.get(selected_card) is None:
                rejected += 1
                continue
            before_value = raw.get("before_frame")
            before_frame = (
                (run_dir / str(before_value)).resolve() if before_value else None
            )
            if before_frame is not None and not before_frame.is_file():
                before_frame = None
            actions.append(
                ExpertAction(
                    run_name=run_dir.name,
                    action_id=str(raw.get("action_id", f"{run_dir.name}-{raw_count}")),
                    battle_index=battle_index,
                    slot_index=slot,
                    deploy_point=(float(deploy[0]), float(deploy[1])),
                    hand=tuple(hand),
                    elixir=float(raw.get("elixir") or 5.0),
                    threats={
                        lane: _lane_threat(dict(raw.get("threats", {})), lane)
                        for lane in ("left", "right")
                    },
                    before_frame=before_frame,
                    selected_card_source=selected_source,
                )
            )
    return actions, raw_count, rejected


def audit_demonstrations(
    project_root: Path,
    catalog: CardCatalog,
    config: dict[str, Any],
) -> ImitationAudit:
    actions, raw_count, rejected = collect_expert_actions(project_root, catalog, config)
    selected_counts: dict[str, int] = {}
    for action in actions:
        selected = action.hand[action.slot_index]
        assert selected is not None
        selected_counts[selected] = selected_counts.get(selected, 0) + 1
    battles = len({action.group_id for action in actions})
    minimum_actions = int(config.get("minimum_actions_for_imitation", 100))
    minimum_battles = int(config.get("minimum_battles_for_imitation", 10))
    minimum_cards = int(config.get("minimum_distinct_cards_for_imitation", 12))
    reasons: list[str] = []
    if len(actions) < minimum_actions:
        reasons.append(f"有效示范动作 {len(actions)}/{minimum_actions}")
    if battles < minimum_battles:
        reasons.append(f"示范对局 {battles}/{minimum_battles}")
    if len(selected_counts) < minimum_cards:
        reasons.append(f"示范卡牌 {len(selected_counts)}/{minimum_cards}")
    return ImitationAudit(
        runs=len(list_demonstration_runs(project_root)),
        raw_actions=raw_count,
        usable_actions=len(actions),
        battles=battles,
        distinct_selected_cards=len(selected_counts),
        selected_card_counts=dict(sorted(selected_counts.items())),
        rejected_actions=rejected,
        automatically_reconstructed_actions=sum(
            action.selected_card_source == "temporal_reconstruction"
            for action in actions
        ),
        ready=not reasons,
        blocking_reasons=tuple(reasons),
    )


def _threat_features(threat: LaneThreat) -> list[float]:
    return [
        min(1.0, max(0.0, threat.score)),
        min(1.0, max(0.0, threat.unit_count / 5.0)),
        min(1.0, max(0.0, threat.proximity)),
        *[1.0 if threat.threat == value else 0.0 for value in THREAT_TYPES],
    ]


def global_features(elixir: float, threats: dict[str, LaneThreat]) -> list[float]:
    return [
        min(1.0, max(0.0, elixir / 10.0)),
        *_threat_features(threats["left"]),
        *_threat_features(threats["right"]),
    ]


def card_features(card: CardDefinition) -> list[float]:
    roles = set(card.roles)
    return [
        min(1.0, max(0.0, float(card.elixir or 3) / 10.0)),
        1.0 if card.kind == "troop" else 0.0,
        1.0 if card.kind == "building" else 0.0,
        1.0 if card.kind == "spell" else 0.0,
        *[1.0 if role in roles else 0.0 for role in ROLE_FEATURES],
    ]


def battlefield_features(image: Image.Image | Path | None) -> np.ndarray:
    """Build a compact, label-free battlefield embedding from the current frame."""

    if image is None:
        return np.zeros(BATTLEFIELD_FEATURE_COUNT, dtype=np.float32)
    try:
        if isinstance(image, Path):
            with Image.open(image) as loaded:
                rgb = np.asarray(loaded.convert("RGB"))
        else:
            rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        arena = rgb[
            int(0.06 * height) : int(0.80 * height),
            int(0.05 * width) : int(0.95 * width),
        ]
        if arena.size == 0:
            raise ValueError("empty arena crop")
        gray = cv2.cvtColor(arena, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 80, 160).astype(np.float32) / 255.0
        grid = cv2.resize(
            edges,
            BATTLEFIELD_GRID,
            interpolation=cv2.INTER_AREA,
        )
        return grid.reshape(-1).astype(np.float32)
    except (OSError, ValueError, cv2.error):
        return np.zeros(BATTLEFIELD_FEATURE_COUNT, dtype=np.float32)


def candidate_features(
    card: CardDefinition,
    elixir: float,
    threats: dict[str, LaneThreat],
    battlefield: np.ndarray | None = None,
    visual_weight: float = 0.0,
) -> list[float]:
    values = [*global_features(elixir, threats), *card_features(card)]
    if battlefield is not None:
        visual = np.asarray(battlefield, dtype=np.float32).reshape(-1)
        if len(visual) != BATTLEFIELD_FEATURE_COUNT:
            raise ValueError("战场特征维度不正确")
        values.extend((visual * float(visual_weight)).tolist())
    return values


def _split_actions(
    actions: list[ExpertAction],
    validation_fraction: float,
    seed: int,
) -> tuple[list[ExpertAction], list[ExpertAction]]:
    groups = sorted({action.group_id for action in actions})
    if len(groups) < 2:
        raise ValueError("至少需要两局手动示范才能划分训练集和验证集")
    ranked = sorted(
        groups,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )
    count = max(1, min(len(groups) - 1, math.ceil(len(groups) * validation_fraction)))
    validation_groups = set(ranked[:count])
    return (
        [action for action in actions if action.group_id not in validation_groups],
        [action for action in actions if action.group_id in validation_groups],
    )


def _knn_predict(
    training_x: np.ndarray,
    training_y: np.ndarray,
    sample: np.ndarray,
    neighbors: int,
) -> float:
    distances = np.sum((training_x - sample.reshape(1, -1)) ** 2, axis=1)
    count = min(max(1, neighbors), len(distances))
    indices = np.argpartition(distances, count - 1)[:count]
    weights = 1.0 / (np.sqrt(distances[indices]) + 0.05)
    return float(np.sum(training_y[indices] * weights) / np.sum(weights))


def _action_visuals(actions: list[ExpertAction]) -> dict[str, np.ndarray]:
    return {
        action.action_id: battlefield_features(action.before_frame)
        for action in actions
    }


def _build_training_arrays(
    actions: list[ExpertAction],
    catalog: CardCatalog,
    visuals: dict[str, np.ndarray],
    visual_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selector_x: list[list[float]] = []
    selector_y: list[float] = []
    deploy_x: list[list[float]] = []
    deploy_y: list[list[float]] = []
    for action in actions:
        visual = visuals[action.action_id]
        for slot, card_id in enumerate(action.hand):
            card = catalog.get(card_id) if card_id else None
            if card is None:
                continue
            selector_x.append(
                candidate_features(
                    card,
                    action.elixir,
                    action.threats,
                    visual,
                    visual_weight,
                )
            )
            selector_y.append(1.0 if slot == action.slot_index else 0.0)
        selected_id = action.hand[action.slot_index]
        selected_card = catalog.get(selected_id) if selected_id else None
        if selected_card is not None:
            deploy_x.append(
                candidate_features(
                    selected_card,
                    action.elixir,
                    action.threats,
                    visual,
                    visual_weight,
                )
            )
            deploy_y.append([action.deploy_point[0], action.deploy_point[1]])
    if not selector_x or not deploy_x:
        raise ValueError("示范动作缺少可识别的手牌，无法训练")
    return (
        np.asarray(selector_x, dtype=np.float32),
        np.asarray(selector_y, dtype=np.float32),
        np.asarray(deploy_x, dtype=np.float32),
        np.asarray(deploy_y, dtype=np.float32),
    )


def _evaluate_group_cross_validation(
    actions: list[ExpertAction],
    catalog: CardCatalog,
    visuals: dict[str, np.ndarray],
    neighbors: int,
    visual_weight: float,
) -> dict[str, float]:
    groups = sorted({action.group_id for action in actions})
    if len(groups) < 2:
        raise ValueError("至少需要两局手动示范才能交叉验证")
    correct = 0
    top_two = 0
    role_agreements = 0
    random_baselines: list[float] = []
    x_errors: list[float] = []
    y_errors: list[float] = []
    for validation_group in groups:
        training = [
            action for action in actions if action.group_id != validation_group
        ]
        validation = [
            action for action in actions if action.group_id == validation_group
        ]
        sx, sy, dx, dy = _build_training_arrays(
            training, catalog, visuals, visual_weight
        )
        for action in validation:
            visual = visuals[action.action_id]
            scored: list[tuple[float, int, CardDefinition]] = []
            for slot, card_id in enumerate(action.hand):
                card = catalog.get(card_id) if card_id else None
                if card is None:
                    continue
                feature = np.asarray(
                    candidate_features(
                        card,
                        action.elixir,
                        action.threats,
                        visual,
                        visual_weight,
                    ),
                    dtype=np.float32,
                )
                scored.append(
                    (_knn_predict(sx, sy, feature, neighbors), slot, card)
                )
            if not scored:
                continue
            scored.sort(key=lambda value: value[0], reverse=True)
            _score, predicted_slot, predicted_card = scored[0]
            correct += int(predicted_slot == action.slot_index)
            top_two += int(
                any(slot == action.slot_index for _value, slot, _card in scored[:2])
            )
            selected_id = action.hand[action.slot_index]
            selected_card = catalog.get(selected_id) if selected_id else None
            if selected_card is None:
                continue
            selected_roles = set(selected_card.roles).intersection(STRATEGIC_ROLES)
            predicted_roles = set(predicted_card.roles).intersection(STRATEGIC_ROLES)
            role_agreements += int(bool(selected_roles.intersection(predicted_roles)))
            random_baselines.append(1.0 / len(scored))

            # Evaluate deployment independently with the actual expert-selected card.
            chosen_feature = np.asarray(
                candidate_features(
                    selected_card,
                    action.elixir,
                    action.threats,
                    visual,
                    visual_weight,
                ),
                dtype=np.float32,
            )
            predicted_x = _knn_predict(dx, dy[:, 0], chosen_feature, neighbors)
            predicted_y = _knn_predict(dx, dy[:, 1], chosen_feature, neighbors)
            x_errors.append(abs(predicted_x - action.deploy_point[0]))
            y_errors.append(abs(predicted_y - action.deploy_point[1]))

    validated = len(y_errors)
    accuracy = correct / max(1, validated)
    random_accuracy = float(np.mean(random_baselines)) if random_baselines else 1.0
    x_mae = float(np.mean(x_errors)) if x_errors else 1.0
    y_mae = float(np.mean(y_errors)) if y_errors else 1.0
    return {
        "selection_accuracy": round(accuracy, 6),
        "selection_top2_accuracy": round(top_two / max(1, validated), 6),
        "selection_random_baseline": round(random_accuracy, 6),
        "selection_lift": round(accuracy - random_accuracy, 6),
        "role_agreement": round(role_agreements / max(1, validated), 6),
        "deploy_x_mae": round(x_mae, 6),
        "deploy_y_mae": round(y_mae, 6),
        "deploy_mae": round((x_mae + y_mae) / 2.0, 6),
        "validation_actions": validated,
        "validation_battles": len(groups),
    }


class ImitationRegistry:
    def __init__(self, project_root: Path):
        self.root = project_root.resolve() / "models" / "imitation"
        self.path = self.root / "registry.json"

    def load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"schema_version": 1, "champion": None, "candidates": []}
        return json.loads(self.path.read_text(encoding="utf-8"))

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
        relative = Path("candidates") / f"imitation_{version}.npz"
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, target)
        accuracy = float(metrics["selection_accuracy"])
        deploy_mae = float(metrics["deploy_mae"])
        top_two = float(metrics.get("selection_top2_accuracy", accuracy))
        role_agreement = float(metrics.get("role_agreement", accuracy))
        deploy_y_mae = float(metrics.get("deploy_y_mae", deploy_mae))
        lift = float(metrics.get("selection_lift", accuracy))
        score = accuracy + 0.2 * top_two + 0.1 * role_agreement - deploy_y_mae
        common_reasons: list[str] = []
        if int(metrics["validation_actions"]) < int(
            config.get("minimum_validation_actions", 20)
        ):
            common_reasons.append("验证动作数量不足")
        full_reasons = list(common_reasons)
        if accuracy < float(config.get("minimum_selection_accuracy", 0.55)):
            full_reasons.append("完整模式选牌准确率未达标")
        if deploy_mae > float(config.get("maximum_deploy_mae", 0.12)):
            full_reasons.append("完整模式落点误差未达标")

        assisted_reasons = list(common_reasons)
        if lift < float(config.get("minimum_assisted_selection_lift", 0.10)):
            assisted_reasons.append("选牌相对随机基线的提升不足")
        if top_two < float(config.get("minimum_assisted_top2_accuracy", 0.65)):
            assisted_reasons.append("前两名选牌覆盖率未达标")
        if role_agreement < float(
            config.get("minimum_assisted_role_agreement", 0.55)
        ):
            assisted_reasons.append("策略角色一致率未达标")
        if deploy_y_mae > float(
            config.get("maximum_assisted_deploy_y_mae", 0.12)
        ):
            assisted_reasons.append("纵向落点误差未达标")

        if not full_reasons:
            maturity = "full"
            reasons: list[str] = []
            qualification_notes: list[str] = []
            influence_scale = 1.0
        elif not assisted_reasons:
            maturity = "assisted"
            reasons = []
            qualification_notes = full_reasons
            influence_scale = float(
                config.get("assisted_policy_weight_scale", 0.30)
            )
        else:
            maturity = "rejected"
            reasons = assisted_reasons
            qualification_notes = full_reasons
            influence_scale = 0.0

        registry = self.load()
        champion = registry.get("champion")
        if isinstance(champion, dict) and maturity != "rejected":
            rank = {"rejected": 0, "assisted": 1, "full": 2}
            champion_maturity = str(champion.get("maturity", "full"))
            if rank.get(maturity, 0) < rank.get(champion_maturity, 2):
                reasons.append("成熟度低于当前示范模型")
            elif rank.get(maturity, 0) == rank.get(champion_maturity, 2) and score < float(
                champion.get("score", 0.0)
            ) + float(config.get("minimum_score_improvement", 0.01)):
                reasons.append("综合分未超过当前示范模型")
        candidate = {
            "version": version,
            "created_at_unix": time.time(),
            "model_path": relative.as_posix(),
            "metrics": metrics,
            "score": round(score, 6),
            "manifest": manifest,
            "maturity": maturity,
            "influence_scale": round(influence_scale, 6),
            "promoted": maturity != "rejected" and not reasons,
            "rejection_reasons": reasons,
            "qualification_notes": qualification_notes,
        }
        if candidate["promoted"]:
            registry["champion"] = dict(candidate)
        registry.setdefault("candidates", []).append(candidate)
        _write_json(self.path, registry)
        return candidate


def train_imitation_policy(
    project_root: Path,
    catalog: CardCatalog,
    config: dict[str, Any],
) -> dict[str, Any]:
    audit = audit_demonstrations(project_root, catalog, config)
    if not audit.ready:
        raise ValueError("示范数据尚未达标：" + "；".join(audit.blocking_reasons))
    actions, _raw, _rejected = collect_expert_actions(project_root, catalog, config)
    neighbors = int(config.get("neighbors", 15))
    visual_weight = float(config.get("battlefield_visual_weight", 0.10))
    visuals = _action_visuals(actions)
    metrics = _evaluate_group_cross_validation(
        actions,
        catalog,
        visuals,
        neighbors,
        visual_weight,
    )
    sx, sy, dx, dy = _build_training_arrays(
        actions, catalog, visuals, visual_weight
    )
    temporary_dir = project_root / "models" / "imitation" / "temporary"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    temporary_model = temporary_dir / "candidate.npz"
    np.savez_compressed(
        temporary_model,
        selector_x=sx,
        selector_y=sy,
        deploy_x=dx,
        deploy_y=dy,
        neighbors=np.asarray([neighbors], dtype=np.int32),
        visual_weight=np.asarray([visual_weight], dtype=np.float32),
        visual_feature_count=np.asarray([BATTLEFIELD_FEATURE_COUNT], dtype=np.int32),
    )
    manifest = {
        "human_train_actions": len(actions),
        "human_validation_actions": int(metrics["validation_actions"]),
        "train_battles": sorted({action.group_id for action in actions}),
        "validation_battles": sorted({action.group_id for action in actions}),
        "validation_method": "leave_one_battle_out",
        "automatically_reconstructed_actions": audit.automatically_reconstructed_actions,
        "features": (
            "battlefield_edges_elixir_lane_threat_and_card_roles_"
            "without_card_identity"
        ),
        "validation_is_human_only": True,
        "final_model_refit_on_all_human_actions": True,
    }
    candidate = ImitationRegistry(project_root).register(
        temporary_model, metrics, config, manifest
    )
    return {"audit": audit.to_dict(), "metrics": metrics, "candidate": candidate}


class ImitationPolicyModel:
    def __init__(self, project_root: Path):
        self.registry = ImitationRegistry(project_root)
        self.champion = self.registry.champion()
        self.available = False
        self.influence_scale = 0.0
        self.visual_weight = 0.0
        self.visual_feature_count = 0
        self._cached_image_id: int | None = None
        self._cached_battlefield: np.ndarray | None = None
        path = self.registry.champion_path()
        if path is None:
            return
        try:
            with np.load(path) as data:
                self.selector_x = data["selector_x"].copy()
                self.selector_y = data["selector_y"].copy()
                self.deploy_x = data["deploy_x"].copy()
                self.deploy_y = data["deploy_y"].copy()
                self.neighbors = int(data["neighbors"][0])
                if "visual_weight" in data:
                    self.visual_weight = float(data["visual_weight"][0])
                if "visual_feature_count" in data:
                    self.visual_feature_count = int(data["visual_feature_count"][0])
            self.influence_scale = float(
                (self.champion or {}).get("influence_scale", 1.0)
            )
            self.available = bool(len(self.selector_x) and len(self.deploy_x))
        except (OSError, ValueError, KeyError):
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
        """Cache one label-free battlefield embedding for all card comparisons."""

        self._battlefield(image)

    def _feature(
        self,
        card: CardDefinition,
        elixir: float,
        threats: dict[str, LaneThreat],
        image: Image.Image | None,
    ) -> np.ndarray:
        return np.asarray(
            candidate_features(
                card,
                elixir,
                threats,
                self._battlefield(image),
                self.visual_weight,
            ),
            dtype=np.float32,
        )

    def card_score(
        self,
        card: CardDefinition,
        elixir: float,
        threats: dict[str, LaneThreat],
        image: Image.Image | None = None,
    ) -> float:
        if not self.available:
            return 0.0
        feature = self._feature(card, elixir, threats, image)
        return _knn_predict(
            self.selector_x, self.selector_y, feature, self.neighbors
        )

    def deploy_point(
        self,
        card: CardDefinition,
        elixir: float,
        threats: dict[str, LaneThreat],
        image: Image.Image | None = None,
    ) -> list[float] | None:
        if not self.available:
            return None
        feature = self._feature(card, elixir, threats, image)
        return [
            _knn_predict(self.deploy_x, self.deploy_y[:, 0], feature, self.neighbors),
            _knn_predict(self.deploy_x, self.deploy_y[:, 1], feature, self.neighbors),
        ]
