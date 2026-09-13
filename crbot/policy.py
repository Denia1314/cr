from __future__ import annotations

import copy
import random
import time
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from PIL import Image

from .battle_perception import HandCardMatch, LaneThreat, UniversalHandRecognizer, detect_lane_threats
from .cards import CardCatalog, CardDefinition
from .learned_perception import LearnedBattlefieldDetector
from .imitation import ImitationPolicyModel
from .replay_learning import ReplayPolicyModel
from .runtime_model import normalize_runtime_model
from .tactics import card_tactics
from .temporal import HandHistory
from .vision import estimate_elixir, motion_score


@dataclass(frozen=True)
class BattleDecision:
    slot_index: int
    card_point: list[float]
    deploy_point: list[float]
    lane: str
    reason: str
    elixir: float
    elixir_source: str
    left_motion: float
    right_motion: float
    card_id: str | None = None
    card_name: str = ""
    card_cost: int | None = None
    hand: tuple[str | None, ...] = ()
    left_threat: float = 0.0
    right_threat: float = 0.0
    left_threat_type: str = "none"
    right_threat_type: str = "none"
    left_threat_proximity: float = 0.0
    right_threat_proximity: float = 0.0
    left_threat_approach_rate: float = 0.0
    right_threat_approach_rate: float = 0.0
    left_unit_count: int = 0
    right_unit_count: int = 0
    battle_elapsed_s: float = 0.0
    threat_type: str = "none"
    threat_proximity: float = 0.0
    threat_approach_rate: float = 0.0
    enemy_cards: tuple[str, ...] = ()
    left_threat_unit_layers: tuple[str, ...] = ()
    right_threat_unit_layers: tuple[str, ...] = ()
    threat_unit_layers: tuple[str, ...] = ()
    left_threat_layer_confidence: float = 0.0
    right_threat_layer_confidence: float = 0.0
    threat_layer_confidence: float = 0.0
    card_attack_targets: tuple[str, ...] = ()
    card_targeting_source: str = "unknown"
    imitation_used: bool = False
    replay_learning_used: bool = False
    card_formation_role: str = ""
    formation_phase: str = ""
    desired_formation_role: str = ""
    action_id: str = ""
    action_status: str = "proposed"
    hand_confidence: float = 0.0
    hand_age_s: float = 0.0
    elixir_confidence: float = 0.0
    elixir_age_s: float = 0.0
    elixir_phase: str = "normal"
    defense_elixir_reserve: float = 0.0
    resource_allocation_reason: str = ""
    placement_role: str = ""
    placement_reason: str = ""
    placement_candidates: tuple[str, ...] = ()
    counterpush_investment: float = 0.0
    rejected_candidate_reasons: tuple[str, ...] = ()
    decision_engine: str = "legacy"
    plan_revision: int = 0
    plan_valid_until: float = 0.0
    knowledge_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["hand"] = list(self.hand)
        value["enemy_cards"] = list(self.enemy_cards)
        value["left_threat_unit_layers"] = list(self.left_threat_unit_layers)
        value["right_threat_unit_layers"] = list(self.right_threat_unit_layers)
        value["threat_unit_layers"] = list(self.threat_unit_layers)
        value["card_attack_targets"] = list(self.card_attack_targets)
        value["placement_candidates"] = list(self.placement_candidates)
        value["rejected_candidate_reasons"] = list(self.rejected_candidate_reasons)
        return value


@dataclass
class LaneFormation:
    """Short-lived memory of friendly ranks that can continue into a counterpush."""

    frontline_until: float = -1_000.0
    frontline_at: float = -1_000.0
    frontline_point: tuple[float, float] | None = None
    frontline_card: str | None = None
    frontline_source: str = ""
    backline_until: float = -1_000.0
    backline_at: float = -1_000.0
    backline_point: tuple[float, float] | None = None
    backline_card: str | None = None
    backline_source: str = ""
    # Appended after the legacy fields to keep positional construction of old
    # snapshots compatible.  A fresh deployment starts at full confidence;
    # confidence then decays even when no newer friendly-unit evidence exists.
    frontline_confidence: float = 1.0
    backline_confidence: float = 1.0

    @staticmethod
    def _confidence_at(
        confidence: float,
        observed_at: float,
        now: float,
        decay_s: float,
    ) -> float:
        age = max(0.0, float(now) - float(observed_at))
        return max(0.0, min(1.0, max(0.0, confidence) * max(0.0, 1.0 - age / max(0.1, decay_s))))

    def frontline_confidence_at(self, now: float, decay_s: float = 11.0) -> float:
        return self._confidence_at(self.frontline_confidence, self.frontline_at, now, decay_s)

    def backline_confidence_at(self, now: float, decay_s: float = 11.0) -> float:
        return self._confidence_at(self.backline_confidence, self.backline_at, now, decay_s)

    def has_frontline(self, now: float, min_confidence: float = 0.35, decay_s: float = 11.0) -> bool:
        return (
            self.frontline_point is not None
            and now <= self.frontline_until
            and self.frontline_confidence_at(now, decay_s) >= min_confidence
        )

    def has_backline(self, now: float, min_confidence: float = 0.35, decay_s: float = 11.0) -> bool:
        return (
            self.backline_point is not None
            and now <= self.backline_until
            and self.backline_confidence_at(now, decay_s) >= min_confidence
        )

    def source(self, now: float) -> str:
        sources: list[str] = []
        if self.has_frontline(now):
            sources.append(self.frontline_source)
        if self.has_backline(now):
            sources.append(self.backline_source)
        return "defense" if "defense" in sources else (sources[0] if sources else "")


class BattlePolicy:
    """Battle policy with a catalog-driven reactive mode and the original baseline mode."""

    def __init__(
        self,
        config: dict[str, Any],
        config_path: Path | None = None,
    ):
        self.vision = config["vision"]
        self.policy = config["policy"]
        self.timing = config["timing"]
        self.config_demonstration = dict(config.get("demonstration", {}))
        self.config_replay = dict(config.get("replay", {}))
        self.mode = str(self.policy.get("mode", "baseline"))
        self.runtime_model = normalize_runtime_model(
            self.policy.get("runtime_model", "m3_c3")
        )
        self.temporal_perception_enabled = bool(
            self.policy.get("temporal_perception_enabled", True)
        )
        self.random = random.Random(self.policy.get("seed"))
        self.virtual_elixir = float(self.policy.get("initial_elixir", 5.0))
        self.confirmed_virtual_elixir = self.virtual_elixir
        self.reserved_elixir = 0.0
        self.hand_history = HandHistory(
            max_age_s=float(self.policy.get("hand_max_age_s", 1.2)),
            min_confidence=float(self.policy.get("hand_min_confidence", 0.40)),
            unknown_grace_s=float(self.policy.get("hand_unknown_grace_s", 0.45)),
            empty_grace_s=float(self.policy.get("hand_empty_grace_s", 0.32)),
            confidence_decay_s=float(self.policy.get("hand_confidence_decay_s", 1.2)),
            stability_frames=int(self.policy.get("hand_stability_frames", 1)),
        )
        self._last_elixir_visual: float | None = None
        self._last_elixir_confidence = 0.0
        self._last_elixir_observed_at = -1_000.0
        self._last_elixir_estimate_value: float | None = None
        self._last_elixir_estimate_source = "timer"
        self._last_elixir_estimate_at = -1_000.0
        self._last_elixir_image: Image.Image | None = None
        self._elixir_phase = "normal"
        self._elixir_rate_multiplier = 1.0
        self.last_update = time.monotonic()
        self.battle_started_at = self.last_update
        self.next_action_at = self.last_update
        self.slot_cursor = self.random.randrange(4)
        self.last_push_lane = "right"
        self.last_push_at = -1_000.0
        self.last_push_roles: frozenset[str] = frozenset()
        self.push_support_count = 0
        self.counterpush_investment = {"left": 0.0, "right": 0.0}
        self.last_formation_rejections: tuple[str, ...] = ()
        self.last_defense_at = -1_000.0
        self.last_defense_lane = "left"
        self.last_defense_snapshots: dict[
            str, tuple[float, float, int, float]
        ] = {}
        self.last_deploy_point: tuple[float, float] | None = None
        self.last_deploy_at = -1_000.0
        self.action_sequence = 0
        self.formations = {
            "left": LaneFormation(),
            "right": LaneFormation(),
        }
        self._pending_action_id: str | None = None
        self._pending_action_snapshot: dict[str, Any] | None = None
        self._pending_action_slot = -1
        self._pending_action_card: str | None = None
        self._resolved_action_ids: dict[str, str] = {}
        self.catalog: CardCatalog | None = None
        self.hand_recognizer: UniversalHandRecognizer | None = None
        self.learned_detector: LearnedBattlefieldDetector | None = None
        self.imitation_model: ImitationPolicyModel | None = None
        self.replay_model: ReplayPolicyModel | None = None
        self.perception_error = ""
        self._observed_image: Image.Image | None = None
        self._observed_at = -1.0
        self._observed_threats: dict[str, LaneThreat] = {}
        self._threat_observed_at = -1.0
        self._last_hand_matches: list[HandCardMatch] = []
        self._last_hand_image: Image.Image | None = None
        self._last_hand_observed_at = -1.0
        if self.mode == "reactive_catalog":
            self._load_catalog(config, config_path)

    def _load_catalog(
        self,
        config: dict[str, Any],
        config_path: Path | None,
    ) -> None:
        catalog_value = str(config.get("dataset", {}).get("card_catalog", "data/cards.json"))
        catalog_path = Path(catalog_value)
        project_root = config_path.parent if config_path is not None else Path.cwd()
        if not catalog_path.is_absolute():
            catalog_path = project_root / catalog_path
        try:
            self.catalog = CardCatalog.load(catalog_path.resolve())
            self.hand_recognizer = UniversalHandRecognizer(self.catalog, self.vision)
            self.learned_detector = LearnedBattlefieldDetector(
                project_root,
                self.catalog,
                dict(config.get("training", {})),
            )
            self.imitation_model = ImitationPolicyModel(project_root)
            if bool(self.config_replay.get("allow_bot_training", False)):
                self.replay_model = ReplayPolicyModel(
                    project_root, self.config_replay
                )
            if not self.hand_recognizer.available:
                self.perception_error = "OpenCV 不可用"
            elif not self.hand_recognizer.templates:
                self.perception_error = "卡图模板为空"
        except (OSError, ValueError) as exc:
            self.perception_error = str(exc)

    @property
    def reactive_ready(self) -> bool:
        return bool(
            self.catalog is not None
            and self.hand_recognizer is not None
            and self.hand_recognizer.available
            and self.hand_recognizer.templates
        )

    def reset_battle(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self._observed_image = None
        self._observed_threats = {}
        self._threat_observed_at = -1.0
        self._last_hand_matches = []
        self._last_hand_image = None
        self._last_hand_observed_at = -1.0
        self.hand_history.reset()
        self.virtual_elixir = float(self.policy.get("initial_elixir", 5.0))
        self.confirmed_virtual_elixir = self.virtual_elixir
        self.reserved_elixir = 0.0
        self._last_elixir_visual = None
        self._last_elixir_confidence = 0.0
        self._last_elixir_observed_at = -1_000.0
        self._last_elixir_estimate_value = None
        self._last_elixir_estimate_source = "timer"
        self._last_elixir_estimate_at = -1_000.0
        self._last_elixir_image = None
        self._elixir_phase = "normal"
        self._elixir_rate_multiplier = 1.0
        self.last_update = now
        self.battle_started_at = now
        self.next_action_at = now + 1.0
        self.last_push_at = -1_000.0
        self.last_push_roles = frozenset()
        self.push_support_count = 0
        self.counterpush_investment = {"left": 0.0, "right": 0.0}
        self.last_formation_rejections = ()
        self.last_defense_at = -1_000.0
        self.last_defense_lane = "left"
        self.last_defense_snapshots = {}
        self.last_deploy_point = None
        self.last_deploy_at = -1_000.0
        self.action_sequence = 0
        self.formations = {
            "left": LaneFormation(),
            "right": LaneFormation(),
        }
        self._pending_action_id = None
        self._pending_action_snapshot = None
        self._pending_action_slot = -1
        self._pending_action_card = None
        self._resolved_action_ids = {}

    def snapshot_state(self) -> dict[str, Any]:
        """Capture only decision state that may be changed by a proposal.

        Catalogs, recognizers and model instances are intentionally excluded;
        copying them for every tap would be expensive and they are read-only
        during a decision.  The random generator state is included so a
        rejected tap does not silently consume a different strategy branch.
        """
        return {
            "virtual_elixir": self.virtual_elixir,
            "confirmed_virtual_elixir": self.confirmed_virtual_elixir,
            "reserved_elixir": self.reserved_elixir,
            "last_update": self.last_update,
            "next_action_at": self.next_action_at,
            "slot_cursor": self.slot_cursor,
            "random_state": self.random.getstate(),
            "last_push_lane": self.last_push_lane,
            "last_push_at": self.last_push_at,
            "last_push_roles": self.last_push_roles,
            "push_support_count": self.push_support_count,
            "counterpush_investment": copy.deepcopy(self.counterpush_investment),
            "last_formation_rejections": self.last_formation_rejections,
            "last_defense_at": self.last_defense_at,
            "last_defense_lane": self.last_defense_lane,
            "last_defense_snapshots": copy.deepcopy(self.last_defense_snapshots),
            "last_deploy_point": self.last_deploy_point,
            "last_deploy_at": self.last_deploy_at,
            "action_sequence": self.action_sequence,
            "formations": copy.deepcopy(self.formations),
            "observed_image": self._observed_image,
            "observed_at": self._observed_at,
            "observed_threats": copy.deepcopy(self._observed_threats),
            "threat_observed_at": self._threat_observed_at,
            "last_hand_matches": copy.deepcopy(self._last_hand_matches),
            "last_hand_image": self._last_hand_image,
            "last_hand_observed_at": self._last_hand_observed_at,
            "hand_history": copy.deepcopy(self.hand_history),
            "last_elixir_visual": self._last_elixir_visual,
            "last_elixir_confidence": self._last_elixir_confidence,
            "last_elixir_observed_at": self._last_elixir_observed_at,
            "last_elixir_estimate_value": self._last_elixir_estimate_value,
            "last_elixir_estimate_source": self._last_elixir_estimate_source,
            "last_elixir_estimate_at": self._last_elixir_estimate_at,
            "last_elixir_image": self._last_elixir_image,
            "elixir_phase": self._elixir_phase,
            "elixir_rate_multiplier": self._elixir_rate_multiplier,
        }

    def _restore_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.virtual_elixir = float(snapshot["virtual_elixir"])
        self.confirmed_virtual_elixir = float(snapshot["confirmed_virtual_elixir"])
        self.reserved_elixir = float(snapshot["reserved_elixir"])
        self.last_update = float(snapshot["last_update"])
        self.next_action_at = float(snapshot["next_action_at"])
        self.slot_cursor = int(snapshot["slot_cursor"])
        self.random.setstate(snapshot["random_state"])
        self.last_push_lane = str(snapshot["last_push_lane"])
        self.last_push_at = float(snapshot["last_push_at"])
        self.last_push_roles = frozenset(snapshot["last_push_roles"])
        self.push_support_count = int(snapshot["push_support_count"])
        self.counterpush_investment = copy.deepcopy(snapshot.get("counterpush_investment", {"left": 0.0, "right": 0.0}))
        self.last_formation_rejections = tuple(snapshot.get("last_formation_rejections", ()))
        self.last_defense_at = float(snapshot["last_defense_at"])
        self.last_defense_lane = str(snapshot["last_defense_lane"])
        self.last_defense_snapshots = copy.deepcopy(snapshot["last_defense_snapshots"])
        self.last_deploy_point = snapshot["last_deploy_point"]
        self.last_deploy_at = float(snapshot["last_deploy_at"])
        self.action_sequence = int(snapshot["action_sequence"])
        self.formations = copy.deepcopy(snapshot["formations"])
        self._observed_image = snapshot["observed_image"]
        self._observed_at = float(snapshot["observed_at"])
        self._observed_threats = copy.deepcopy(snapshot["observed_threats"])
        self._threat_observed_at = float(snapshot.get("threat_observed_at", -1.0))
        self._last_hand_matches = copy.deepcopy(snapshot.get("last_hand_matches", []))
        self._last_hand_image = snapshot.get("last_hand_image")
        self._last_hand_observed_at = float(snapshot.get("last_hand_observed_at", -1.0))
        self.hand_history = copy.deepcopy(snapshot.get("hand_history", self.hand_history))
        self._last_elixir_visual = snapshot.get("last_elixir_visual")
        self._last_elixir_confidence = float(snapshot.get("last_elixir_confidence", 0.0))
        self._last_elixir_observed_at = float(snapshot.get("last_elixir_observed_at", -1_000.0))
        self._last_elixir_estimate_value = snapshot.get("last_elixir_estimate_value")
        self._last_elixir_estimate_source = str(snapshot.get("last_elixir_estimate_source", "timer"))
        self._last_elixir_estimate_at = float(snapshot.get("last_elixir_estimate_at", -1_000.0))
        self._last_elixir_image = snapshot.get("last_elixir_image")
        self._elixir_phase = str(snapshot.get("elixir_phase", "normal"))
        self._elixir_rate_multiplier = float(snapshot.get("elixir_rate_multiplier", 1.0))

    def prepare_action(
        self,
        decision: BattleDecision,
        snapshot: dict[str, Any],
    ) -> BattleDecision:
        """Turn a mutable policy proposal into a reserved action."""
        if self._pending_action_id is not None:
            raise RuntimeError(f"动作尚未确认：{self._pending_action_id}")
        action_id = f"{self.policy.get('version', 'policy')}-{uuid.uuid4().hex[:16]}"
        self._pending_action_id = action_id
        self._pending_action_snapshot = snapshot
        self._pending_action_slot = int(decision.slot_index)
        self._pending_action_card = decision.card_id
        self.reserved_elixir = float(
            decision.card_cost
            if decision.card_cost is not None
            else self.policy.get("unknown_card_cost", 3)
        )
        return replace(decision, action_id=action_id, action_status="proposed")

    @property
    def pending_action_id(self) -> str | None:
        return self._pending_action_id

    def resolve_action(
        self,
        action_id: str,
        status: str,
        *,
        now: float | None = None,
    ) -> bool:
        """Commit or roll back one reservation exactly once.

        ``unknown`` rolls back all speculative state but inserts a short retry
        guard.  The caller must not resend the same action automatically.
        """
        action_id = str(action_id).strip()
        if status not in {"confirmed", "rejected", "unknown"}:
            raise ValueError(f"不能提交动作状态：{status}")
        if action_id in self._resolved_action_ids:
            return False
        if action_id != self._pending_action_id or self._pending_action_snapshot is None:
            raise ValueError(f"未知或已过期的动作：{action_id}")
        snapshot = self._pending_action_snapshot
        if status in {"rejected", "unknown"}:
            self._restore_snapshot(snapshot)
            if status == "unknown":
                current = time.monotonic() if now is None else float(now)
                guard = float(self.policy.get("unknown_action_retry_guard_s", 1.2))
                self.next_action_at = max(self.next_action_at, current + max(0.2, guard))
        else:
            self.confirmed_virtual_elixir = self.virtual_elixir
            self.reserved_elixir = 0.0
            self.hand_history.consume(
                getattr(self, "_pending_action_slot", -1),
                getattr(self, "_pending_action_card", None),
                time.monotonic() if now is None else float(now),
            )
        self._resolved_action_ids[action_id] = status
        self._pending_action_id = None
        self._pending_action_snapshot = None
        self._pending_action_slot = -1
        self._pending_action_card = None
        if len(self._resolved_action_ids) > 256:
            self._resolved_action_ids.pop(next(iter(self._resolved_action_ids)), None)
        return True

    def _elixir_phase_for(self, now: float) -> tuple[str, float]:
        """Return a conservative recovery-rate phase from elapsed battle time.

        The visual meter remains authoritative when it is fresh.  The phase is
        only used for timer fallback, and can be disabled by setting the
        threshold to a non-positive value in a legacy configuration.
        """
        elapsed = max(0.0, float(now) - self.battle_started_at)
        threshold = float(self.policy.get("double_elixir_after_s", 120.0))
        multiplier = float(self.policy.get("double_elixir_multiplier", 2.0))
        if threshold > 0.0 and elapsed >= threshold and multiplier > 1.0:
            return "double", max(1.0, multiplier)
        return "normal", 1.0

    def _update_virtual_elixir(self, now: float) -> None:
        phase, multiplier = self._elixir_phase_for(now)
        seconds_per = max(0.1, float(self.policy.get("seconds_per_elixir", 2.8)))
        start = self.last_update
        end = max(start, float(now))
        threshold = self.battle_started_at + float(
            self.policy.get("double_elixir_after_s", 120.0)
        )
        if threshold > start and threshold < end and multiplier > 1.0:
            elapsed = (threshold - start) + (end - threshold) * multiplier
        else:
            _prior_phase, prior_multiplier = self._elixir_phase_for(start)
            elapsed = (end - start) * prior_multiplier
        self.virtual_elixir = min(
            10.0,
            self.virtual_elixir + elapsed / seconds_per,
        )
        self._elixir_phase = phase
        self._elixir_rate_multiplier = multiplier
        self.last_update = max(self.last_update, float(now))

    def _estimate_elixir(
        self,
        current: Image.Image,
        now: float | None = None,
    ) -> tuple[float, str]:
        now = self.last_update if now is None else float(now)
        if (
            self._last_elixir_image is current
            and abs(self._last_elixir_estimate_at - now) <= 1e-9
            and self._last_elixir_estimate_value is not None
        ):
            return self._last_elixir_estimate_value, self._last_elixir_estimate_source
        visual_elixir, confidence = estimate_elixir(current, self.vision["elixir_roi"])
        minimum_confidence = float(self.policy.get("elixir_visual_min_confidence", 0.12))
        if visual_elixir is not None and confidence >= minimum_confidence:
            self._last_elixir_visual = float(visual_elixir)
            self._last_elixir_confidence = float(confidence)
            self._last_elixir_observed_at = now
            # Lower-confidence visual readings are blended less aggressively;
            # this prevents a purple animation or partial crop from granting
            # spendable elixir that was not actually observed.
            blend = 0.45 if confidence >= 0.22 else 0.20
            fused = (1.0 - blend) * self.virtual_elixir + blend * float(visual_elixir)
            self.virtual_elixir = min(10.0, max(0.0, fused))
            trusted_confidence = max(
                minimum_confidence,
                float(self.policy.get("elixir_visual_trusted_confidence", 0.22)),
            )
            if confidence >= trusted_confidence:
                value = float(visual_elixir)
                source = "vision"
            else:
                # A barely visible meter can be a transient animation.  Keep
                # the estimate at the conservative fused value instead of
                # allowing one noisy frame to unlock an expensive card.
                value = min(float(visual_elixir), self.virtual_elixir)
                source = "vision_low_confidence"
            self._last_elixir_estimate_value = value
            self._last_elixir_estimate_source = source
            self._last_elixir_estimate_at = now
            self._last_elixir_image = current
            return value, source
        stale_after = max(0.1, float(self.policy.get("elixir_visual_stale_s", 2.0)))
        if self._last_elixir_visual is not None and now - self._last_elixir_observed_at <= stale_after:
            # The meter is still useful as evidence, but timer accrual remains
            # the value used for spending so stale pixels cannot over-credit.
            self._last_elixir_confidence *= 0.85
        else:
            self._last_elixir_confidence = 0.0
        value = self.virtual_elixir
        self._last_elixir_estimate_value = value
        self._last_elixir_estimate_source = "timer"
        self._last_elixir_estimate_at = now
        self._last_elixir_image = current
        return value, "timer"

    def _point_for_lane(self, lane: str, defending: bool) -> list[float]:
        x_range = self.policy["left_x"] if lane == "left" else self.policy["right_x"]
        y_range = self.policy["defense_y"] if defending else self.policy["push_y"]
        return [
            self.random.uniform(float(x_range[0]), float(x_range[1])),
            self.random.uniform(float(y_range[0]), float(y_range[1])),
        ]

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return min(upper, max(lower, value))

    def _remember_formation(
        self,
        lane: str,
        card: CardDefinition,
        point: list[float],
        now: float,
        source: str,
    ) -> None:
        """Remember likely surviving ranks without pretending to track exact units."""
        tactics = card_tactics(card)
        if card.kind != "troop":
            return
        formation = self.formations[lane]
        cost = self._effective_cost(
            card, float(self.policy.get("unknown_card_cost", 3))
        )
        position = (float(point[0]), float(point[1]))
        if tactics.is_frontline:
            lifetime = 7.5 + 0.85 * cost
            if "tank" in card.roles:
                lifetime += 3.0
            if source == "defense":
                lifetime = max(
                    lifetime,
                    float(self.policy.get("counterpush_window_s", 10.0)),
                )
            formation.frontline_at = now
            formation.frontline_until = now + lifetime
            formation.frontline_point = position
            formation.frontline_card = card.card_id
            formation.frontline_source = source
            formation.frontline_confidence = float(
                self.policy.get("formation_initial_confidence", 0.86)
            )
        if tactics.is_backline:
            lifetime = 7.0 + 0.75 * cost
            if source == "defense":
                lifetime = max(
                    lifetime,
                    float(self.policy.get("counterpush_window_s", 10.0)),
                )
            formation.backline_at = now
            formation.backline_until = now + lifetime
            formation.backline_point = position
            formation.backline_card = card.card_id
            formation.backline_source = source
            formation.backline_confidence = float(
                self.policy.get("formation_initial_confidence", 0.82)
            )

    def _formation_card_score(
        self,
        card: CardDefinition,
        desired_role: str,
    ) -> float:
        tactics = card_tactics(card)
        roles = set(card.roles)
        cost = self._effective_cost(
            card, float(self.policy.get("unknown_card_cost", 3))
        )
        if desired_role == "frontline":
            if card.kind == "spell":
                return 10.0 if tactics.is_win_condition else -30.0
            if card.kind == "building":
                return -30.0
            score = 2.6 * tactics.frontline_score - 0.8 * tactics.backline_score
            score += 5.0 if tactics.is_win_condition else 0.0
            score += 1.2 if "tank_killer" in roles else 0.0
            score += 0.8 if "swarm" in roles else 0.0
        else:
            if card.kind != "troop":
                return -30.0
            score = 2.7 * tactics.backline_score - 0.55 * tactics.frontline_score
            score += 2.0 if "support" in roles else 0.0
            score += 1.2 if "splash" in roles else 0.0
            score -= 3.0 if tactics.is_win_condition else 0.0
        score += max(-1.0, 1.3 - 0.22 * cost)
        return score

    @staticmethod
    def _fits_formation_role(card: CardDefinition, desired_role: str) -> bool:
        tactics = card_tactics(card)
        if desired_role == "frontline":
            if card.kind == "spell":
                return tactics.is_win_condition
            return card.kind == "troop" and tactics.formation_role != "backline"
        return card.kind == "troop" and tactics.formation_role != "frontline"

    def _formation_lane(self, desired_rank: str, now: float) -> str | None:
        candidates: list[tuple[float, int, str]] = []
        for lane, formation in self.formations.items():
            if desired_rank == "backline" and self._formation_has_frontline(formation, now):
                until = formation.frontline_until
            elif desired_rank == "frontline" and self._formation_has_backline(formation, now):
                until = formation.backline_until
            else:
                continue
            defense_priority = 1 if self._formation_source(formation, now) == "defense" else 0
            candidates.append((until, defense_priority, lane))
        if not candidates:
            return None
        return max(candidates, key=lambda value: (value[1], value[0]))[2]

    def _defense_point(
        self,
        lane: str,
        card: CardDefinition,
        threat: LaneThreat,
    ) -> list[float]:
        """Choose a role-aware interception point from the observed push."""
        point, _, _, _ = self._defense_placement(lane, card, threat)
        return point

    def _defense_placement(
        self,
        lane: str,
        card: CardDefinition,
        threat: LaneThreat,
        *,
        observation_age_s: float = 0.0,
    ) -> tuple[list[float], str, str, tuple[str, ...]]:
        """Return a legal role-aware point plus compact placement evidence."""
        roles = set(card.roles)
        tactics = card_tactics(card)
        lane_range = self.policy["left_x"] if lane == "left" else self.policy["right_x"]
        lane_axis = (float(lane_range[0]) + float(lane_range[1])) / 2.0
        if threat.centers:
            front = max(threat.centers, key=lambda point: point[1])
            target_x = float(front[0])
        else:
            target_x = lane_axis
        target_x = self._clamp(
            target_x,
            float(lane_range[0]) - 0.04,
            float(lane_range[1]) + 0.04,
        )

        if card.kind == "spell":
            if threat.centers:
                if not bool(
                    self.policy.get("role_aware_placement_enabled", True)
                ):
                    target_x = sum(point[0] for point in threat.centers) / len(
                        threat.centers
                    )
                    target_y = sum(point[1] for point in threat.centers) / len(
                        threat.centers
                    )
                    return (
                        [
                            self._clamp(target_x, 0.12, 0.88),
                            self._clamp(target_y, 0.22, 0.75),
                        ],
                        "legacy_spell_center",
                        "pre_m3_c2_average_center",
                        (),
                    )
                max_age = float(self.policy.get("placement_prediction_max_age_s", 0.8))
                delay = float(self.policy.get("spell_prediction_delay_s", 0.10))
                prediction = max(0.0, threat.approach_rate) * delay if observation_age_s <= max_age else 0.0
                predicted = tuple(
                    (float(x), self._clamp(float(y) + prediction, 0.22, 0.75))
                    for x, y in threat.centers
                )
                radius = float(self.policy.get("spell_cluster_radius", 0.075))
                candidates = list(predicted)
                for index, first in enumerate(predicted):
                    for second in predicted[index + 1 :]:
                        candidates.append(((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0))

                def coverage(point: tuple[float, float]) -> tuple[int, float, float]:
                    distances = [((point[0] - unit[0]) ** 2 + (point[1] - unit[1]) ** 2) ** 0.5 for unit in predicted]
                    covered = sum(distance <= radius for distance in distances)
                    return covered, -sum(min(distance, radius * 2.0) for distance in distances), point[1]

                target_x, target_y = max(candidates, key=coverage)
                covered = coverage((target_x, target_y))[0]
                freshness = "predicted" if prediction > 0.0 else ("stale_no_prediction" if observation_age_s > max_age else "stationary")
                evidence = tuple(f"{x:.3f},{y:.3f}:cover={coverage((x, y))[0]}" for x, y in candidates[:8])
                reason = f"cluster_cover={covered}/{len(predicted)};{freshness}"
            else:
                target_y = threat.proximity
                evidence = ()
                reason = "no_centers_lane_fallback"
            return ([
                self._clamp(target_x, 0.12, 0.88),
                self._clamp(target_y, 0.22, 0.75),
            ], "spell_cluster", reason, evidence)

        # Meet an advancing unit between the bridge and our tower.  Earlier
        # observations produce a forward setup; fast motion pulls the point a
        # little deeper so the defender does not walk past the attacker.
        intercept_y = 0.53 + max(0.0, threat.proximity - 0.30) * 0.28
        intercept_y += min(0.025, threat.approach_rate * 0.7)
        if card.kind == "building":
            target_x = 0.445 if lane == "left" else 0.555
            intercept_y = 0.56 + max(0.0, threat.proximity - 0.34) * 0.16
            bounds = self.policy.get("building_defense_y", [0.55, 0.63])
            placement_role = "building_pull"
            placement_reason = "inner_lane_pull_with_legal_bounds"
        elif tactics.is_backline and not tactics.is_frontline:
            inner_axis = 0.40 if lane == "left" else 0.60
            target_x = 0.35 * target_x + 0.65 * inner_axis
            intercept_y += 0.045
            bounds = self.policy.get("ranged_defense_y", [0.56, 0.69])
            placement_role = "ranged_backline"
            placement_reason = "protected_range_behind_intercept"
        else:
            bounds = self.policy.get("intercept_y", [0.52, 0.67])
            if intercept_y >= 0.57:
                inner_axis = 0.40 if lane == "left" else 0.60
                target_x = 0.55 * target_x + 0.45 * inner_axis
            placement_role = "melee_intercept"
            placement_reason = "intercept_path_scaled_by_proximity_and_speed"
        return ([
            self._clamp(target_x, 0.14, 0.86),
            self._clamp(intercept_y, float(bounds[0]), float(bounds[1])),
        ], placement_role, placement_reason, ())

    def _push_point(
        self,
        lane: str,
        card: CardDefinition,
        *,
        desired_role: str,
        staged_backline: bool,
        elixir: float,
        threats: dict[str, LaneThreat],
        now: float,
    ) -> list[float]:
        x_range = self.policy["left_x"] if lane == "left" else self.policy["right_x"]
        lane_axis = (float(x_range[0]) + float(x_range[1])) / 2.0
        roles = set(card.roles)
        tactics = card_tactics(card)
        formation = self.formations[lane]
        cost = self._effective_cost(
            card, float(self.policy.get("unknown_card_cost", 3))
        )
        # X is continuous and responds to lane pressure, card role and action
        # phase. This avoids repeatedly tapping a small set of anchor points.
        pressure_delta = threats["right"].score - threats["left"].score
        pressure_bias = self._clamp(pressure_delta * 0.035, -0.025, 0.025)
        role_bias = 0.0
        if tactics.is_backline and not tactics.is_frontline:
            role_bias = 0.075 if lane == "left" else -0.075
        cadence = ((self.action_sequence % 5) - 2) * 0.006
        x = lane_axis + pressure_bias + role_bias + cadence

        # A rear rank follows the estimated position of the remembered front
        # rank. A new front rank protecting surviving defenders is placed in
        # front of them. Coordinates decrease while units advance upward.
        if desired_role == "backline" and self._formation_has_frontline(formation, now):
            assert formation.frontline_point is not None
            age = max(0.0, now - formation.frontline_at)
            estimated_front_y = formation.frontline_point[1] - min(0.10, age * 0.010)
            x = 0.65 * formation.frontline_point[0] + 0.35 * x
            y = estimated_front_y + 0.095 + (cost - 4.0) * 0.003
        elif desired_role == "frontline" and self._formation_has_backline(formation, now):
            assert formation.backline_point is not None
            age = max(0.0, now - formation.backline_at)
            estimated_back_y = formation.backline_point[1] - min(0.08, age * 0.008)
            x = 0.70 * formation.backline_point[0] + 0.30 * x
            y = estimated_back_y - 0.105 + (cost - 4.0) * 0.003
        elif card.kind == "spell" and tactics.is_win_condition:
            x = 0.25 if lane == "left" else 0.75
            y = 0.235
        elif staged_backline:
            y = 0.72 + (cost - 4.0) * 0.004
        elif tactics.is_win_condition and "tank" not in roles:
            y = 0.505 + (10.0 - elixir) * 0.004
        elif tactics.is_frontline and "tank" in roles:
            y = 0.73 - self._clamp((elixir - cost) * 0.006, 0.0, 0.035)
        elif desired_role == "frontline":
            y = 0.55 + (cost - 4.0) * 0.006
        else:
            y = 0.71 + (cost - 4.0) * 0.005
        return [
            self._clamp(x, float(x_range[0]), float(x_range[1])),
            self._clamp(y, 0.22 if card.kind == "spell" else 0.49, 0.75),
        ]

    @staticmethod
    def _effective_cost(card: CardDefinition, unknown_cost: float) -> float:
        if card.elixir is None or card.elixir <= 0:
            return unknown_cost
        return float(card.elixir)

    def _defense_score(
        self,
        card: CardDefinition,
        threat: LaneThreat,
        *,
        multi_lane: bool = False,
    ) -> float:
        roles = set(card.roles)
        tactics = card_tactics(card)
        target_coverage = tactics.target_coverage(
            self._trusted_threat_layers(threat)
        )
        if target_coverage == 0.0:
            # This is a hard mechanics constraint, not a preference that a
            # learned score may override.  Unknown target mechanics remain
            # eligible and are represented by ``None`` instead.
            return -1_000.0
        cost = self._effective_cost(card, float(self.policy.get("unknown_card_cost", 3)))
        score = 0.0
        if target_coverage is not None:
            score += 2.5 * target_coverage
            score -= 1.5 * (1.0 - target_coverage)
        exact_counter = any(
            enemy_card in set(card.counters) for enemy_card in threat.enemy_cards
        )
        if exact_counter:
            score += 8.0
        if threat.threat == "swarm":
            score += 8.0 * tactics.splash_strength
            score += 6.0 if card.kind == "spell" else 0.0
            score += 2.0 if "ranged" in roles else 0.0
            score -= 2.0 if "tank_killer" in roles and "splash" not in roles else 0.0
            if tactics.splash_strength < 0.35:
                score -= 2.5
            if threat.score >= 0.65 and cost <= 1.5 and card.kind != "spell":
                score -= 2.0
        elif threat.threat == "heavy":
            score += 7.5 * tactics.heavy_counter_strength
            score -= 2.0 if "splash" in roles and "tank_killer" not in roles else 0.0
        else:
            score += 4.0 if "tank_killer" in roles else 0.0
            score += 3.5 if card.kind == "building" else 0.0
            score += 3.0 if "swarm" in roles else 0.0
            score += 2.0 if "cheap" in roles else 0.0
            score += 1.0 if "splash" in roles else 0.0
        if card.kind == "troop":
            score += 1.0
            # Direct defensive deployment is allowed for either rank. Front
            # ranks body-block heavy threats; rear ranks get a small reward
            # for staying alive and becoming the core of a counterpush.
            if threat.threat == "heavy":
                score += 0.45 * tactics.frontline_score
            elif threat.threat == "swarm":
                score += 0.30 * tactics.backline_score
        if "support" in roles:
            score += 0.75
        if card.kind == "spell" and not roles.intersection({"splash", "tank_killer"}):
            score -= 6.0
        if threat.threat == "heavy" and card.kind == "spell" and "tank_killer" not in roles:
            score -= 4.0
        if tactics.offensive_commitment >= 0.55 and not exact_counter:
            # Preserve offensive tanks and building-targeting win conditions.
            # Their large health pool alone does not make them an efficient
            # answer, especially when a second lane also needs elixir.
            score -= 4.0 + 2.0 * tactics.offensive_commitment
        if multi_lane:
            score -= max(0.0, cost - 3.0) * 0.55
        score += max(-1.5, 1.8 - 0.35 * cost)
        return score

    def _threat_urgency(self, threat: LaneThreat) -> float:
        """Combine pressure, arrival distance and motion for lane priority."""
        return (
            float(threat.score)
            + 0.35 * float(threat.proximity)
            + min(0.25, max(0.0, float(threat.approach_rate)) * 1.5)
            + min(0.15, max(0, int(threat.unit_count)) * 0.03)
        )

    def _minimum_defense_cost(
        self,
        candidates: list[tuple[HandCardMatch, CardDefinition]],
        threat: LaneThreat,
    ) -> float:
        unknown_cost = float(self.policy.get("unknown_card_cost", 3))
        minimum_score = float(self.policy.get("defense_min_card_score", 1.25))
        costs = [
            self._effective_cost(card, unknown_cost)
            for _, card in candidates
            if self._defense_score(card, threat, multi_lane=True) >= minimum_score
        ]
        return min(costs, default=0.0)

    def _trusted_threat_layers(self, threat: LaneThreat) -> tuple[str, ...]:
        minimum = float(self.policy.get("threat_layer_min_confidence", 0.70))
        if float(threat.layer_confidence) < minimum:
            return ()
        return tuple(
            dict.fromkeys(
                layer
                for layer in threat.unit_layers
                if layer in {"ground", "air"}
            )
        )

    def _push_score(self, card: CardDefinition, *, supporting: bool) -> float:
        roles = set(card.roles)
        cost = self._effective_cost(card, float(self.policy.get("unknown_card_cost", 3)))
        if supporting:
            score = 0.0
            score += 7.0 if "support" in roles else 0.0
            score += 4.0 if "ranged" in roles else 0.0
            score += 3.0 if "splash" in roles else 0.0
            score += 2.0 if "cheap" in roles else 0.0
            score += 1.5 if card.kind == "troop" else 0.0
            score -= 5.0 if "tank" in roles or "win_condition" in roles else 0.0
            score -= 8.0 if card.kind == "spell" else 0.0
        else:
            score = 0.0
            score += 9.0 if "win_condition" in roles else 0.0
            score += 6.0 if "tank" in roles else 0.0
            score += 3.0 if "building_target" in roles else 0.0
            score += 2.5 if card.kind == "troop" else 0.0
            score += 1.25 if roles.intersection({"ranged", "support", "spawner"}) else 0.0
            score += 0.75 if roles.intersection({"tank_killer", "swarm", "cheap"}) else 0.0
            score -= 5.0 if card.kind == "spell" or card.kind == "building" else 0.0
        score += max(-1.0, 1.5 - 0.25 * cost)
        return score

    def _recognized_affordable(
        self,
        matches: list[HandCardMatch],
        elixir: float,
    ) -> list[tuple[HandCardMatch, CardDefinition]]:
        if self.catalog is None:
            return []
        result: list[tuple[HandCardMatch, CardDefinition]] = []
        unknown_cost = float(self.policy.get("unknown_card_cost", 3))
        reserve = float(self.policy.get("defense_elixir_reserve", 0.15))
        for match in matches:
            if match.card_id is None:
                continue
            card = self.catalog.get(match.card_id)
            if card is None:
                continue
            if self._effective_cost(card, unknown_cost) <= elixir + reserve:
                result.append((match, card))
        return result

    def _stable_hand_matches(
        self,
        current: Image.Image,
        now: float,
    ) -> list[HandCardMatch]:
        """Read the hand once and expose only fresh, confidence-checked slots."""
        if self.hand_recognizer is None:
            return []
        try:
            observed = list(self.hand_recognizer.recognize(current))
        except Exception as exc:  # perception failure must fail closed
            self.perception_error = f"手牌识别失败：{exc}"
            return []
        self.hand_history.update(observed, now)
        self._last_hand_matches = list(observed)
        self._last_hand_image = current
        self._last_hand_observed_at = float(now)
        if not self.temporal_perception_enabled:
            return observed
        return self.hand_history.matches_for_decision(observed, now)

    @property
    def last_hand_matches(self) -> list[HandCardMatch]:
        return list(self._last_hand_matches)

    @property
    def last_hand_image(self) -> Image.Image | None:
        return self._last_hand_image

    @property
    def last_elixir_image(self) -> Image.Image | None:
        return self._last_elixir_image

    @property
    def last_elixir_visual(self) -> float | None:
        return self._last_elixir_visual

    @property
    def last_elixir_estimate_value(self) -> float | None:
        return self._last_elixir_estimate_value

    @property
    def last_elixir_estimate_source(self) -> str:
        return self._last_elixir_estimate_source

    @property
    def last_elixir_confidence(self) -> float:
        return float(self._last_elixir_confidence)

    def hand_metadata(self, now: float | None = None) -> dict[str, dict[str, float | int | str | None]]:
        return self.hand_history.metadata(self.last_update if now is None else float(now))

    def formation_metadata(self, now: float | None = None) -> dict[str, dict[str, Any]]:
        at = self.last_update if now is None else float(now)
        decay = float(self.policy.get("formation_confidence_decay_s", 11.0))
        return {
            lane: {
                "frontline_card": formation.frontline_card,
                "frontline_confidence": round(formation.frontline_confidence_at(at, decay), 4),
                "frontline_age_s": round(max(0.0, at - formation.frontline_at), 3)
                if formation.frontline_at > -999.0
                else -1.0,
                "backline_card": formation.backline_card,
                "backline_confidence": round(formation.backline_confidence_at(at, decay), 4),
                "backline_age_s": round(max(0.0, at - formation.backline_at), 3)
                if formation.backline_at > -999.0
                else -1.0,
                "source": self._formation_source(formation, at),
            }
            for lane, formation in sorted(self.formations.items())
        }

    def _formation_has_frontline(self, formation: LaneFormation, now: float) -> bool:
        return formation.has_frontline(
            now,
            float(self.policy.get("formation_min_confidence", 0.35)),
            float(self.policy.get("formation_confidence_decay_s", 11.0)),
        )

    def _formation_has_backline(self, formation: LaneFormation, now: float) -> bool:
        return formation.has_backline(
            now,
            float(self.policy.get("formation_min_confidence", 0.35)),
            float(self.policy.get("formation_confidence_decay_s", 11.0)),
        )

    def _formation_source(self, formation: LaneFormation, now: float) -> str:
        sources: list[str] = []
        if self._formation_has_frontline(formation, now):
            sources.append(formation.frontline_source)
        if self._formation_has_backline(formation, now):
            sources.append(formation.backline_source)
        return "defense" if "defense" in sources else (sources[0] if sources else "")

    def _counterpush_gate(
        self,
        lane: str,
        threats: dict[str, LaneThreat],
        current: Image.Image,
        previous: Image.Image | None,
    ) -> tuple[bool, tuple[str, ...]]:
        """Require current survival evidence and acceptable off-lane risk."""
        if not bool(self.policy.get("counterpush_evidence_enabled", False)):
            return True, ()
        reasons: list[str] = []
        other_lane = "right" if lane == "left" else "left"
        other_risk = threats[other_lane].score
        if other_risk >= float(self.policy.get("counterpush_other_lane_risk_max", 0.12)):
            reasons.append(f"other_lane_risk={other_risk:.3f}")

        detector = self.learned_detector
        supports_allies = bool(detector is not None and getattr(detector, "allies_observed", False))
        allies = getattr(detector, "observed_allies", []) if detector is not None else []
        min_confidence = float(self.policy.get("counterpush_ally_min_confidence", 0.55))
        ally_seen = any(
            ally.get("lane") == lane and float(ally.get("confidence", 0.0)) >= min_confidence
            for ally in allies
        )
        roi = self.vision["friendly_left_roi"] if lane == "left" else self.vision["friendly_right_roi"]
        lane_motion = motion_score(current, previous, roi)
        motion_seen = lane_motion >= float(self.policy.get("counterpush_motion_min", 0.018))
        if not ally_seen and not motion_seen:
            evidence = "ally_detector_no_match" if supports_allies else "no_fresh_motion_evidence"
            reasons.append(evidence)
        return not reasons, tuple(reasons)

    def _defense_response_due(self, threat: LaneThreat, now: float) -> bool:
        previous = self.last_defense_snapshots.get(threat.lane)
        if previous is None:
            return True
        previous_at, previous_score, previous_count, previous_proximity = previous
        cooldown = float(self.policy.get("defense_reaction_cooldown_s", 4.2))
        if now - previous_at >= cooldown:
            return True
        return bool(
            threat.unit_count > previous_count
            or threat.score
            >= previous_score
            + float(self.policy.get("defense_score_escalation", 0.18))
            or threat.proximity
            >= previous_proximity
            + float(self.policy.get("defense_proximity_escalation", 0.09))
        )

    def observe_replay_state(
        self, current: Image.Image, previous: Image.Image | None, *, now: float | None = None,
    ) -> dict[str, Any]:
        now = time.monotonic() if now is None else now
        self._update_virtual_elixir(now)
        threats = self._perceive_threats(current, previous, now)
        elixir, elixir_source = self._estimate_elixir(current, now)
        state: dict[str, Any] = {
            "schema": "action_observation_v1",
            "temporal_schema": "hand_elixir_formation_v1",
            "tactical_schema": "threat_layers_v1",
            "battle_elapsed_s": round(max(0.0, now - self.battle_started_at), 3),
            "elixir": round(float(elixir), 3),
            "elixir_source": elixir_source,
            "elixir_confidence": round(self._last_elixir_confidence, 4),
            "elixir_age_s": round(
                max(0.0, now - self._last_elixir_observed_at)
                if self._last_elixir_observed_at > -999.0
                else -1.0,
                3,
            ),
            "elixir_phase": self._elixir_phase,
            "formation_metadata": self.formation_metadata(now),
            "counterpush_investment": dict(self.counterpush_investment),
            "formation_rejections": list(self.last_formation_rejections),
            "allies_observed": bool(getattr(self.learned_detector, "allies_observed", False)),
            "observed_allies": list(getattr(self.learned_detector, "observed_allies", [])),
        }
        if self.mode == "reactive_catalog" and self.hand_recognizer is not None:
            matches = self._stable_hand_matches(current, now)
            state["hand"] = [match.card_id for match in matches]
            state["hand_metadata"] = self.hand_metadata(now)
            state["hand_confidence"] = round(
                max((match.confidence for match in matches), default=0.0), 4
            )
        else:
            state["hand"] = []
        for lane, threat in threats.items():
            state.update({lane + "_threat": threat.score,
                          lane + "_threat_proximity": threat.proximity,
                          lane + "_unit_count": threat.unit_count,
                          lane + "_threat_unit_layers": list(threat.unit_layers),
                          lane + "_layer_confidence": threat.layer_confidence})
        return state

    def _perceive_threats(
        self, current: Image.Image, previous: Image.Image | None, now: float,
    ) -> dict[str, LaneThreat]:
        if current is self._observed_image and now == self._observed_at:
            return self._observed_threats
        ignore_points = ()
        if self.last_deploy_point is not None and now - self.last_deploy_at <= float(
            self.policy.get("own_deploy_indicator_s", 2.8)
        ):
            ignore_points = (self.last_deploy_point,)
        frame_dt_s: float | None = None
        if self._threat_observed_at > -999.0:
            frame_dt_s = max(0.05, now - self._threat_observed_at)
        try:
            threats = detect_lane_threats(
                current,
                previous,
                ignore_points,
                frame_dt_s=frame_dt_s,
            )
        except TypeError:
            # Third-party/test detectors written against the M1 signature are
            # still valid; their output simply keeps the legacy rate units.
            threats = detect_lane_threats(current, previous, ignore_points)
        if self.learned_detector is not None:
            threats = self.learned_detector.detect(current, threats)
        self._observed_image, self._observed_threats = current, threats
        self._observed_at = now
        self._threat_observed_at = now
        return threats

    def _reactive_decide(
        self,
        current: Image.Image,
        previous: Image.Image | None,
        now: float,
    ) -> BattleDecision | None:
        if not self.reactive_ready or self.hand_recognizer is None:
            if str(self.policy.get("fallback_mode", "hold")) == "baseline":
                return self._baseline_decide(current, previous, now, already_updated=True)
            return None

        elixir, source = self._estimate_elixir(current, now)
        if (
            self._last_hand_image is current
            and abs(self._last_hand_observed_at - now) <= 1e-9
        ):
            matches = (
                self.hand_history.matches_for_decision(
                    self._last_hand_matches, now
                )
                if self.temporal_perception_enabled
                else list(self._last_hand_matches)
            )
        else:
            matches = self._stable_hand_matches(current, now)
        hand = tuple(match.card_id for match in matches)
        affordable = self._recognized_affordable(matches, elixir)
        if not affordable:
            return None

        threats = self._perceive_threats(current, previous, now)
        left_threat = threats["left"]
        right_threat = threats["right"]
        imitation_available = bool(
            self.imitation_model is not None and self.imitation_model.available
        )
        imitation_weight = float(
            self.config_demonstration.get("policy_card_weight", 4.0)
        )
        if imitation_available and self.imitation_model is not None:
            imitation_weight *= float(
                getattr(self.imitation_model, "influence_scale", 1.0)
            )
            prepare_frame = getattr(self.imitation_model, "prepare_frame", None)
            if callable(prepare_frame):
                prepare_frame(current)
        replay_available = bool(
            self.replay_model is not None and self.replay_model.available
        )
        replay_weight = float(self.config_replay.get("policy_card_weight", 2.0))
        if replay_available and self.replay_model is not None:
            replay_weight *= float(
                getattr(self.replay_model, "influence_scale", 0.10)
            )
            self.replay_model.prepare_frame(current)
        threat_threshold = float(self.policy.get("enemy_pressure_threshold", 0.20))
        for threat in (left_threat, right_threat):
            if threat.score < threat_threshold:
                self.last_defense_snapshots.pop(threat.lane, None)
        threatening = [
            threat
            for threat in (left_threat, right_threat)
            if threat.score >= threat_threshold
        ]
        due_threats = [
            threat
            for threat in threatening
            if self._defense_response_due(threat, now)
        ]
        if threatening and not due_threats:
            return None
        defending = bool(due_threats)
        strongest = (
            max(due_threats, key=self._threat_urgency)
            if defending
            else (
                left_threat
                if left_threat.score >= right_threat.score
                else right_threat
            )
        )

        formation_phase = "direct_defense" if defending else ""
        desired_role = ""
        formation_source = "defense" if defending else "attack"
        staged_backline = False
        defense_reserve = 0.0
        resource_allocation_reason = ""
        rejected_candidate_reasons: tuple[str, ...] = ()

        if defending:
            lane = strongest.lane
            trusted_threat_layers = self._trusted_threat_layers(strongest)
            if strongest.proximity < 0.43:
                defense_reserve = float(
                    self.policy.get("early_defense_elixir_reserve", 1.5)
                )
            elif strongest.proximity < 0.56:
                defense_reserve = float(
                    self.policy.get("engaged_defense_elixir_reserve", 0.75)
                )
            else:
                defense_reserve = 0.0
            candidate_coverages = [
                (
                    value,
                    card_tactics(value[1]).target_coverage(trusted_threat_layers),
                )
                for value in affordable
            ]
            known_capability_candidates = [
                value
                for value, coverage in candidate_coverages
                if coverage is not None and coverage > 0.0
            ]
            unknown_capability_candidates = [
                value
                for value, coverage in candidate_coverages
                if coverage is None
            ]
            capability_candidates = (
                known_capability_candidates
                if known_capability_candidates
                else unknown_capability_candidates
            )
            if not capability_candidates:
                return None
            other: LaneThreat | None = None
            if bool(self.policy.get("dual_lane_elixir_enabled", True)) and len(due_threats) > 1:
                other = max(
                    (threat for threat in due_threats if threat.lane != strongest.lane),
                    key=self._threat_urgency,
                )
            stage_reserve = defense_reserve
            reserve_cap = float(self.policy.get("dual_lane_defense_reserve_max", 4.0))
            reserve_by_slot: dict[int, tuple[float, float]] = {}
            for value in capability_candidates:
                other_reserve = (
                    self._minimum_defense_cost(
                        [candidate for candidate in affordable if candidate[0].slot_index != value[0].slot_index],
                        other,
                    )
                    if other is not None
                    else 0.0
                )
                reserve_by_slot[int(value[0].slot_index)] = (
                    max(stage_reserve, min(max(0.0, reserve_cap), other_reserve)),
                    other_reserve,
                )
            defensive_affordable = [
                value
                for value in capability_candidates
                if self._effective_cost(
                    value[1], float(self.policy.get("unknown_card_cost", 3))
                )
                <= elixir - reserve_by_slot[int(value[0].slot_index)][0]
            ]
            if not defensive_affordable:
                emergency_score = float(
                    self.policy.get("emergency_threat_score", 0.75)
                )
                if strongest.score < emergency_score:
                    if len(due_threats) <= 1:
                        return None
                defensive_affordable = capability_candidates
                resource_allocation_reason = (
                    "dual_lane_priority_insufficient_elixir"
                    if len(due_threats) > 1
                    else "single_lane_emergency_override"
                )
            match, card = max(
                defensive_affordable,
                key=lambda value: (
                    self._defense_score(
                        value[1], strongest, multi_lane=len(threatening) > 1
                    )
                    + (
                        imitation_weight
                        * self.imitation_model.card_score(value[1], elixir, threats)
                        if imitation_available and self.imitation_model is not None
                        else 0.0
                    )
                    + (
                        replay_weight
                        * (
                            self.replay_model.card_score(
                                value[1],
                                elixir,
                                threats,
                                formation_phase=formation_phase,
                                desired_role=desired_role,
                                battle_elapsed_s=now - self.battle_started_at,
                            )
                            - 0.5
                        )
                        if replay_available and self.replay_model is not None
                        else 0.0
                    ),
                    value[0].confidence,
                    -self._effective_cost(
                        value[1], float(self.policy.get("unknown_card_cost", 3))
                    ),
                ),
            )
            defense_reserve, other_reserve = reserve_by_slot[int(match.slot_index)]
            if not resource_allocation_reason:
                resource_allocation_reason = (
                    "dual_lane_reserve"
                    if other is not None and other_reserve > 0
                    else "dual_lane_other_lane_no_solution"
                    if other is not None
                    else "single_lane_stage_reserve"
                )
            minimum_defense_score = float(self.policy.get("defense_min_card_score", 1.25))
            if self._defense_score(
                card, strongest, multi_lane=len(threatening) > 1
            ) < minimum_defense_score:
                return None
            reason = f"counter_{strongest.threat}_{lane}"
            observation_age_s = max(0.0, now - self._threat_observed_at) if self._threat_observed_at > -999.0 else 0.0
            deploy, placement_role, placement_reason, placement_candidates = self._defense_placement(
                lane, card, strongest, observation_age_s=observation_age_s
            )
            self.last_defense_at = now
            self.last_defense_lane = lane
            self.last_defense_snapshots[lane] = (
                now,
                strongest.score,
                strongest.unit_count,
                strongest.proximity,
            )
            self.push_support_count = 0
            self.counterpush_investment[lane] = 0.0
        else:
            placement_role = "formation"
            placement_reason = "formation_role_spacing"
            placement_candidates = ()
            push_minimum = float(self.policy.get("push_min_elixir", 7.0))
            maximum_supports = int(self.policy.get("maximum_push_supports", 2))
            back_only = [
                lane_name
                for lane_name, formation in self.formations.items()
                if self._formation_has_backline(formation, now)
                and not self._formation_has_frontline(formation, now)
            ]
            front_only = [
                lane_name
                for lane_name, formation in self.formations.items()
                if self._formation_has_frontline(formation, now)
                and not self._formation_has_backline(formation, now)
            ]
            complete = [
                lane_name
                for lane_name, formation in self.formations.items()
                if self._formation_has_frontline(formation, now)
                and self._formation_has_backline(formation, now)
            ]

            def preferred_lane(values: list[str], rank: str) -> str:
                def key(lane_name: str) -> tuple[int, float]:
                    formation = self.formations[lane_name]
                    until = (
                        formation.backline_until
                        if rank == "frontline"
                        else formation.frontline_until
                    )
                    return (1 if self._formation_source(formation, now) == "defense" else 0, until)

                return max(values, key=key)

            if back_only:
                desired_role = "frontline"
                lane = preferred_lane(back_only, desired_role)
                formation_phase = (
                    "protect_surviving_backline"
                    if self._formation_source(self.formations[lane], now) == "defense"
                    else "complete_frontline"
                )
            elif front_only:
                desired_role = "backline"
                lane = preferred_lane(front_only, desired_role)
                formation_phase = (
                    "support_counterpush"
                    if self._formation_source(self.formations[lane], now) == "defense"
                    else "support_frontline"
                )
            elif complete and self.push_support_count < maximum_supports:
                desired_role = "backline"
                lane = preferred_lane(complete, desired_role)
                formation_phase = (
                    "support_counterpush"
                    if self._formation_source(self.formations[lane], now) == "defense"
                    else "reinforce_push"
                )
            elif complete:
                # Do not dump more elixir into a formation that already has
                # both ranks and the configured number of supporting plays.
                return None
            else:
                desired_role = "frontline"
                if abs(left_threat.score - right_threat.score) >= 0.05:
                    lane = "left" if left_threat.score < right_threat.score else "right"
                else:
                    lane = "left" if self.last_push_lane == "right" else "right"
                formation_phase = "form_frontline"

            existing_formation = self.formations[lane]
            formation_source = (
                "defense" if self._formation_source(existing_formation, now) == "defense" else "attack"
            )
            has_formation = self._formation_has_frontline(
                existing_formation, now
            ) or self._formation_has_backline(existing_formation, now)
            if has_formation and self._formation_source(existing_formation, now) == "defense":
                allowed, rejected_candidate_reasons = self._counterpush_gate(
                    lane, threats, current, previous
                )
                if not allowed:
                    self.last_formation_rejections = rejected_candidate_reasons
                    return None
            if has_formation and self._formation_source(existing_formation, now) == "defense":
                required_elixir = float(
                    self.policy.get("counterpush_min_elixir", 3.0)
                )
            elif has_formation:
                required_elixir = float(self.policy.get("support_min_elixir", 4.0))
            else:
                required_elixir = push_minimum
            if elixir < required_elixir:
                return None

            formation_candidates = [
                value
                for value in affordable
                if self._fits_formation_role(value[1], desired_role)
            ]
            if not formation_candidates:
                overflow = float(self.policy.get("overflow_elixir", 9.0))
                backline_candidates = [
                    value
                    for value in affordable
                    if value[1].kind == "troop"
                    and card_tactics(value[1]).formation_role == "backline"
                ]
                if (
                    desired_role != "frontline"
                    or self._formation_has_backline(existing_formation, now)
                    or elixir < overflow
                    or not backline_candidates
                ):
                    return None
                formation_candidates = backline_candidates
                desired_role = "backline"
                staged_backline = True
                formation_phase = "stage_backline"

            match, card = max(
                formation_candidates,
                key=lambda value: (
                    self._formation_card_score(value[1], desired_role)
                    + (
                        imitation_weight
                        * self.imitation_model.card_score(value[1], elixir, threats)
                        if imitation_available and self.imitation_model is not None
                        else 0.0
                    )
                    + (
                        replay_weight
                        * (
                            self.replay_model.card_score(
                                value[1],
                                elixir,
                                threats,
                                formation_phase=formation_phase,
                                desired_role=desired_role,
                                battle_elapsed_s=now - self.battle_started_at,
                            )
                            - 0.5
                        )
                        if replay_available and self.replay_model is not None
                        else 0.0
                    ),
                    value[0].confidence,
                    -self._effective_cost(
                        value[1], float(self.policy.get("unknown_card_cost", 3))
                    ),
                ),
            )
            best_score = self._formation_card_score(card, desired_role)
            minimum_score = float(self.policy.get("formation_min_card_score", 1.0))
            if best_score < minimum_score:
                return None
            card_cost = self._effective_cost(card, float(self.policy.get("unknown_card_cost", 3)))
            if formation_source == "defense":
                investment_cap = float(self.policy.get("counterpush_investment_max", 6.0))
                projected = self.counterpush_investment[lane] + card_cost
                if projected > investment_cap:
                    self.last_formation_rejections = (
                        f"investment_cap={projected:.1f}>{investment_cap:.1f}",
                    )
                    return None
                self.counterpush_investment[lane] = projected
            else:
                self.counterpush_investment[lane] = 0.0
            self.last_formation_rejections = ()
            reason = f"{formation_phase}_{lane}"
            deploy = self._push_point(
                lane,
                card,
                desired_role=desired_role,
                staged_backline=staged_backline,
                elixir=elixir,
                threats=threats,
                now=now,
            )
            roles = frozenset(card.roles)
            if self._formation_has_frontline(existing_formation, now) or self._formation_has_backline(existing_formation, now):
                self.push_support_count += 1
            else:
                self.push_support_count = 0
            self.last_push_at = now
            self.last_push_roles = roles
            self.last_push_lane = lane

        if imitation_available and self.imitation_model is not None:
            learned_deploy = self.imitation_model.deploy_point(card, elixir, threats)
            if learned_deploy is not None:
                x_range = self.policy["left_x"] if lane == "left" else self.policy["right_x"]
                safe_x = min(float(x_range[1]), max(float(x_range[0]), learned_deploy[0]))
                if defending:
                    y_range = self.policy.get("defense_deploy_bounds", [0.50, 0.72])
                elif card.kind == "spell" and card_tactics(card).is_win_condition:
                    y_range = [0.18, 0.42]
                else:
                    y_range = [0.50, 0.75]
                safe_y = min(float(y_range[1]), max(float(y_range[0]), learned_deploy[1]))
                deploy_weight = float(
                    self.config_demonstration.get("policy_deploy_weight", 0.5)
                )
                deploy_weight *= float(
                    getattr(self.imitation_model, "influence_scale", 1.0)
                )
                deploy = [
                    deploy[0] * (1.0 - deploy_weight) + safe_x * deploy_weight,
                    deploy[1] * (1.0 - deploy_weight) + safe_y * deploy_weight,
                ]

        if replay_available and self.replay_model is not None:
            learned_deploy = self.replay_model.deploy_point(
                card,
                elixir,
                threats,
                formation_phase=formation_phase,
                desired_role=desired_role,
                battle_elapsed_s=now - self.battle_started_at,
            )
            if learned_deploy is not None:
                x_range = self.policy["left_x"] if lane == "left" else self.policy["right_x"]
                safe_x = min(float(x_range[1]), max(float(x_range[0]), learned_deploy[0]))
                if defending:
                    y_range = self.policy.get("defense_deploy_bounds", [0.50, 0.72])
                elif card.kind == "spell" and card_tactics(card).is_win_condition:
                    y_range = [0.18, 0.42]
                else:
                    y_range = [0.50, 0.75]
                safe_y = min(float(y_range[1]), max(float(y_range[0]), learned_deploy[1]))
                deploy_weight = float(
                    self.config_replay.get("policy_deploy_weight", 0.20)
                ) * float(getattr(self.replay_model, "influence_scale", 0.10))
                deploy = [
                    deploy[0] * (1.0 - deploy_weight) + safe_x * deploy_weight,
                    deploy[1] * (1.0 - deploy_weight) + safe_y * deploy_weight,
                ]

        self._remember_formation(lane, card, deploy, now, formation_source)
        unknown_cost = float(self.policy.get("unknown_card_cost", 3))
        cost = self._effective_cost(card, unknown_cost)
        self.virtual_elixir = max(0.0, self.virtual_elixir - cost)
        self.last_deploy_point = (float(deploy[0]), float(deploy[1]))
        self.last_deploy_at = now
        self.action_sequence += 1
        cooldown = self.timing["battle_action_cooldown_s"]
        urgent = defending and (
            strongest.score >= float(self.policy.get("emergency_threat_score", 0.75))
            or strongest.proximity >= float(self.policy.get("emergency_threat_proximity", 0.62))
            or strongest.approach_rate >= float(self.policy.get("emergency_approach_rate", 0.08))
        )
        if urgent:
            self.next_action_at = now + max(
                0.2, float(self.policy.get("emergency_action_cooldown_s", 0.65))
            )
        else:
            self.next_action_at = now + self.random.uniform(float(cooldown[0]), float(cooldown[1]))
        left_motion = motion_score(current, previous, self.vision["friendly_left_roi"])
        right_motion = motion_score(current, previous, self.vision["friendly_right_roi"])
        selected_hand_state = self.hand_history.slots.get(int(match.slot_index))
        selected_hand_age = (
            selected_hand_state.age(now) if selected_hand_state is not None else 0.0
        )
        selected_hand_confidence = (
            self.hand_history._decayed_confidence(selected_hand_state, now)
            if selected_hand_state is not None
            else float(match.confidence)
        )
        slots = self.vision["card_slot_centers"]
        return BattleDecision(
            slot_index=match.slot_index,
            card_point=list(slots[match.slot_index]),
            deploy_point=deploy,
            lane=lane,
            reason=reason,
            elixir=round(float(elixir), 3),
            elixir_source=source,
            left_motion=round(left_motion, 5),
            right_motion=round(right_motion, 5),
            card_id=card.card_id,
            card_name=card.name_zh or card.name_en,
            card_cost=card.elixir,
            hand=hand,
            left_threat=left_threat.score,
            right_threat=right_threat.score,
            left_threat_type=left_threat.threat,
            right_threat_type=right_threat.threat,
            left_threat_proximity=left_threat.proximity,
            right_threat_proximity=right_threat.proximity,
            left_threat_approach_rate=left_threat.approach_rate,
            right_threat_approach_rate=right_threat.approach_rate,
            left_unit_count=left_threat.unit_count,
            right_unit_count=right_threat.unit_count,
            battle_elapsed_s=round(max(0.0, now - self.battle_started_at), 3),
            threat_type=strongest.threat if defending else "none",
            threat_proximity=strongest.proximity if defending else 0.0,
            threat_approach_rate=strongest.approach_rate if defending else 0.0,
            enemy_cards=strongest.enemy_cards if defending else (),
            left_threat_unit_layers=left_threat.unit_layers,
            right_threat_unit_layers=right_threat.unit_layers,
            threat_unit_layers=strongest.unit_layers if defending else (),
            left_threat_layer_confidence=left_threat.layer_confidence,
            right_threat_layer_confidence=right_threat.layer_confidence,
            threat_layer_confidence=(
                strongest.layer_confidence if defending else 0.0
            ),
            card_attack_targets=card_tactics(card).attack_targets,
            card_targeting_source=card_tactics(card).targeting_source,
            imitation_used=imitation_available,
            replay_learning_used=replay_available,
            card_formation_role=card_tactics(card).formation_role,
            formation_phase=formation_phase,
            desired_formation_role=desired_role,
            hand_confidence=round(selected_hand_confidence, 4),
            hand_age_s=round(selected_hand_age, 3),
            elixir_confidence=round(self._last_elixir_confidence, 4),
            elixir_age_s=round(
                max(0.0, now - self._last_elixir_observed_at)
                if self._last_elixir_observed_at > -999.0
                else -1.0,
                3,
            ),
            elixir_phase=self._elixir_phase,
            defense_elixir_reserve=round(defense_reserve, 3),
            resource_allocation_reason=resource_allocation_reason,
            placement_role=placement_role,
            placement_reason=placement_reason,
            placement_candidates=placement_candidates,
            counterpush_investment=round(self.counterpush_investment.get(lane, 0.0), 3),
            rejected_candidate_reasons=rejected_candidate_reasons,
        )

    def _baseline_decide(
        self,
        current: Image.Image,
        previous: Image.Image | None,
        now: float,
        *,
        already_updated: bool = False,
    ) -> BattleDecision | None:
        if not already_updated:
            self._update_virtual_elixir(now)
        if now < self.next_action_at:
            return None

        elixir, source = self._estimate_elixir(current, self.last_update)
        minimum = float(self.policy.get("min_elixir_to_act", 4))
        if elixir < minimum:
            return None

        left = motion_score(current, previous, self.vision["friendly_left_roi"])
        right = motion_score(current, previous, self.vision["friendly_right_roi"])
        threshold = float(self.vision.get("motion_threshold", 0.045))
        defending = max(left, right) >= threshold

        if defending:
            lane = "left" if left >= right else "right"
            reason = f"defend_{lane}"
        else:
            lane = "left" if self.last_push_lane == "right" else "right"
            self.last_push_lane = lane
            reason = f"push_{lane}"

        slots = self.vision["card_slot_centers"]
        self.slot_cursor = (self.slot_cursor + 1 + self.random.randrange(3)) % len(slots)
        slot = self.slot_cursor
        deploy = self._point_for_lane(lane, defending)

        estimated_cost = float(self.policy.get("unknown_card_cost", 3))
        self.virtual_elixir = max(0.0, self.virtual_elixir - estimated_cost)
        cooldown = self.timing["battle_action_cooldown_s"]
        self.next_action_at = now + self.random.uniform(float(cooldown[0]), float(cooldown[1]))

        return BattleDecision(
            slot_index=slot,
            card_point=list(slots[slot]),
            deploy_point=deploy,
            lane=lane,
            reason=reason,
            elixir=round(float(elixir), 3),
            elixir_source=source,
            left_motion=round(left, 5),
            right_motion=round(right, 5),
            elixir_confidence=round(self._last_elixir_confidence, 4),
            elixir_age_s=round(
                max(0.0, self.last_update - self._last_elixir_observed_at)
                if self._last_elixir_observed_at > -999.0
                else -1.0,
                3,
            ),
            elixir_phase=self._elixir_phase,
        )

    def decide(
        self,
        current: Image.Image,
        previous: Image.Image | None,
        *,
        now: float | None = None,
    ) -> BattleDecision | None:
        now = time.monotonic() if now is None else now
        if self._pending_action_id is not None:
            return None
        self._update_virtual_elixir(now)
        if now < self.next_action_at:
            return None
        if self.mode == "reactive_catalog":
            return self._reactive_decide(current, previous, now)
        return self._baseline_decide(current, previous, now, already_updated=True)
