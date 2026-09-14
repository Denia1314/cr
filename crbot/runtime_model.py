from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from . import current_stage_label

DECISION_ENGINE_LABELS = {
    "legacy": "原有策略 · M3-E / 旧版本",
    "shadow": "推演观察 · 旧策略出牌",
    "predictive": current_stage_label() + " / P1 推演",
}

RUNTIME_MODEL_LABELS: dict[str, str] = {
    "v5": "V5 · 基础策略",
    "m1": "M1 · 动作确认",
    "m2": "M2 · 时序感知",
    "m2_1": "M2.1 · 稳定修复",
    "m3_c1": "M3-C1 · 双路留费",
    "m3_c2": "M3-C2 · 职责落点",
    # Historical profile keeps its own stage; the new engine has a separate selector.
    "m3_c3": "M3-E · 受控实测",
}
DEFAULT_RUNTIME_MODEL = "m3_c3"
LOCAL_SETTINGS_NAME = ".royal-lab.json"


RUNTIME_MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "v5": {
        "policy_version": "adaptive_counterpush_v5",
        "training_policy_version": "adaptive_counterpush_v5",
        "transfer_policy_versions": ["formation_counterpush_v4"],
        "require_action_confirmation": False,
        "temporal_perception_enabled": False,
        "hand_stability_frames": 1,
        "dual_lane_elixir_enabled": False,
        "role_aware_placement_enabled": False,
        "counterpush_evidence_enabled": False,
    },
    "m1": {
        "policy_version": "adaptive_counterpush_v5",
        "training_policy_version": "adaptive_counterpush_v5",
        "transfer_policy_versions": ["formation_counterpush_v4"],
        "require_action_confirmation": True,
        "temporal_perception_enabled": False,
        "hand_stability_frames": 1,
        "dual_lane_elixir_enabled": False,
        "role_aware_placement_enabled": False,
        "counterpush_evidence_enabled": False,
    },
    "m2": {
        "policy_version": "adaptive_counterpush_v5_m2",
        "training_policy_version": "adaptive_counterpush_v5_m2",
        "transfer_policy_versions": [
            "adaptive_counterpush_v5",
            "formation_counterpush_v4",
        ],
        "require_action_confirmation": True,
        "temporal_perception_enabled": True,
        "hand_stability_frames": 1,
        "dual_lane_elixir_enabled": False,
        "role_aware_placement_enabled": False,
        "counterpush_evidence_enabled": False,
    },
    "m2_1": {
        "policy_version": "adaptive_counterpush_v5_m2_1",
        "training_policy_version": "adaptive_counterpush_v5_m2_1",
        "transfer_policy_versions": [
            "adaptive_counterpush_v5_m2",
            "adaptive_counterpush_v5",
            "formation_counterpush_v4",
        ],
        "require_action_confirmation": True,
        "temporal_perception_enabled": True,
        "hand_stability_frames": 2,
        "dual_lane_elixir_enabled": False,
        "role_aware_placement_enabled": False,
        "counterpush_evidence_enabled": False,
    },
    "m3_c1": {
        "policy_version": "adaptive_counterpush_v5_m3_c1",
        "training_policy_version": "adaptive_counterpush_v5_m3_c1",
        "transfer_policy_versions": [
            "adaptive_counterpush_v5_m2_1",
            "adaptive_counterpush_v5_m2",
            "adaptive_counterpush_v5",
            "formation_counterpush_v4",
        ],
        "require_action_confirmation": True,
        "temporal_perception_enabled": True,
        "hand_stability_frames": 2,
        "dual_lane_elixir_enabled": True,
        "role_aware_placement_enabled": False,
        "counterpush_evidence_enabled": False,
    },
    "m3_c2": {
        "policy_version": "adaptive_counterpush_v5_m3_c2",
        "training_policy_version": "adaptive_counterpush_v5_m3_c2",
        "transfer_policy_versions": [
            "adaptive_counterpush_v5_m3_c1",
            "adaptive_counterpush_v5_m2_1",
            "adaptive_counterpush_v5_m2",
            "adaptive_counterpush_v5",
            "formation_counterpush_v4",
        ],
        "require_action_confirmation": True,
        "temporal_perception_enabled": True,
        "hand_stability_frames": 2,
        "dual_lane_elixir_enabled": True,
        "role_aware_placement_enabled": True,
        "counterpush_evidence_enabled": False,
    },
    "m3_c3": {
        "policy_version": "adaptive_counterpush_v5_m3_c3",
        "training_policy_version": "adaptive_counterpush_v5_m3_c3",
        "transfer_policy_versions": [
            "adaptive_counterpush_v5_m3_c2",
            "adaptive_counterpush_v5_m3_c1",
            "adaptive_counterpush_v5_m2_1",
            "adaptive_counterpush_v5_m2",
            "adaptive_counterpush_v5",
            "formation_counterpush_v4",
        ],
        "require_action_confirmation": True,
        "temporal_perception_enabled": True,
        "hand_stability_frames": 2,
        "dual_lane_elixir_enabled": True,
        "role_aware_placement_enabled": True,
        "counterpush_evidence_enabled": True,
    },
}


def normalize_runtime_model(value: object) -> str:
    key = str(value or "").strip().lower()
    return key if key in RUNTIME_MODEL_LABELS else DEFAULT_RUNTIME_MODEL


def runtime_model_label(value: object) -> str:
    return RUNTIME_MODEL_LABELS[normalize_runtime_model(value)]


def local_settings_path(config_path: Path) -> Path:
    return config_path.resolve().parent / LOCAL_SETTINGS_NAME


def load_runtime_model(config_path: Path, config: dict[str, Any] | None = None) -> str:
    fallback = normalize_runtime_model(
        (config or {}).get("policy", {}).get("runtime_model", DEFAULT_RUNTIME_MODEL)
    )
    path = local_settings_path(config_path)
    if not path.is_file():
        return fallback
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return fallback
    return normalize_runtime_model(payload.get("runtime_model", fallback)) if isinstance(payload, dict) else fallback


def save_runtime_model(config_path: Path, value: object) -> str:
    key = normalize_runtime_model(value)
    _save_selection(config_path, "runtime_model", key)
    return key


def _save_selection(config_path: Path, field: str, value: str) -> None:
    path = local_settings_path(config_path)
    payload = {}
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("本机设置不是 JSON 对象，保留文件并停止写入")
    payload[field] = value
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def load_decision_engine(config_path: Path, config: dict | None = None) -> str:
    fallback = (config or {}).get("policy", {}).get("decision_engine", "legacy")
    try:
        payload = json.loads(local_settings_path(config_path).read_text(encoding="utf-8"))
        mode = payload.get("decision_engine", fallback)
    except (OSError, ValueError, AttributeError):
        mode = fallback
    return mode if mode in DECISION_ENGINE_LABELS else "legacy"


def save_decision_engine(config_path: Path, value: str) -> str:
    if value not in DECISION_ENGINE_LABELS:
        raise ValueError("未知决策引擎")
    _save_selection(config_path, "decision_engine", value)
    return value


def apply_decision_engine(config: dict, value: str) -> dict:
    if value not in DECISION_ENGINE_LABELS:
        raise ValueError("未知决策引擎")
    effective = copy.deepcopy(config)
    effective.setdefault("policy", {})["decision_engine"] = value
    if value == "predictive":
        # Predictor trajectories must not masquerade as legacy training labels.
        effective.setdefault("replay", {}).update({"training_policy_version": "predictive_calibrated_v2",
            "transfer_policy_versions": [], "require_action_confirmation": True})
        effective["policy"]["version"] = "predictive_calibrated_v2"
        effective["policy"]["temporal_perception_enabled"] = True
        effective["policy"]["hand_stability_frames"] = max(2, int(effective["policy"].get("hand_stability_frames", 2)))
    return effective


def apply_runtime_model(config: dict[str, Any], value: object) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    key = normalize_runtime_model(value)
    profile = RUNTIME_MODEL_PROFILES[key]
    policy = effective.setdefault("policy", {})
    replay = effective.setdefault("replay", {})
    policy["runtime_model"] = key
    policy["version"] = profile["policy_version"]
    policy["hand_stability_frames"] = profile["hand_stability_frames"]
    policy["temporal_perception_enabled"] = profile[
        "temporal_perception_enabled"
    ]
    policy["dual_lane_elixir_enabled"] = profile["dual_lane_elixir_enabled"]
    policy["role_aware_placement_enabled"] = profile[
        "role_aware_placement_enabled"
    ]
    policy["counterpush_evidence_enabled"] = profile[
        "counterpush_evidence_enabled"
    ]
    replay["training_policy_version"] = profile["training_policy_version"]
    replay["transfer_policy_versions"] = list(profile["transfer_policy_versions"])
    replay["require_action_confirmation"] = profile[
        "require_action_confirmation"
    ]
    return effective
