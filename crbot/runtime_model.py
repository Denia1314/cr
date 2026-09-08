from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


RUNTIME_MODEL_LABELS: dict[str, str] = {
    "hybrid": "混合增强（推荐）",
    "rules": "规则策略",
    "imitation": "示范模仿模型",
    "replay": "回放自学习模型",
}
DEFAULT_RUNTIME_MODEL = "hybrid"
LOCAL_SETTINGS_NAME = ".royal-lab.json"


def normalize_runtime_model(value: object) -> str:
    key = str(value or "").strip().lower()
    return key if key in RUNTIME_MODEL_LABELS else DEFAULT_RUNTIME_MODEL


def runtime_model_label(value: object) -> str:
    return RUNTIME_MODEL_LABELS[normalize_runtime_model(value)]


def runtime_model_allows(value: object, source: str) -> bool:
    key = normalize_runtime_model(value)
    if source == "imitation":
        return key in {"hybrid", "imitation"}
    if source == "replay":
        return key in {"hybrid", "replay"}
    return True


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
    return normalize_runtime_model(payload.get("runtime_model", fallback))


def save_runtime_model(config_path: Path, value: object) -> str:
    key = normalize_runtime_model(value)
    path = local_settings_path(config_path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump({"runtime_model": key}, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)
    return key


def apply_runtime_model(config: dict[str, Any], value: object) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    effective.setdefault("policy", {})["runtime_model"] = normalize_runtime_model(value)
    return effective
