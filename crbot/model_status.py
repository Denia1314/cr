"""Read model versions for the console without loading or promoting weights."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MODEL_KINDS = (
    ("replay_policy", "回放", "replay_model"),
    ("imitation", "示范", "imitation_model"),
    ("battlefield", "识别", "learned_detector"),
)


def _created_at(candidate: dict[str, Any]) -> float:
    try:
        return float(candidate.get("created_at_unix") or 0)
    except (TypeError, ValueError):
        return 0


def _qualification(candidate: dict[str, Any]) -> str:
    if candidate.get("quality_passed") is False or candidate.get("status") == "rejected":
        return "未通过验证"
    if candidate.get("promoted") is True:
        return "已晋级"
    if candidate.get("quality_passed") is True:
        return "验证通过，未晋级"
    if candidate.get("promoted") is False:
        return "未晋级"
    return "验证结果未记录"


def model_status(project_root: Path, *, policy: Any = None, running: bool = False,
                 demonstration: bool = False) -> tuple[str, str]:
    """Return a compact latest-version summary and a complete read-only inventory.

    Only the live engine can establish what this run loaded. A registry champion
    or an old runtime-status.json is never treated as proof of current loading.
    """
    latest_models = []
    sections = []
    loaded = []
    errors = []
    for directory, label, attribute in MODEL_KINDS:
        model_root = (project_root / "models" / directory).resolve()
        try:
            registry = json.loads((model_root / "registry.json").read_text(encoding="utf-8"))
            if not isinstance(registry, dict) or not isinstance(registry.get("candidates", []), list):
                raise ValueError("模型目录格式无效")
        except FileNotFoundError:
            registry = {}
        except (OSError, ValueError) as exc:
            errors.append(label)
            sections.append(f"{label}模型：读取失败（{exc}）")
            registry = {}
        candidates = [item for item in registry.get("candidates", []) if isinstance(item, dict)]
        champion = registry.get("champion")
        champion = champion if isinstance(champion, dict) else {}
        if champion and champion not in candidates:
            candidates.append(champion)
        candidates.sort(key=lambda item: (_created_at(item), str(item.get("version", ""))), reverse=True)

        model = getattr(policy, attribute, None) if running and not demonstration else None
        if model is not None and model.available:
            version = str((model.champion or {}).get("version", "未知版本"))
            loaded.append(f"{label} {version}")
            runtime = f"已加载 {version}"
            scale = getattr(model, "influence_scale", None)
            if scale is not None:
                runtime += f"（当前影响权重 {scale:g}）"
        elif not running:
            runtime = "未启动"
        elif demonstration:
            runtime = "示范录制中"
        elif policy is None:
            runtime = "启动中"
        else:
            reason = getattr(model, "load_error", "") or getattr(model, "error", "")
            runtime = "未加载" + (f"（{reason}）" if reason else "")
        lines = [f"{label}模型 · 本次运行：{runtime}",
                 f"当前冠军：{champion.get('version', '无')}"]
        if candidates:
            latest_models.append((_created_at(candidates[0]), label, candidates[0]))
        else:
            lines.append("暂无本地模型")
        for candidate in candidates:
            version = str(candidate.get("version", "未知版本"))
            source = "同步接收" if candidate.get("sync_publication") else "本机训练"
            model_path = (model_root / str(candidate.get("model_path") or "")).resolve()
            present = model_path.is_relative_to(model_root) and model_path.is_file()
            lines.append(f"{version} · {_qualification(candidate)} · {source} · "
                         + ("文件就绪" if present else "文件缺失"))
            for field in ("rejection_reasons", "promotion_blockers"):
                reasons = candidate.get(field)
                if isinstance(reasons, list) and reasons:
                    lines.append("  " + "；".join(str(reason) for reason in reasons))
        sections.append("\n".join(lines))

    if latest_models:
        _, label, candidate = max(latest_models, key=lambda item: (item[0], str(item[2].get("version", ""))))
        summary = f"最新模型：{label} {candidate.get('version', '未知版本')}（{_qualification(candidate)}）"
    else:
        summary = "模型：暂无本地模型"
    if errors:
        summary += " · " + "、".join(errors) + "目录读取失败"
    runtime_summary = "；".join(loaded) if loaded else (
        "未启动" if not running else "示范录制中" if demonstration
        else "启动中" if policy is None else "未加载模型"
    )
    summary += f"\n本次加载：{runtime_summary} · 点击查看全部版本"
    return summary, "\n\n".join(sections)
