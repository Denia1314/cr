"""Read-only sampling and independent review metrics for action confirmation."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from .replay import ACTION_CONFIRMATION_STATUSES, _read_jsonl


def _status(row: dict[str, Any]) -> str:
    action = row.get("action", {})
    value = str(row.get("action_status") or (action.get("action_status") if isinstance(action, dict) else "") or "legacy_unknown")
    return value if value in ACTION_CONFIRMATION_STATUSES else "unknown"


def export_action_review(project_root: Path, output: Path, *, limit: int = 200, seed: int = 20260908) -> dict[str, Any]:
    """Export a deterministic, status-interleaved review sheet without images."""
    output = output.resolve()
    if output.exists():
        raise ValueError(f"复核文件已存在，未覆盖：{output}")
    rows: list[dict[str, Any]] = []
    for run_dir in sorted((project_root.resolve() / "runs").glob("*")):
        for row in _read_jsonl(run_dir / "replay_transitions.jsonl"):
            action = row.get("action", {})
            if not isinstance(action, dict):
                continue
            transition_id = str(row.get("transition_id", "")).strip()
            if not transition_id:
                continue
            status = _status(row)
            if status not in {"confirmed", "rejected", "unknown"}:
                continue
            state_frame = row.get("state_frame")
            rows.append({
                "transition_id": transition_id,
                "run": run_dir.name,
                "battle_index": row.get("battle_index"),
                "action_status": status,
                "action_id": action.get("action_id"),
                "card_id": action.get("card_id"),
                "lane": action.get("lane"),
                "state_frame": str((run_dir / state_frame).resolve()) if isinstance(state_frame, str) and state_frame else None,
                "confirmation": action.get("action_confirmation", {}),
                "review_actual_success": None,
                "review_delay_s": None,
                "review_notes": "",
            })
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row["action_status"]), []).append(row)
    for values in buckets.values():
        values.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['transition_id']}".encode()).hexdigest())
    selected: list[dict[str, Any]] = []
    statuses = sorted(buckets)
    while len(selected) < max(1, int(limit)) and statuses:
        remaining = []
        for status in statuses:
            if buckets[status] and len(selected) < max(1, int(limit)):
                selected.append(buckets[status].pop())
            if buckets[status]:
                remaining.append(status)
        statuses = remaining
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "output": str(output),
        "sampled_attempts": len(selected),
        "status_counts": {status: sum(row["action_status"] == status for row in selected) for status in sorted({row["action_status"] for row in selected})},
        "labels_required": ["true", "false", "unknown"],
    }


def _metric(rows: list[dict[str, Any]], kind: str) -> float | None:
    known = [row for row in rows if row["actual"] in {True, False}]
    if kind == "precision":
        denominator = [row for row in known if row["status"] == "confirmed"]
        return sum(row["actual"] is True for row in denominator) / len(denominator) if denominator else None
    denominator = [row for row in known if row["actual"] is True]
    return sum(row["status"] == "confirmed" for row in denominator) / len(denominator) if denominator else None


def audit_action_review(path: Path, *, seed: int = 20260908) -> dict[str, Any]:
    reviewed = []
    for row in _read_jsonl(path.resolve()):
        raw = row.get("review_actual_success")
        actual = raw if isinstance(raw, bool) else None
        if isinstance(raw, str) and raw.lower() in {"true", "false"}:
            actual = raw.lower() == "true"
        reviewed.append({
            "battle": f"{row.get('run')}:{row.get('battle_index')}",
            "status": _status(row),
            "actual": actual,
            "review_unknown": raw == "unknown",
        })
    labeled = [row for row in reviewed if row["actual"] in {True, False}]
    battles = sorted({row["battle"] for row in labeled})
    rng = random.Random(seed)
    intervals: dict[str, list[float] | None] = {}
    for kind in ("precision", "recall"):
        samples = []
        if battles:
            for _ in range(1000):
                chosen = [rng.choice(battles) for _ in battles]
                value = _metric([row for battle in chosen for row in labeled if row["battle"] == battle], kind)
                if value is not None:
                    samples.append(value)
        samples.sort()
        intervals[kind] = [round(samples[int(0.025 * (len(samples) - 1))], 6), round(samples[int(0.975 * (len(samples) - 1))], 6)] if samples else None
    return {
        "read_only": True,
        "attempts": len(reviewed),
        "labeled_attempts": len(labeled),
        "review_unknown_attempts": sum(row["review_unknown"] for row in reviewed),
        "battles": len(battles),
        "confirmation_precision": _metric(labeled, "precision"),
        "successful_action_recall": _metric(labeled, "recall"),
        "system_unknown_rate": sum(row["status"] == "unknown" for row in reviewed) / len(reviewed) if reviewed else None,
        "battle_cluster_bootstrap_95": intervals,
    }
