from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .temporal import TimingStats


def audit_predictions(project_root: Path) -> dict:
    engines, fallbacks = Counter(), Counter()
    times, searches, confirmed = [], 0, 0
    errors = []
    trajectory=[]
    for path in sorted((project_root / "runs").glob("*/events.jsonl")):
        tracking_events=[]
        try:
            with path.open(encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    try:
                        event = json.loads(line)
                    except ValueError:
                        errors.append(f"{path.name}:{number}:invalid_json")
                        continue
                    payload = event
                    if event.get("event") == "battle_prediction":
                        engines[payload.get("actual_engine", "unknown")] += 1
                        p=payload.get("plan") or {}
                        tracking_events.append({"event":"battle_prediction","world":payload.get("world",{}),"plan":{"compute":p.get("compute",{}),"enemy_forecast":p.get("enemy_forecast",[])}})
                        plan = payload.get("plan") or {}
                        if plan.get("fallback_reason"):
                            fallbacks[plan["fallback_reason"]] += 1
                        if plan.get("elapsed_ms") is not None:
                            times.append(float(plan["elapsed_ms"]))
                            searches += 1
                    if event.get("event") == "battle_action" and payload.get("decision_engine") == "predictive" and payload.get("action_status") == "confirmed":
                        confirmed += 1
            from .trajectory_audit import trajectory_errors
            trajectory.append(trajectory_errors(tracking_events))
        except OSError as exc:
            errors.append(str(exc))
    return {"actual_engine_counts": dict(engines), "fallback_reasons": dict(fallbacks),
            "searches": searches, "confirmed_predictive_actions": confirmed,
            "search_p50_ms": TimingStats.percentile(times, .5),
            "search_p95_ms": TimingStats.percentile(times, .95),
            "errors": errors, "trajectory_consistency": trajectory, "battle_acceptance": False,
            "note": "运行覆盖和延迟审计，不等于胜率提升或反事实验证"}
