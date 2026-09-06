"""Conservative short-horizon labels; these are visual proxies, not causal rewards."""
from __future__ import annotations

import math
from typing import Any


def _number(state: dict[str, Any], key: str) -> float | None:
    try:
        value = float(state[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def action_feedback(row: dict[str, Any]) -> tuple[float, float, str]:
    """Return signed effect, confidence, source. Never read terminal outcome.

    The observation window ends before the next friendly action. Legacy next
    states have a lower confidence than sustained post-action observations.
    """
    before, action = row.get("state", {}), row.get("action", {})
    if not isinstance(before, dict) or not isinstance(action, dict):
        return 0.0, 0.0, "missing_state"
    lane = action.get("lane")
    start = _number(before, "battle_elapsed_s")
    if lane not in {"left", "right"} or start is None:
        return 0.0, 0.0, "missing_context"
    observations = row.get("action_observations", [])
    if not isinstance(observations, list):
        observations = []
    end = row.get("next_state")
    # A terminal screen cannot establish what happened to this action.
    if isinstance(end, dict) and end.get("schema") == "terminal_state_v1":
        end = None
    end_at = _number(end, "battle_elapsed_s") if isinstance(end, dict) else None
    samples = []
    for state in observations:
        if not isinstance(state, dict):
            continue
        stamp = _number(state, "battle_elapsed_s")
        if stamp is not None and 2.8 <= stamp - start <= 8.0 and (end_at is None or stamp <= end_at):
            samples.append(state)
    # Two independently timed frames are needed to claim sustained relief.
    samples = list({float(s["battle_elapsed_s"]): s for s in samples}.values())
    samples.sort(key=lambda s: s["battle_elapsed_s"])
    sustained = len(samples) >= 2 and samples[-1]["battle_elapsed_s"] - samples[0]["battle_elapsed_s"] >= 0.6
    if not sustained:
        if not isinstance(end, dict) or end_at is None or not 2.8 <= end_at - start <= 8.0:
            return 0.0, 0.0, "unobserved_window"
        samples = [end]
    else:
        samples = samples[-3:]
    phase = str(action.get("formation_phase") or action.get("reason", "")).lower()
    defense = "defense" in phase or "defend" in phase or (phase.startswith("counter_") and "counterpush" not in phase)
    if defense:
        score = _number(before, lane + "_threat")
        proximity = _number(before, lane + "_threat_proximity")
        count = _number(before, lane + "_unit_count")
        if score is None or proximity is None or count is None or not 0.17 <= score <= 1 or not 0 <= proximity <= 1 or count < 1:
            return 0.0, 0.0, "no_initial_threat"
        # Near a tower, disappearing markers could mean tower damage/death.
        if proximity >= 0.67:
            return 0.0, 0.0, "tower_zone_ambiguous"
        effects = []
        for after in samples:
            pressure = _number(after, lane + "_threat")
            position = _number(after, lane + "_threat_proximity")
            units = _number(after, lane + "_unit_count")
            if pressure is None or position is None or units is None:
                return 0.0, 0.0, "incomplete_observation"
            if not 0 <= pressure <= 1 or not 0 <= position <= 1 or units < 0:
                return 0.0, 0.0, "invalid_observation"
            # One blank frame is not evidence of a successful defense.
            if units == 0 and not sustained:
                return 0.0, 0.0, "single_frame_disappearance"
            relief = score - pressure
            advance = max(0.0, position - proximity) if units else 0.0
            # Ignore the heuristic's inferred type (e.g. 'heavy' from proximity).
            effects.append(max(-1.0, min(1.0, 1.5 * relief - 2.0 * advance)))
        effect = min(effects)  # All retained frames must support the benefit.
        return effect, 0.5 if sustained else 0.2, "defense_pressure_proxy"

    # Attack/counterpush: only measured allied detections can show advancement.
    # Formation memory predicts survival; it must never supply reward labels.
    allies = before.get("observed_allies")
    if not before.get("allies_observed") or not isinstance(allies, list) or not sustained:
        return 0.0, 0.0, "attack_unobserved"
    changes = []
    for after in samples:
        if not after.get("allies_observed"):
            return 0.0, 0.0, "attack_unobserved"
        observed = after.get("observed_allies")
        if not isinstance(observed, list):
            return 0.0, 0.0, "attack_unobserved"
        remaining = [dict(o, x=_number(o, "x"), y=_number(o, "y")) for o in observed
                     if isinstance(o, dict) and _number(o, "x") is not None and _number(o, "y") is not None]
        motion = []
        for ally in allies:
            if not isinstance(ally, dict) or ally.get("lane") != lane:
                continue
            y, x = _number(ally, "y"), _number(ally, "x")
            if x is None or y is None:
                continue
            matches = [other for other in remaining if isinstance(other, dict)
                       and other.get("card_id") == ally.get("card_id")
                       and _number(other, "x") is not None and _number(other, "y") is not None
                       and abs(other["x"] - x) <= 0.1 and abs(other["y"] - y) <= 0.20]
            if matches:
                match = min(matches, key=lambda o: abs(o["x"] - x) + abs(o["y"] - y))
                remaining.remove(match)
                motion.append(max(-1.0, min(1.0, (y - match["y"]) / 0.15)))
        if not motion:
            return 0.0, 0.0, "attack_tracking_ambiguous"
        changes.append(sum(motion) / len(motion))
    return min(changes), 0.35, "allied_advance_proxy"
