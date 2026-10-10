"""Audit calibrated action values on a frozen replay snapshot; optionally train a candidate."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def audit_snapshot(snapshot, catalog, config):
    from crbot.action_value import train_head
    from crbot.replay_learning import (collect_replay_learning_actions, _feature,
                                       _stratified_group_split, _temporal_group_split)
    from crbot.replay_evaluation import reserve_frozen_groups
    replay = config["replay"]
    actions = collect_replay_learning_actions(snapshot, catalog, replay)
    seed = int(replay.get("training_seed", 20260903))
    frozen = reserve_frozen_groups({a.group_id for a in actions}, validation_fraction=0.,
                                  freeze_fraction=float(replay.get("freeze_evaluation_fraction", 0.)), seed=seed)
    trainable = [a for a in actions if a.group_id not in set(frozen.frozen)]
    train, validation = _stratified_group_split(trainable, float(replay.get("validation_fraction", .25)), seed)
    require_temporal = bool(replay.get("require_temporal_validation", False))
    temporal_train, temporal_validation = _temporal_group_split(
        trainable, float(replay.get("temporal_validation_fraction", .25))) if require_temporal else ([], [])
    _, report = train_head(trainable, train, validation, temporal_train, temporal_validation,
                          lambda a: _feature(replace(a, formation_phase="", desired_role=""),
                                             catalog.by_id[a.card_id], float(replay.get("battlefield_visual_weight", .08))),
                          require_temporal=require_temporal, seed=seed)
    report.update(frozen_battles=list(frozen.frozen), observed_actions=len(actions),
                  target=replay.get("training_policy_version"), candidate_generated=False, promoted=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path)
    parser.add_argument("--train-candidate", action="store_true", help="Register a candidate only; never promote")
    args = parser.parse_args()
    root = args.root.resolve()
    report_path = args.report or root / "reports/action-value-audit.json"
    from crbot.cards import CardCatalog
    from crbot.config import load_config
    from crbot.learning_store import atomic_json
    from crbot.self_learning import effective_learning_config, freeze_snapshot, request_config, verify_snapshot
    from crbot.replay_learning import train_replay_policy
    config, path = load_config(root / "config.json")
    config = effective_learning_config(config, path)
    request = request_config(root, config)
    with tempfile.TemporaryDirectory(prefix="royal-action-audit-") as directory:
        snapshot = Path(directory)
        freeze_snapshot(root, snapshot, request)
        verify_snapshot(snapshot)
        catalog = CardCatalog.load(snapshot / "data/cards.json")
        print("Frozen replay ready; evaluating action values...", flush=True)
        report = audit_snapshot(snapshot, catalog, config)
        atomic_json(report_path, report)
        print(json.dumps({k: report[k] for k in ("enabled", "recipe", "observed_actions", "target")}), flush=True)
        if args.train_candidate:
            print("Training complete replay candidate from the same snapshot...", flush=True)
            result = train_replay_policy(root, catalog, config["replay"], candidate_only=True,
                                         snapshot_root=snapshot,
                                         learning_manifest={"code_hash": request["code_hash"],
                                                            "recipe": "action_value_calibration_audit_v2"})
            candidate = result["candidate"]
            report.update(candidate_generated=True, candidate_version=candidate["version"],
                          candidate_quality_passed=candidate.get("quality_passed"),
                          promoted=candidate.get("promoted", False))
            atomic_json(report_path, report)
            print(json.dumps({k: report[k] for k in ("candidate_version", "candidate_quality_passed", "promoted")}), flush=True)


if __name__ == "__main__":
    main()
