"""Reproduce the remote replay audit and a local-only cross-generation trial.

Run from the project root: python tools/evaluate_replay_transfer.py --train
No GitHub writes or game actions. Results include the fixed V5-only baseline.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crbot.cards import CardCatalog
from crbot.replay_learning import (
    ReplayPolicyRegistry, collect_replay_learning_actions, train_replay_policy,
    _policy_version, _feature,
)
from crbot.training_sync import ReplaySync, exclusive, training_allowed
import numpy as np


def profile(root: Path, config: dict, catalog: CardCatalog) -> dict:
    records = list((root / ".training-sync/repository/records").glob("*/*/*.json"))
    raw = Counter()
    anomalies = Counter()
    identifiers = set()
    devices = Counter()
    for path in records:
        payload = json.loads(path.read_text(encoding="utf-8"))
        episode, rows = payload["episode"], payload["transitions"]
        raw[_policy_version(episode)] += 1
        devices[payload["device_id"]] += 1
        anomalies["action_count_mismatch"] += len(rows) != episode["action_count"]
        anomalies["duplicate_episode_id"] += episode["episode_id"] in identifiers
        identifiers.add(episode["episode_id"])
        if episode.get("reward_verified"):
            anomalies["verified_low_confidence"] += float(episode.get("confidence", 0)) < config.get("minimum_result_confidence", 0.75)
        for row in rows:
            anomalies["episode_transition_outcome_mismatch"] += row.get("outcome") != episode.get("outcome")
            anomalies["episode_transition_policy_mismatch"] += _policy_version(row) != _policy_version(episode)
            anomalies["episode_transition_verification_mismatch"] += bool(row.get("reward_verified")) != bool(episode.get("reward_verified"))
    actions = collect_replay_learning_actions(root, catalog, {})
    versions = {}
    for version in sorted({a.policy_version for a in actions}):
        rows = [a for a in actions if a.policy_version == version]
        outcomes = {a.group_id: a.outcome for a in rows}
        stamps = [a.timestamp_unix for a in rows if math.isfinite(a.timestamp_unix)]
        versions[version] = {
            "unique_usable_battles": len(outcomes), "usable_actions": len(rows),
            "wins": sum(v == "win" for v in outcomes.values()),
            "losses": sum(v == "loss" for v in outcomes.values()),
            "cards": len({a.card_id for a in rows}),
            "first_timestamp_utc": datetime.fromtimestamp(min(stamps), timezone.utc).isoformat(),
            "last_timestamp_utc": datetime.fromtimestamp(max(stamps), timezone.utc).isoformat(),
            "nonfinite_feature_rows": sum(not np.isfinite(_feature(a, catalog.by_id[a.card_id], 0.08)).all() for a in rows),
        }
    return {"remote_records": len(records), "remote_policy_counts": dict(raw),
            "remote_device_counts": dict(devices), "integrity_checks": dict(anomalies),
            "usable_after_content_deduplication": versions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))["replay"]
    catalog = CardCatalog.load(ROOT / "data/cards.json")
    evidence = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "source_repository": ReplaySync(ROOT).config.get("repository"),
                "remote_commit": ReplaySync(ROOT).git("rev-parse", "origin/main"),
                "config": config, "profile": profile(ROOT, config, catalog)}
    print(json.dumps(evidence["profile"], ensure_ascii=False, indent=2), flush=True)
    if args.train:
        if any(evidence["profile"]["integrity_checks"].values()) or any(
            row["nonfinite_feature_rows"]
            for row in evidence["profile"]["usable_after_content_deduplication"].values()
        ):
            raise ValueError("数据一致性或特征有效性检查失败，停止训练")
        if not training_allowed(ROOT):
            raise ValueError("本机不是训练机")
        with exclusive(ROOT / ".training-sync/training.lock"):
            # The user requested reading remote data. Bypass the network sync
            # wrapper (which uploads models) while retaining the training lock.
            evidence["training"] = train_replay_policy.__wrapped__(ROOT, catalog, config)
        candidate = evidence["training"]["candidate"]
        print(json.dumps({k: candidate[k] for k in (
            "version", "quality_passed", "promoted", "metrics", "rejection_reasons", "promotion_blockers"
        )}, ensure_ascii=False, indent=2), flush=True)
    evidence["active_champion"] = ReplayPolicyRegistry(ROOT).champion()
    output = ROOT / "reports" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_replay_transfer.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Evidence: {output}", flush=True)


if __name__ == "__main__":
    main()
