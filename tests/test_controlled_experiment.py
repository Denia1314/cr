from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from crbot.controlled_experiment import ControlledExperiment, audit_controlled_experiment


class ControlledExperimentTests(unittest.TestCase):
    def test_assigns_whole_batches_in_alternating_order(self) -> None:
        model = SimpleNamespace(influence_scale=0.1, available=True, champion={"version": "candidate-1"})
        experiment = ControlledExperiment({"enabled": True, "batch_size": 2}, model)

        arms = [experiment.activate(index)["experiment_arm"] for index in range(1, 6)]

        self.assertEqual(arms, ["baseline", "baseline", "candidate", "candidate", "baseline"])
        self.assertEqual(model.influence_scale, 0.0)

    def test_disabled_experiment_does_not_change_model_influence(self) -> None:
        model = SimpleNamespace(influence_scale=0.15, available=True, champion={})
        metadata = ControlledExperiment({"enabled": False}, model).activate(1)
        self.assertEqual(model.influence_scale, 0.15)
        self.assertEqual(metadata["experiment_arm"], "normal")

    def test_guardrail_stops_only_after_minimum_complete_battles(self) -> None:
        experiment = ControlledExperiment(
            {
                "enabled": True,
                "minimum_battles_before_guardrail": 2,
                "maximum_unknown_result_rate": 0.25,
                "maximum_unconfirmed_action_rate": 0.25,
            },
            None,
        )
        unknown = {"reward_verified": False, "action_count": 2, "confirmed_action_count": 2}
        verified = {"reward_verified": True, "action_count": 2, "confirmed_action_count": 2}

        self.assertEqual(experiment.observe(unknown), "")
        self.assertEqual(experiment.observe(verified), "unknown_result_rate=0.500")

    def test_unconfirmed_action_guardrail_has_explicit_reason(self) -> None:
        experiment = ControlledExperiment(
            {
                "enabled": True,
                "minimum_battles_before_guardrail": 1,
                "maximum_unknown_result_rate": 1.0,
                "maximum_unconfirmed_action_rate": 0.10,
            },
            None,
        )
        reason = experiment.observe(
            {"reward_verified": True, "action_count": 4, "confirmed_action_count": 3}
        )
        self.assertEqual(reason, "unconfirmed_action_rate=0.250")

    def test_rollback_zeros_candidate_weight_and_records_reason(self) -> None:
        model = SimpleNamespace(influence_scale=0.1)
        experiment = ControlledExperiment({"enabled": True}, model)
        experiment.stop_reason = "unknown_result_rate=0.200"

        metadata = experiment.rollback_to_baseline()

        self.assertEqual(model.influence_scale, 0.0)
        self.assertEqual(metadata["experiment_arm"], "baseline")
        self.assertEqual(metadata["rollback_reason"], "unknown_result_rate=0.200")

    def test_audit_reports_arms_and_detects_contamination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs" / "one"
            run.mkdir(parents=True)
            rows = [
                {
                    "episode_id": "base-1", "reward_verified": True, "outcome": "win",
                    "action_count": 2, "confirmed_action_count": 2,
                    "policy": {"experiment_enabled": True, "experiment_arm": "baseline", "runtime_replay_loaded": True, "runtime_replay_influence_scale": 0.0},
                },
                {
                    "episode_id": "candidate-1", "reward_verified": True, "outcome": "loss",
                    "action_count": 4, "confirmed_action_count": 3,
                    "policy": {"experiment_enabled": True, "experiment_arm": "candidate", "runtime_replay_loaded": True, "runtime_replay_influence_scale": 0.1},
                },
            ]
            (run / "replay_episodes.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
            )

            audit = audit_controlled_experiment(root)

            self.assertTrue(audit["ready_for_comparison"])
            self.assertEqual(audit["arms"]["baseline"]["win_rate"], 1.0)
            self.assertEqual(audit["arms"]["candidate"]["confirmation_rate"], 0.75)

            rows[0]["policy"]["runtime_replay_influence_scale"] = 0.1
            (run / "replay_episodes.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
            )
            contaminated = audit_controlled_experiment(root)
            self.assertFalse(contaminated["ready_for_comparison"])
            self.assertIn("base-1:baseline_nonzero_scale", contaminated["contamination"])


if __name__ == "__main__":
    unittest.main()
