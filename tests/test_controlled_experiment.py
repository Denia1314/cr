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
        policy = {"experiment_arm": "candidate"}
        unknown = {"reward_verified": False, "action_count": 2, "confirmed_action_count": 2, "policy": policy}
        verified = {"reward_verified": True, "action_count": 2, "confirmed_action_count": 2, "policy": policy}

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
            {"reward_verified": True, "action_count": 4, "confirmed_action_count": 3, "policy": {"experiment_arm": "candidate"}}
        )
        self.assertEqual(reason, "unconfirmed_action_rate=0.250")

    def test_baseline_does_not_trigger_candidate_guardrail(self) -> None:
        experiment = ControlledExperiment(
            {
                "enabled": True,
                "minimum_battles_before_guardrail": 1,
                "maximum_unknown_result_rate": 0.0,
                "maximum_unconfirmed_action_rate": 0.0,
            },
            None,
        )
        reason = experiment.observe(
            {
                "reward_verified": False,
                "action_count": 4,
                "confirmed_action_count": 0,
                "policy": {"experiment_arm": "baseline"},
            }
        )
        self.assertEqual(reason, "")

    def test_baseline_does_not_repeat_a_historical_candidate_stop(self) -> None:
        prior = [
            {
                "reward_verified": True,
                "action_count": 4,
                "confirmed_action_count": 0,
                "policy": {"experiment_arm": "candidate"},
            }
        ]
        experiment = ControlledExperiment(
            {
                "enabled": True,
                "minimum_battles_before_guardrail": 1,
                "maximum_unknown_result_rate": 0.0,
                "maximum_unconfirmed_action_rate": 0.0,
            },
            None,
            prior_episodes=prior,
        )

        reason = experiment.observe(
            {
                "reward_verified": True,
                "action_count": 3,
                "confirmed_action_count": 0,
                "policy": {"experiment_arm": "baseline"},
            }
        )

        self.assertEqual(reason, "")
        self.assertEqual(experiment.stop_reason, "")

    def test_guardrail_counts_only_the_current_candidate_batch(self) -> None:
        prior = [
            {
                "reward_verified": True,
                "action_count": 4,
                "confirmed_action_count": 0,
                "policy": {
                    "experiment_arm": "candidate",
                    "experiment_batch_index": 1,
                },
            }
            for _ in range(10)
        ]
        experiment = ControlledExperiment(
            {
                "enabled": True,
                "minimum_battles_before_guardrail": 2,
                "maximum_unknown_result_rate": 0.0,
                "maximum_unconfirmed_action_rate": 0.0,
            },
            None,
            prior_episodes=prior,
        )
        new_batch = {
            "reward_verified": True,
            "action_count": 4,
            "confirmed_action_count": 4,
            "policy": {
                "experiment_arm": "candidate",
                "experiment_batch_index": 3,
            },
        }

        self.assertEqual(experiment.observe(new_batch), "")
        self.assertEqual(experiment.observe(new_batch), "")

    def test_relative_confirmation_guardrail_compares_previous_baseline(self) -> None:
        baseline = [
            {
                "timestamp_unix": index,
                "action_count": 10,
                "confirmed_action_count": 6,
                "policy": {
                    "experiment_arm": "baseline",
                    "experiment_batch_index": 2,
                    "experiment_battle_index": 21 + index,
                },
            }
            for index in range(2)
        ]
        experiment = ControlledExperiment(
            {
                "enabled": True,
                "minimum_battles_before_guardrail": 2,
                "maximum_unknown_result_rate": 1.0,
                "maximum_confirmation_rate_drop_vs_baseline": 0.05,
            },
            None,
            prior_episodes=baseline,
        )
        candidate = {
            "reward_verified": True,
            "action_count": 10,
            "confirmed_action_count": 5,
            "policy": {
                "experiment_arm": "candidate",
                "experiment_batch_index": 3,
            },
        }

        self.assertEqual(experiment.observe(candidate), "")
        self.assertEqual(
            experiment.observe(candidate),
            "confirmation_rate_drop_vs_baseline=0.100",
        )

    def test_guardrail_freezes_reused_recorder_policy_between_battles(self) -> None:
        experiment = ControlledExperiment(
            {
                "enabled": True,
                "minimum_battles_before_guardrail": 2,
                "maximum_unknown_result_rate": 1.0,
                "maximum_confirmation_rate_drop_vs_baseline": 0.05,
            },
            None,
        )
        shared_policy = {"experiment_arm": "baseline", "experiment_batch_index": 2}
        for index in range(2):
            shared_policy["experiment_battle_index"] = 21 + index
            self.assertEqual(experiment.observe({
                "reward_verified": True, "action_count": 10,
                "confirmed_action_count": 6, "policy": shared_policy,
            }), "")
        shared_policy.update(experiment_arm="candidate", experiment_batch_index=3)
        for index in range(2):
            shared_policy["experiment_battle_index"] = 31 + index
            reason = experiment.observe({
                "reward_verified": True, "action_count": 10,
                "confirmed_action_count": 5, "policy": shared_policy,
            })
        self.assertEqual(reason, "confirmation_rate_drop_vs_baseline=0.100")
        self.assertEqual(
            [row["policy"]["experiment_battle_index"] for row in experiment.episodes],
            [21, 22, 31, 32],
        )

    def test_prior_episodes_resume_absolute_batch_position(self) -> None:
        model = SimpleNamespace(
            influence_scale=0.1,
            available=True,
            champion={"version": "candidate-1"},
        )
        prior = [
            {"policy": {"experiment_arm": "baseline"}}
            for _ in range(7)
        ]
        experiment = ControlledExperiment(
            {"enabled": True, "batch_size": 10},
            model,
            prior_episodes=prior,
        )

        self.assertEqual(experiment.activate(1)["experiment_battle_index"], 8)
        self.assertEqual(experiment.activate(1)["experiment_arm"], "baseline")
        self.assertEqual(experiment.activate(4)["experiment_arm"], "candidate")

    def test_restart_excess_baselines_still_get_full_candidate_batch(self) -> None:
        model = SimpleNamespace(
            influence_scale=0.1,
            available=True,
            champion={"version": "candidate-1"},
        )
        prior = [
            {"policy": {"experiment_arm": "baseline"}}
            for _ in range(18)
        ]
        experiment = ControlledExperiment(
            {"enabled": True, "batch_size": 10},
            model,
            prior_episodes=prior,
        )

        self.assertEqual(experiment.battle_offset, 10)
        self.assertEqual(experiment.activate(1)["experiment_arm"], "candidate")
        self.assertEqual(experiment.activate(10)["experiment_arm"], "candidate")
        self.assertEqual(experiment.activate(11)["experiment_arm"], "baseline")

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

            self.assertFalse(audit["ready_for_comparison"])
            self.assertEqual(len(audit["comparison"]["groups"]), 1)
            self.assertIn("code_commit", audit["comparison"]["groups"][0]["missing_identity_fields"])
            self.assertEqual(audit["arms"]["baseline"]["win_rate"], 1.0)
            self.assertEqual(audit["arms"]["candidate"]["confirmation_rate"], 0.75)

            rows[0]["policy"]["runtime_replay_influence_scale"] = 0.1
            (run / "replay_episodes.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
            )
            contaminated = audit_controlled_experiment(root)
            self.assertFalse(contaminated["ready_for_comparison"])
            self.assertIn("base-1:baseline_nonzero_scale", contaminated["contamination"])

    def test_audit_deduplicates_repeated_experiment_indices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs" / "one"
            run.mkdir(parents=True)
            policy = {
                "experiment_enabled": True,
                "experiment_arm": "candidate",
                "experiment_batch_index": 1,
                "experiment_battle_index": 11,
                "runtime_replay_version": "model-1",
                "runtime_replay_loaded": True,
                "runtime_replay_influence_scale": 0.1,
            }
            rows = [
                {
                    "episode_id": "first",
                    "timestamp_unix": 1.0,
                    "reward_verified": True,
                    "outcome": "win",
                    "action_count": 2,
                    "confirmed_action_count": 2,
                    "policy": policy,
                },
                {
                    "episode_id": "duplicate",
                    "timestamp_unix": 2.0,
                    "reward_verified": True,
                    "outcome": "win",
                    "action_count": 2,
                    "confirmed_action_count": 2,
                    "policy": policy,
                },
            ]
            (run / "replay_episodes.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            audit = audit_controlled_experiment(root)

            self.assertEqual(audit["raw_episodes"], 2)
            self.assertEqual(audit["episodes"], 1)
            self.assertEqual(audit["arms"]["candidate"]["episodes"], 1)
            self.assertEqual(len(audit["duplicate_experiment_indices"]), 1)
            self.assertEqual(audit["comparison"]["conflicting_indices"], ["model-1:11"])
            self.assertEqual(
                [item["reason"] for item in audit["comparison"]["exclusions"]],
                ["conflicting_experiment_index", "conflicting_experiment_index"],
            )

    def test_comparison_separates_protocols_and_requires_complete_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs" / "one"
            run.mkdir(parents=True)
            identity = {
                "experiment_enabled": True, "runtime_replay_version": "model-1",
                "mode": "reactive", "runtime_model": "m3", "rule_version": "rule-1",
                "imitation_version": "imitation-1", "code_commit": "commit-1",
                "config_sha256": "config-1", "replay_model_sha256": "hash-1",
                "deck_id": "deck-1", "battle_mode": "ranked",
                "environment_id": "environment-1", "runtime_replay_loaded": True,
            }
            rows = []
            for guardrail, offset in (("old", 0), ("new", 20)):
                for arm, index, batch, scale in (("baseline", 1, 0, 0.0), ("candidate", 11, 1, 0.1)):
                    for number in range(10):
                        rows.append({
                            "episode_id": f"{guardrail}-{arm}-{number}", "reward_verified": True,
                            "outcome": "win", "policy": {
                                **identity, "experiment_guardrail_version": guardrail,
                                "experiment_arm": arm, "experiment_battle_index": index + offset + number,
                                "experiment_batch_index": batch + offset // 10,
                                "runtime_replay_influence_scale": scale,
                            },
                        })
            (run / "replay_episodes.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
            )
            audit = audit_controlled_experiment(root)
            self.assertTrue(audit["ready_for_comparison"])
            self.assertEqual(len(audit["comparison"]["groups"]), 2)
            self.assertTrue(all(group["ready_for_comparison"] for group in audit["comparison"]["groups"]))
            self.assertEqual([group["complete_batch_pairs"] for group in audit["comparison"]["groups"]], [[[0, 1]], [[2, 3]]])


if __name__ == "__main__":
    unittest.main()
