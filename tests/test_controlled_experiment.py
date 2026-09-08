from __future__ import annotations

import unittest
from types import SimpleNamespace

from crbot.controlled_experiment import ControlledExperiment


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


if __name__ == "__main__":
    unittest.main()
