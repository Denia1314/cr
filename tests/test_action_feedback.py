from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from crbot.action_feedback import action_feedback
from crbot.replay import ExperienceReplayRecorder
from crbot.replay_learning import (
    CONTEXT_FEATURE_COUNT,
    _deployment_examples, _feature, _local_arrays, _local_predict,
    collect_replay_learning_actions, ReplayPolicyRegistry, ReplayPolicyModel, train_replay_policy,
)
from crbot.vision import BattleResult
from tests import test_replay as replay_fixtures


def state(at: float, pressure: float = 0.6, proximity: float = 0.5, count: int = 2) -> dict:
    return {"schema": "action_observation_v1", "battle_elapsed_s": at,
            "left_threat": pressure, "left_threat_proximity": proximity, "left_unit_count": count}


def transition() -> dict:
    return {"state": state(10), "action": {"lane": "left", "formation_phase": "direct_defense"},
            "next_state": state(14, 0.3, 0.48, 1)}


class ActionFeedbackTests(unittest.TestCase):
    def test_losing_battle_can_have_helpful_defense_without_outcome_leakage(self):
        row = transition()
        row["outcome"] = "loss"
        reward, confidence, source = action_feedback(row)
        self.assertGreater(reward, 0)
        self.assertEqual(confidence, 0.2)
        row["outcome"] = "win"
        row["return_to_go"] = 999
        self.assertEqual(action_feedback(row), (reward, confidence, source))

    def test_enemy_advancing_is_negative_and_spending_alone_is_not_positive(self):
        row = transition()
        row["next_state"] = state(14, 0.8, 0.62, 2)
        self.assertLess(action_feedback(row)[0], 0)
        row["next_state"] = state(14)
        row["state"]["elixir"], row["next_state"]["elixir"] = 8, 3
        self.assertEqual(action_feedback(row)[0], 0)

    def test_single_frame_disappearance_and_near_tower_are_unknown(self):
        row = transition()
        row["next_state"] = state(14, 0, 0, 0)
        self.assertEqual(action_feedback(row)[1], 0)
        row["action_observations"] = [state(13, 0, 0, 0), state(14, 0, 0, 0)]
        self.assertGreater(action_feedback(row)[0], 0)
        self.assertEqual(action_feedback(row)[1], 0.5)
        row["state"]["left_threat_proximity"] = 0.7
        self.assertEqual(action_feedback(row)[1], 0)

    def test_too_early_late_missing_terminal_and_nonfinite_states_are_unknown(self):
        for end in (state(11), state(20), {"schema": "terminal_state_v1"}, {}, state(float("nan"))):
            row = transition()
            row["next_state"] = end
            self.assertEqual(action_feedback(row)[1], 0)

    def test_observations_after_next_action_cannot_credit_previous_card(self):
        row = transition()
        row["next_state"] = state(11)
        row["action_observations"] = [state(14, 0, 0, 0), state(15, 0, 0, 0)]
        self.assertEqual(action_feedback(row)[1], 0)

    def test_attack_requires_measured_allies_not_formation_memory(self):
        row = transition()
        row["action"]["formation_phase"] = "support_counterpush"
        row["action_observations"] = [state(13), state(14)]
        self.assertEqual(action_feedback(row)[2], "ally_detection_unavailable")
        ally = {"card_id": "tank", "lane": "left", "x": 0.3, "y": 0.6}
        row["state"].update(allies_observed=True, observed_allies=[ally])
        for sample in row["action_observations"]:
            sample.update(allies_observed=True, observed_allies=[{**ally, "y": 0.53}])
        self.assertGreater(action_feedback(row)[0], 0)
        self.assertEqual(action_feedback(row)[2], "allied_advance_proxy")

    def test_new_attacker_can_use_first_reliable_post_action_frame_as_baseline(self):
        row = transition()
        row["action"]["formation_phase"] = "support_counterpush"
        row["state"].update(allies_observed=False, observed_allies=[])
        ally = {"card_id": "tank", "lane": "left", "x": 0.3, "y": 0.62}
        row["action_observations"] = [
            {**state(13), "allies_observed": True, "observed_allies": [ally]},
            {**state(14), "allies_observed": True, "observed_allies": [{**ally, "y": 0.53}]},
        ]

        effect, confidence, source = action_feedback(row)

        self.assertGreater(effect, 0)
        self.assertEqual(confidence, 0.35)
        self.assertEqual(source, "allied_advance_proxy")

    def test_recorder_observes_holding_periods_but_not_other_actions_or_battles(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = ExperienceReplayRecorder(Path(folder), {"enabled": True})
            image = Image.new("RGB", (600, 1000))
            payload = {**replay_fixtures.action_payload(), "battle_elapsed_s": 10, "desired_formation_role": "backline"}
            recorder.record_action(1, image, payload, None)
            recorder.observe_action_effect(2, state(13))
            recorder.observe_action_effect(1, state(13))
            recorder.observe_action_effect(1, state(13.1))
            recorder.observe_action_effect(1, state(14))
            recorder.record_action(1, image, {**payload, "battle_elapsed_s": 15}, None)
            recorder.observe_action_effect(1, state(18))
            self.assertEqual(len(recorder.pending[0]["action_observations"]), 2)
            self.assertEqual(len(recorder.pending[1]["action_observations"]), 1)
            self.assertEqual(recorder.pending[0]["action"]["desired_formation_role"], "backline")
            recorder.finish_battle(1, BattleResult("loss", 0, 1, 1, (0, 0, 0), (1, 0, 0)), None)
            saved = [json.loads(line) for line in recorder.transitions_path.read_text().splitlines()]
            self.assertEqual(len(saved[0]["action_observations"]), 2)

    def test_good_local_moves_from_losses_can_teach_placement_without_future_feature_leakage(self):
        helper = replay_fixtures.ReplayPolicyLearningTests()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            catalog = helper._dataset(root)
            rows = collect_replay_learning_actions(root, catalog, helper._config(False))
            good = replace(rows[-1], local_effect=0.8, local_confidence=0.5)
            bad = replace(rows[0], local_effect=-0.5, local_confidence=0.2)
            self.assertEqual(_deployment_examples([good, bad], 0.15), [good])
            self.assertEqual(_deployment_examples([good, bad], 0), [bad])
            np.testing.assert_array_equal(_feature(good, catalog.by_id[good.card_id], 0),
                                          _feature(rows[-1], catalog.by_id[good.card_id], 0))
            x, y, weights = _local_arrays([good], [], catalog, 0, 0.25)
            self.assertAlmostEqual(float(y[0]), 0.4)

    def test_defense_only_feedback_does_not_score_attack(self):
        sample = np.zeros(CONTEXT_FEATURE_COUNT + 4)
        defensive = sample.copy()
        defensive[-CONTEXT_FEATURE_COUNT] = 1
        arrays = (np.asarray([defensive]), np.asarray([0.4]), np.asarray([1.0]))
        self.assertEqual(_local_predict(arrays, sample, 1), 0)
        self.assertAlmostEqual(_local_predict(arrays, defensive, 1), 0.4)

    def test_local_model_roundtrip_and_promotion_gate(self):
        helper = replay_fixtures.ReplayPolicyLearningTests()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            catalog = helper._dataset(root)
            result = train_replay_policy(root, catalog, helper._config(True))
            path = ReplayPolicyRegistry(root).champion_path()
            with np.load(path) as data:
                values = {k: data[k].copy() for k in data.files}
            model = ReplayPolicyModel(root)
            rows = collect_replay_learning_actions(root, catalog, helper._config(False))
            row = rows[-1]
            feature = _feature(row, catalog.by_id[row.card_id], model.visual_weight)
            values.update(local_x=np.asarray([feature]), local_y=np.asarray([0.4]),
                          local_sample_weights=np.asarray([1.0]), local_feedback_weight=np.asarray([0.15]))
            np.savez_compressed(path, **values)
            registry = ReplayPolicyRegistry(root)
            payload = registry.load()
            payload["champion"]["model_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            registry.path.write_text(json.dumps(payload), encoding="utf-8")
            weighted = ReplayPolicyModel(root)
            self.assertAlmostEqual(weighted.card_score(catalog.by_id[row.card_id], row.elixir, row.threats()), 0.06)
            candidate = ReplayPolicyRegistry(root).register(path, result["metrics"], helper._config(True), {"local_feedback_weight": 0.15})
            self.assertFalse(candidate["promoted"])
            self.assertTrue(any("短期效果" in reason for reason in candidate["rejection_reasons"]))


if __name__ == "__main__":
    unittest.main()
