from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from crbot.battle_perception import LaneThreat
from crbot.cards import CardCatalog, CardDefinition
from crbot.replay import ExperienceReplayRecorder, audit_replay
from crbot.replay_learning import (
    CONTEXT_FEATURE_COUNT,
    TACTICAL_FEATURE_COUNT,
    ReplayPolicyRegistry,
    ReplayPolicyModel,
    _tactical_features,
    _context_features,
    _deduplicate_battles,
    _transfer_before,
    _transfer_actions,
    _training_arrays,
    _balanced_knn_predict,
    _temperature_scale_probability,
    _battle_cluster_intervals,
    _validation_segments,
    collect_replay_learning_actions,
    audit_replay_learning,
    train_replay_policy,
)
from crbot.vision import BattleResult, detect_battle_result


def result_screen(player_crowns: int, opponent_crowns: int) -> Image.Image:
    height, width = 1000, 600
    data = np.full((height, width, 3), (18, 25, 38), dtype=np.uint8)
    x_ranges = ((0.14, 0.36), (0.39, 0.61), (0.64, 0.86))
    for index in range(opponent_crowns):
        left, right = x_ranges[index]
        data[125:245, int(left * width) : int(right * width)] = (230, 170, 30)
    for index in range(player_crowns):
        left, right = x_ranges[index]
        data[445:565, int(left * width) : int(right * width)] = (230, 170, 30)
    image = Image.fromarray(data)
    template_path = Path(__file__).resolve().parents[1] / "templates" / "result_continue.png"
    with Image.open(template_path) as source:
        scale = width / 1080.0
        template = source.convert("RGB").resize(
            (round(source.width * scale), round(source.height * scale)),
            Image.Resampling.LANCZOS,
        )
    image.paste(template, (round(width / 2 - template.width / 2), 872))
    return image


def action_payload(slot_index: int = 1) -> dict[str, object]:
    return {
        "slot_index": slot_index,
        "card_id": "synthetic-card",
        "deploy_point": [0.3, 0.7],
        "lane": "left",
        "reason": "defend_left",
        "elixir": 6.0,
        "elixir_source": "estimated",
        "hand": ["a", "b", "c", "d"],
        "left_threat": 0.7,
        "right_threat": 0.1,
        "left_threat_type": "heavy",
        "right_threat_type": "none",
        "left_threat_proximity": 0.48,
        "left_unit_count": 2,
        "battle_elapsed_s": 42.0,
        "threat_type": "single",
        "enemy_cards": [],
        "left_threat_unit_layers": ["air"],
        "right_threat_unit_layers": [],
        "threat_unit_layers": ["air"],
        "left_threat_layer_confidence": 0.91,
        "right_threat_layer_confidence": 0.0,
        "threat_layer_confidence": 0.91,
        "card_attack_targets": ["ground", "air"],
        "card_targeting_source": "catalog",
    }


class BattleResultTests(unittest.TestCase):
    def test_detects_player_win_from_fixed_crown_slots(self) -> None:
        result = detect_battle_result(result_screen(2, 1))

        self.assertEqual(result.outcome, "win")
        self.assertEqual(result.player_crowns, 2)
        self.assertEqual(result.opponent_crowns, 1)
        self.assertGreaterEqual(result.confidence, 0.75)

    def test_detects_player_loss_from_fixed_crown_slots(self) -> None:
        result = detect_battle_result(result_screen(0, 3))

        self.assertEqual(result.outcome, "loss")
        self.assertEqual(result.player_crowns, 0)
        self.assertEqual(result.opponent_crowns, 3)

    def test_requires_result_confirmation_button(self) -> None:
        image = result_screen(2, 1)
        data = np.asarray(image).copy()
        data[780:960] = (18, 25, 38)

        result = detect_battle_result(Image.fromarray(data))

        self.assertEqual(result.outcome, "unknown")
        self.assertEqual(result.confidence, 0.0)


class ExperienceReplayTests(unittest.TestCase):
    def test_writes_discounted_verified_episode_without_enabling_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "run-1"
            recorder = ExperienceReplayRecorder(
                run_dir,
                {
                    "enabled": True,
                    "allow_bot_training": False,
                    "gamma": 0.9,
                    "crown_difference_reward": 0.1,
                },
                policy_metadata={"mode": "test"},
            )
            image = Image.new("RGB", (600, 1000), (20, 30, 40))
            recorder.start_battle(1)
            recorder.record_action(1, image, action_payload(0), "frames/one.jpg")
            recorder.record_action(1, image, action_payload(1), "frames/two.jpg")
            episode = recorder.finish_battle(
                1,
                BattleResult("win", 2, 1, 1.0, (1.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
                "frames/result.jpg",
            )

            self.assertIsNotNone(episode)
            assert episode is not None
            self.assertEqual(episode["terminal_reward"], 1.1)
            rows = [
                json.loads(line)
                for line in (run_dir / "replay_transitions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["reward"], 0.0)
            self.assertEqual(rows[0]["return_to_go"], 0.99)
            self.assertEqual(rows[0]["state"]["left_threat_type"], "heavy")
            self.assertEqual(rows[0]["state"]["left_unit_count"], 2)
            self.assertEqual(rows[0]["state"]["battle_elapsed_s"], 42.0)
            self.assertEqual(rows[0]["state"]["threat_unit_layers"], ["air"])
            self.assertAlmostEqual(
                rows[0]["state"]["threat_layer_confidence"], 0.91
            )
            self.assertEqual(
                rows[0]["action"]["card_attack_targets"], ["ground", "air"]
            )
            self.assertEqual(rows[0]["action"]["card_targeting_source"], "catalog")
            self.assertFalse(rows[0]["done"])
            self.assertEqual(rows[1]["reward"], 1.1)
            self.assertTrue(rows[1]["done"])
            self.assertTrue(rows[1]["reward_verified"])
            self.assertFalse(rows[1]["eligible_for_training"])

            audit = audit_replay(
                root,
                {
                    "allow_bot_training": False,
                    "minimum_verified_episodes": 1,
                    "minimum_wins": 1,
                    "minimum_losses": 0,
                    "minimum_verified_transitions": 2,
                },
            )
            self.assertEqual(audit.verified_episodes, 1)
            self.assertEqual(audit.verified_transitions, 2)
            self.assertTrue(audit.collection_ready)
            self.assertEqual(audit.training_eligible_transitions, 0)
            self.assertEqual(audit.policy_breakdown["test"]["wins"], 1)
            self.assertEqual(
                audit.policy_breakdown["test"]["verified_transitions"], 2
            )

    def test_uncertain_result_is_retained_but_never_training_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "run-1"
            recorder = ExperienceReplayRecorder(
                run_dir,
                {"enabled": True, "allow_bot_training": True},
            )
            image = Image.new("RGB", (600, 1000), (20, 30, 40))
            recorder.start_battle(1)
            recorder.record_action(1, image, action_payload(), "frames/one.jpg")
            recorder.finish_battle(
                1,
                BattleResult(
                    "unknown",
                    None,
                    None,
                    0.0,
                    (0.1, 0.1, 0.1),
                    (0.1, 0.1, 0.1),
                ),
                "frames/result.jpg",
            )

            row = json.loads(
                (run_dir / "replay_transitions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(row["reward"], 0.0)
            self.assertFalse(row["reward_verified"])
            self.assertFalse(row["eligible_for_training"])


class ReplayPolicyLearningTests(unittest.TestCase):
    @staticmethod
    def _card(card_id: str, kind: str, roles: tuple[str, ...]) -> CardDefinition:
        return CardDefinition(
            card_id=card_id,
            official_id=None,
            name_zh="",
            name_en=card_id,
            elixir=4,
            rarity="common",
            max_level=None,
            icon_url="",
            icon_path="",
            icon_variants=(),
            kind=kind,
            targets=(),
            roles=roles,
            counters=(),
            synergies=(),
        )

    def _dataset(self, root: Path) -> CardCatalog:
        catalog = CardCatalog(
            [
                self._card("tank", "troop", ("tank", "win_condition")),
                self._card("spell", "spell", ("spell", "splash")),
            ]
        )
        run = root / "runs" / "learning-run"
        run.mkdir(parents=True)
        rows: list[dict[str, object]] = []
        for battle_index in range(1, 9):
            outcome = "win" if battle_index <= 4 else "loss"
            selected = "tank" if outcome == "win" else "spell"
            point = [0.3, 0.7] if outcome == "win" else [0.7, 0.55]
            for action_index in range(2):
                rows.append(
                    {
                        "schema_version": 1,
                        "transition_id": f"b{battle_index}-a{action_index}",
                        "reward_verified": True,
                        "outcome": outcome,
                        "return_to_go": 1.0 if outcome == "win" else -1.0,
                        "battle_index": battle_index,
                        "policy": {"rule_version": "v4"},
                        "state": {
                            "elixir": 8.0,
                            "hand": ["tank", "spell", None, None],
                            "left_threat": 0.1,
                            "right_threat": 0.0,
                            "threat_type": "single",
                            "threat_proximity": 0.3,
                            "threat_approach_rate": 0.0,
                            "battlefield_edges": [0.0] * 96,
                        },
                        "action": {
                            "card_id": selected,
                            "slot_index": 0 if selected == "tank" else 1,
                            "deploy_point": point,
                            "lane": "left",
                        },
                    }
                )
        with (run / "replay_transitions.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return catalog

    @staticmethod
    def _config(allow: bool) -> dict[str, object]:
        return {
            "allow_bot_training": allow,
            "training_policy_version": "v4",
            "minimum_verified_episodes": 4,
            "minimum_wins": 2,
            "minimum_losses": 2,
            "minimum_verified_transitions": 8,
            "minimum_distinct_cards_for_training": 2,
            "validation_fraction": 0.25,
            "training_seed": 7,
            "training_neighbors": 1,
            "battlefield_visual_weight": 0.0,
            "minimum_validation_episodes": 2,
            "minimum_validation_wins": 1,
            "minimum_validation_losses": 1,
            "minimum_balanced_accuracy": 0.9,
            "minimum_auc": 0.9,
            "minimum_brier_improvement": 0.1,
            "minimum_rank_separation": 0.9,
            "maximum_win_deploy_mae": 0.01,
            "assisted_policy_weight_scale": 0.1,
        }

    def test_candidate_stays_shadow_until_bot_training_is_explicitly_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = self._dataset(root)
            config = self._config(False)

            audit = audit_replay_learning(root, catalog, config)
            result = train_replay_policy(root, catalog, config)

            self.assertTrue(audit.ready)
            self.assertIsInstance(audit.feedback_source_counts, dict)
            self.assertIsInstance(audit.feedback_stage_counts, dict)
            self.assertGreaterEqual(audit.valid_feedback_actions, 0)
            self.assertGreaterEqual(audit.valid_feedback_battles, 0)
            self.assertTrue(result["candidate"]["quality_passed"])
            self.assertEqual(result["candidate"]["status"], "shadow_pass")
            self.assertFalse(result["candidate"]["promoted"])
            self.assertFalse(ReplayPolicyModel(root).available)
            self.assertTrue(
                result["candidate"]["manifest"]["validation_groups_are_disjoint"]
            )

    def test_valid_candidate_can_promote_and_score_unseen_hand_by_role(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = self._dataset(root)

            result = train_replay_policy(root, catalog, self._config(True))
            model = ReplayPolicyModel(root)

            self.assertTrue(result["candidate"]["promoted"])
            self.assertEqual(
                result["metrics"]["outcome_metric_unit"],
                "whole_battle_mean_prediction",
            )
            self.assertEqual(
                result["candidate"]["manifest"]["training_weight_unit"],
                "one_total_weight_per_current_battle",
            )
            self.assertTrue(model.available)
            self.assertEqual(model.tactical_feature_count, TACTICAL_FEATURE_COUNT)
            np.testing.assert_array_equal(
                model.value_sample_weights,
                np.full(len(model.value_y), 0.5, dtype=np.float32),
            )
            quiet = LaneThreat("left", 0.0, 0, 0.0, "none", ())
            threats = {"left": quiet, "right": LaneThreat("right", 0.0, 0, 0.0, "none", ())}
            tank = catalog.get("tank")
            spell = catalog.get("spell")
            assert tank is not None and spell is not None
            self.assertGreater(
                model.card_score(tank, 8.0, threats),
                model.card_score(spell, 8.0, threats),
            )

    def test_replay_tactical_features_apply_to_unseen_building_target_card(self) -> None:
        card = CardDefinition(
            card_id="unseen_building_target",
            official_id=None,
            name_zh="",
            name_en="Unseen Building Target",
            elixir=6,
            rarity="",
            max_level=None,
            icon_url="",
            icon_path="",
            icon_variants=(),
            kind="troop",
            targets=("buildings",),
            roles=(),
            counters=(),
            synergies=(),
        )
        quiet = LaneThreat("left", 0.0, 0, 0.0, "none", ())
        threats = {
            "left": quiet,
            "right": LaneThreat("right", 0.0, 0, 0.0, "none", ()),
        }

        attack = _tactical_features(card, threats, "form_frontline", "frontline")
        defense = _tactical_features(card, threats, "counter_defense", "frontline")

        self.assertEqual(len(attack), TACTICAL_FEATURE_COUNT)
        self.assertEqual(attack[2], 1.0)
        self.assertGreater(attack[16], 0.0)
        self.assertEqual(defense[16], 0.0)
        self.assertGreater(defense[15], 0.0)

    def test_replay_context_features_distinguish_global_lane_pressure(self) -> None:
        single_lane = {
            "left": LaneThreat("left", 0.8, 4, 0.9, "swarm", (), approach_rate=0.16),
            "right": LaneThreat("right", 0.0, 0, 0.0, "none", ()),
        }
        dual_lane = {
            "left": LaneThreat("left", 0.5, 2, 0.7, "swarm", (), approach_rate=0.08),
            "right": LaneThreat("right", 0.5, 2, 0.7, "heavy", (), approach_rate=0.08),
        }

        single = _context_features("counter_defense", "frontline", 90.0, single_lane)
        dual = _context_features("counter_defense", "frontline", 90.0, dual_lane)

        self.assertEqual(len(single), CONTEXT_FEATURE_COUNT)
        self.assertEqual(single[:6], dual[:6])
        self.assertNotEqual(single[6:], dual[6:])
        self.assertEqual(single[-1], 0.0)
        self.assertEqual(dual[-1], 1.0)

    def test_temperature_scaling_preserves_neutral_order_and_endpoints(self) -> None:
        self.assertEqual(_temperature_scale_probability(0.0, 0.5), 0.0)
        self.assertEqual(_temperature_scale_probability(0.5, 0.5), 0.5)
        self.assertEqual(_temperature_scale_probability(1.0, 0.5), 1.0)
        self.assertGreater(
            _temperature_scale_probability(0.7, 0.5),
            _temperature_scale_probability(0.7, 1.0),
        )

    def test_copied_device_battles_are_deduplicated_before_splitting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = self._dataset(root)
            original = collect_replay_learning_actions(root, catalog, self._config(False))[:2]
            original = [replace(a, timestamp_unix=100.0 + i) for i, a in enumerate(original)]
            copied = [replace(a, group_id="another-device:battle-1", transition_id="copy:" + a.transition_id)
                      for a in original]
            self.assertEqual(_deduplicate_battles(original + copied), original)
            later = [replace(a, timestamp_unix=a.timestamp_unix + 300) for a in copied]
            self.assertEqual(len(_deduplicate_battles(original + later)), 4)
            untimed = [replace(a, timestamp_unix=0) for a in original + copied]
            self.assertEqual(len(_deduplicate_battles(untimed)), 4)

    def test_temporal_transfer_excludes_future_and_crossing_whole_battles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = self._dataset(root)
            row = collect_replay_learning_actions(root, catalog, self._config(False))[0]
            validation = [replace(row, group_id="holdout", timestamp_unix=100)]
            old = replace(row, group_id="old", timestamp_unix=90)
            crossing = [replace(row, group_id="crossing", timestamp_unix=t) for t in (99, 101)]
            missing = replace(row, group_id="missing", timestamp_unix=0)
            self.assertEqual(_transfer_before([old, *crossing, missing], validation), [old])

    def test_transfer_preserves_v5_holdout_and_uses_only_v5_deployments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = self._dataset(root)
            source = root / "runs/learning-run/replay_transitions.jsonl"
            rows = [json.loads(line) for line in source.read_text().splitlines()]
            current = root / "runs/current"
            current.mkdir()
            for row in rows:
                row["transition_id"] = "current:" + row["transition_id"]
                row["policy"] = {"rule_version": "v5"}
            (current / "replay_transitions.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
            )
            config = {**self._config(False), "training_policy_version": "v5",
                      "transfer_policy_versions": ["v4"]}
            self.assertEqual({a.policy_version for a in _transfer_actions(root, catalog, config)}, {"v4"})
            result = train_replay_policy(root, catalog, config)
            manifest = result["candidate"]["manifest"]
            self.assertEqual(manifest["transfer_actions"], 16)
            self.assertTrue(all("current:" in g for g in manifest["validation_battles"]))
            self.assertFalse(set(manifest["transfer_battles"]) & set(manifest["validation_battles"]))
            with np.load(root / "models/replay_policy" / result["candidate"]["model_path"]) as data:
                self.assertEqual(len(data["value_y"]), 32)
                self.assertEqual(len(data["deploy_y"]), 8)
                self.assertAlmostEqual(float(data["value_sample_weights"][-1]), 0.125)
            self.assertFalse(result["candidate"]["quality_passed"])
            self.assertIn("旧版本经验未改善当前版本整局验证", result["candidate"]["rejection_reasons"])
            adaptive = train_replay_policy(root, catalog, {
                **config, "adaptive_transfer_selection_enabled": True,
            })
            self.assertEqual(adaptive["candidate"]["manifest"]["transfer_actions"], 0)
            self.assertEqual(
                adaptive["candidate"]["manifest"]["transfer_selection_reason"],
                "current_only_due_to_validation_regression",
            )
            self.assertEqual(adaptive["candidate"]["metrics"]["evaluated_transfer_score_gain"], 0.0)
            # Synthetic identical policies have zero gain; explicitly allow it
            # here to exercise weighted runtime serialization, not production gates.
            promoted = train_replay_policy(root, catalog, {
                **config, "allow_bot_training": True, "minimum_transfer_score_gain": 0,
            })
            model = ReplayPolicyModel(root)
            self.assertTrue(promoted["candidate"]["promoted"])
            self.assertTrue(model.available)
            self.assertAlmostEqual(float(model.value_sample_weights[-1]), 0.125)
            row = collect_replay_learning_actions(root, catalog, config)[0]
            card = catalog.by_id[row.card_id]
            self.assertAlmostEqual(model.card_score(card, row.elixir, row.threats()), 1.0)

    def test_adaptive_local_feedback_excludes_regressing_signal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = self._dataset(root)
            result = train_replay_policy(root, catalog, {
                **self._config(False),
                "local_feedback_weight": 0.15,
                "adaptive_local_feedback_selection_enabled": True,
                "adaptive_joint_signal_selection_enabled": True,
            })

            manifest = result["candidate"]["manifest"]
            self.assertEqual(manifest["local_feedback_weight"], 0.0)
            self.assertEqual(manifest["configured_local_feedback_weight"], 0.15)
            self.assertEqual(
                manifest["local_feedback_selection_reason"],
                "joint_selection:current_outcome",
            )
            self.assertTrue(
                manifest["value_probability_calibration"]["available"]
            )
            self.assertTrue(manifest["joint_signal_selection"]["enabled"])
            self.assertEqual(
                manifest["joint_signal_selection"]["selected"], "current_outcome"
            )
            self.assertFalse(
                manifest["joint_signal_selection"]["combinations"]["current_local"]["eligible"]
            )
            with np.load(root / "models/replay_policy" / result["candidate"]["model_path"]) as data:
                self.assertEqual(float(data["local_feedback_weight"][0]), 0.0)

    def test_existing_unweighted_champion_remains_loadable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = self._dataset(root)
            train_replay_policy(root, catalog, self._config(True))
            path = ReplayPolicyRegistry(root).champion_path()
            assert path is not None
            with np.load(path) as data:
                legacy = {key: data[key].copy() for key in data.files if key != "value_sample_weights"}
            np.savez_compressed(path, **legacy)
            model = ReplayPolicyModel(root)
            self.assertTrue(model.available)
            np.testing.assert_array_equal(model.value_sample_weights, np.ones(len(model.value_y)))

    def test_incompatible_champion_is_not_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model_root = root / "models" / "replay_policy"
            model_root.mkdir(parents=True)
            candidate = {
                "version": "old",
                "model_path": "candidate.npz",
                "sync_compatibility": "not-current",
            }
            (model_root / "registry.json").write_text(
                json.dumps({"champion": candidate, "candidates": [candidate]}),
                encoding="utf-8",
            )
            (model_root / "candidate.npz").write_bytes(b"not-a-model")
            (root / "config.json").write_text(
                json.dumps({"replay": {}, "dataset": {"card_catalog": "cards.json"}}),
                encoding="utf-8",
            )

            model = ReplayPolicyModel(root, {})

            self.assertFalse(model.available)
            self.assertEqual(model.load_error, "champion_incompatible")

    def test_old_data_weight_is_capped_and_applied_at_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = self._dataset(root)
            actions = collect_replay_learning_actions(root, catalog, self._config(False))
            x, y, weights = _training_arrays(actions[:2], actions * 4, catalog, 0.0, 0.25)
            self.assertLessEqual(float(weights[2:].sum()), float(weights[:2].sum()) * 0.5)
            prediction = _balanced_knn_predict(
                np.zeros((2, 1)), np.asarray([1., 0.]), np.zeros(1), 2, 1., 1., np.asarray([1., 0.25])
            )
            self.assertAlmostEqual(prediction, 0.8)

    def test_current_training_weight_sums_to_one_per_battle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = self._dataset(root)
            actions = collect_replay_learning_actions(root, catalog, self._config(False))
            rows = [
                replace(actions[0], group_id="short"),
                replace(actions[1], group_id="long"),
                replace(actions[2], group_id="long"),
                replace(actions[3], group_id="long"),
            ]

            _, _, weights = _training_arrays(rows, [], catalog, 0.0, 0.25)

            self.assertAlmostEqual(float(weights[0]), 1.0)
            self.assertAlmostEqual(float(weights[1:].sum()), 1.0)

    def test_temporal_regression_blocks_an_otherwise_valid_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "candidate.npz"
            np.savez_compressed(model_path, placeholder=np.asarray([1]))
            metrics = {
                "validation_episodes": 20,
                "validation_wins": 5,
                "validation_losses": 15,
                "balanced_accuracy": 0.62,
                "auc": 0.65,
                "brier_improvement": 0.02,
                "rank_separation": 0.58,
                "win_deploy_mae": 0.08,
                "temporal_balanced_accuracy": 0.51,
                "temporal_auc": 0.57,
                "temporal_brier_improvement": -0.01,
                "temporal_rank_separation": 0.50,
            }
            config = {
                "allow_bot_training": True,
                "require_temporal_validation": True,
            }

            candidate = ReplayPolicyRegistry(root).register(
                model_path, metrics, config, {"test": True}
            )

            self.assertFalse(candidate["quality_passed"])
            self.assertFalse(candidate["promoted"])
            self.assertTrue(
                any("时间外推" in reason for reason in candidate["rejection_reasons"])
            )

    def test_uncertainty_resamples_whole_battles_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = self._dataset(root)
            actions = collect_replay_learning_actions(root, catalog, self._config(False))
            predictions = np.asarray([0.8 if action.outcome == "win" else 0.2 for action in actions])

            first = _battle_cluster_intervals(actions, predictions, 0.5, seed=9, samples=100)
            second = _battle_cluster_intervals(actions, predictions, 0.5, seed=9, samples=100)

            self.assertEqual(first, second)
            self.assertEqual(first["balanced_accuracy"], [1.0, 1.0])

    def test_validation_segments_report_battle_and_action_denominators(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = self._dataset(root)
            actions = collect_replay_learning_actions(root, catalog, self._config(False))

            segments = _validation_segments(actions)

            self.assertIn("formation_phase", segments)
            self.assertIn("hand", segments)
            self.assertEqual(
                sum(bucket["actions"] for bucket in segments["hand"].values()),
                len(actions),
            )
            self.assertTrue(all(bucket["battles"] > 0 for bucket in segments["hand"].values()))


if __name__ == "__main__":
    unittest.main()
