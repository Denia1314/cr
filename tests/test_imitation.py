from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from crbot.battle_perception import LaneThreat
from crbot.cards import CardCatalog, CardDefinition
from crbot.imitation import (
    ImitationRegistry,
    ImitationPolicyModel,
    audit_demonstrations,
    train_imitation_policy,
)


def card(card_id: str, roles: tuple[str, ...], kind: str = "troop") -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        official_id=None,
        name_zh="",
        name_en=card_id,
        elixir=3,
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


class ImitationTrainingTests(unittest.TestCase):
    def test_selected_card_is_reconstructed_from_previous_hand(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "demonstrations" / "demo"
            run.mkdir(parents=True)
            catalog = CardCatalog(
                [
                    card("a", ("tank",)),
                    card("b", ("splash",)),
                    card("c", ("support",)),
                    card("d", ("swarm",)),
                    card("new_a", ("ranged",)),
                ]
            )
            common = {
                "schema_version": 1,
                "source": "human_demonstration",
                "human_action_ground_truth": True,
                "machine_perception_ground_truth": False,
                "offline_verified": True,
                "battle_index": 1,
                "deploy_point": [0.4, 0.6],
                "elixir": 6,
                "threats": {},
            }
            rows = [
                {
                    **common,
                    "action_id": "a1",
                    "slot_index": 0,
                    "selected_card_id": "a",
                    "selected_card_confidence": 0.9,
                    "hand": ["a", "b", "c", "d"],
                },
                {
                    **common,
                    "action_id": "a2",
                    "slot_index": 1,
                    "selected_card_id": None,
                    "selected_card_confidence": 0.0,
                    "hand": ["new_a", None, "c", "d"],
                },
            ]
            with (run / "actions.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            audit = audit_demonstrations(
                root,
                catalog,
                {
                    "minimum_selected_card_confidence": 0.45,
                    "minimum_actions_for_imitation": 1,
                    "minimum_battles_for_imitation": 1,
                    "minimum_distinct_cards_for_imitation": 1,
                },
            )

            self.assertEqual(audit.usable_actions, 2)
            self.assertEqual(audit.automatically_reconstructed_actions, 1)
            self.assertEqual(audit.selected_card_counts["b"], 1)

    def test_assisted_candidate_can_promote_with_reduced_influence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "candidate.npz"
            np.savez_compressed(
                source,
                selector_x=np.zeros((1, 29), dtype=np.float32),
                selector_y=np.ones(1, dtype=np.float32),
                deploy_x=np.zeros((1, 29), dtype=np.float32),
                deploy_y=np.zeros((1, 2), dtype=np.float32),
                neighbors=np.asarray([1], dtype=np.int32),
            )
            candidate = ImitationRegistry(root).register(
                source,
                {
                    "selection_accuracy": 0.43,
                    "selection_top2_accuracy": 0.71,
                    "selection_random_baseline": 0.29,
                    "selection_lift": 0.14,
                    "role_agreement": 0.59,
                    "deploy_x_mae": 0.18,
                    "deploy_y_mae": 0.09,
                    "deploy_mae": 0.135,
                    "validation_actions": 100,
                },
                {
                    "minimum_validation_actions": 20,
                    "minimum_selection_accuracy": 0.55,
                    "maximum_deploy_mae": 0.12,
                    "minimum_assisted_selection_lift": 0.10,
                    "minimum_assisted_top2_accuracy": 0.65,
                    "minimum_assisted_role_agreement": 0.55,
                    "maximum_assisted_deploy_y_mae": 0.12,
                    "assisted_policy_weight_scale": 0.30,
                },
                {"features": "without_card_identity"},
            )
            model = ImitationPolicyModel(root)

            self.assertTrue(candidate["promoted"])
            self.assertEqual(candidate["maturity"], "assisted")
            self.assertAlmostEqual(model.influence_scale, 0.30)

    def test_manual_actions_train_and_promote_role_based_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "demonstrations" / "demo"
            run.mkdir(parents=True)
            catalog = CardCatalog(
                [
                    card("arrows", ("spell", "splash"), "spell"),
                    card("guards", ("troop", "swarm")),
                    card("hog_rider", ("troop", "win_condition")),
                    card("golem", ("troop", "tank", "win_condition")),
                ]
            )
            rows: list[dict] = []
            for battle in (1, 2):
                for index in range(6):
                    hand = ["guards", "arrows", "hog_rider", "golem"]
                    rows.append(
                        {
                            "schema_version": 1,
                            "action_id": f"b{battle}-a{index}",
                            "source": "human_demonstration",
                            "human_action_ground_truth": True,
                            "machine_perception_ground_truth": False,
                            "offline_verified": True,
                            "battle_index": battle,
                            "slot_index": 1,
                            "deploy_point": [0.31, 0.58],
                            "selected_card_confidence": 0.9,
                            "selected_card_id": "arrows",
                            "hand": hand,
                            "elixir": 6,
                            "threats": {
                                "left": {
                                    "lane": "left",
                                    "score": 0.9,
                                    "unit_count": 4,
                                    "proximity": 0.58,
                                    "threat": "swarm",
                                    "centers": [[0.3, 0.58]],
                                },
                                "right": {
                                    "lane": "right",
                                    "score": 0.0,
                                    "unit_count": 0,
                                    "proximity": 0.0,
                                    "threat": "none",
                                    "centers": [],
                                },
                            },
                        }
                    )
            with (run / "actions.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            config = {
                "minimum_selected_card_confidence": 0.45,
                "minimum_actions_for_imitation": 8,
                "minimum_battles_for_imitation": 2,
                "minimum_distinct_cards_for_imitation": 1,
                "validation_fraction": 0.5,
                "seed": 3,
                "neighbors": 5,
                "minimum_validation_actions": 4,
                "minimum_selection_accuracy": 0.75,
                "maximum_deploy_mae": 0.05,
                "minimum_score_improvement": 0.01,
            }

            audit = audit_demonstrations(root, catalog, config)
            result = train_imitation_policy(root, catalog, config)
            model = ImitationPolicyModel(root)
            threats = {
                "left": LaneThreat("left", 0.9, 4, 0.58, "swarm", ((0.3, 0.58),)),
                "right": LaneThreat("right", 0.0, 0, 0.0, "none", ()),
            }

            self.assertTrue(audit.ready)
            self.assertEqual(audit.usable_actions, 12)
            self.assertTrue(result["candidate"]["promoted"])
            self.assertGreaterEqual(result["metrics"]["selection_accuracy"], 0.75)
            self.assertLessEqual(result["metrics"]["deploy_mae"], 0.05)
            self.assertTrue(model.available)
            arrows = catalog.get("arrows")
            guards = catalog.get("guards")
            assert arrows is not None and guards is not None
            self.assertGreater(
                model.card_score(arrows, 6, threats),
                model.card_score(guards, 6, threats),
            )
            point = model.deploy_point(arrows, 6, threats)
            assert point is not None
            self.assertAlmostEqual(point[0], 0.31, places=2)
            self.assertAlmostEqual(point[1], 0.58, places=2)
            self.assertIn(
                "without_card_identity",
                result["candidate"]["manifest"]["features"],
            )


if __name__ == "__main__":
    unittest.main()
