from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from crbot.cards import CardCatalog, CardDefinition
from crbot.dataset import DatasetStore
from crbot.learning import (
    ModelRegistry,
    audit_learning_data,
    export_yolo_dataset,
    run_learning_cycle,
)
from crbot.battle_perception import LaneThreat
from crbot.learned_perception import LearnedBattlefieldDetector
from crbot.recorder import TrainingRecorder


def make_card(card_id: str) -> CardDefinition:
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
        kind="troop",
        targets=("ground",),
        roles=("troop",),
        counters=(),
        synergies=(),
    )


def training_config() -> dict:
    return {
        "minimum_human_frames": 2,
        "minimum_human_boxes": 2,
        "minimum_human_battles": 2,
        "minimum_distinct_cards": 2,
        "validation_fraction": 0.5,
        "seed": 11,
        "include_pseudo_labels": True,
        "pseudo_confidence": 0.94,
        "maximum_pseudo_frame_ratio": 1.0,
    }


class LearningDatasetTests(unittest.TestCase):
    def test_learning_cycle_waits_without_human_seed_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_learning_cycle(
                Path(directory),
                CardCatalog([make_card("archers")]),
                training_config(),
            )

            self.assertEqual(result["status"], "waiting_for_human_seed_labels")
            self.assertIsNone(result["training"])

    def _build_dataset(
        self,
        root: Path,
        catalog: CardCatalog,
    ) -> tuple[TrainingRecorder, DatasetStore, list[str]]:
        recorder = TrainingRecorder(
            root,
            {"enabled": True, "battle_frame_interval_s": 0.0, "jpeg_quality": 90},
        )
        sample_ids: list[str] = []
        for index, (battle, card_id) in enumerate(
            ((1, "archers"), (1, "knight"), (2, "archers"), (2, "knight")),
            start=1,
        ):
            sample = recorder.record_battle_sample(
                Image.new("RGB", (320, 480), (index * 20, 40, 80)),
                battle_index=battle,
                observed_at_monotonic=float(index),
            )
            assert sample is not None
            sample_ids.append(sample.sample_id)
        store = DatasetStore(recorder.run_dir)
        for index, sample_id in enumerate(sample_ids):
            card_id = "archers" if index % 2 == 0 else "knight"
            self.assertIsNotNone(catalog.get(card_id))
            store.save_annotation(
                {
                    "sample_id": sample_id,
                    "source": "human",
                    "usable": True,
                    "battle_phase": "defense",
                    "hand_cards": [None, None, None, None],
                    "objects": [
                        {
                            "card_id": card_id,
                            "team": "enemy",
                            "bbox": [0.2, 0.3, 0.4, 0.6],
                        }
                    ],
                    "notes": "test",
                }
            )
        return recorder, store, sample_ids

    def test_audit_and_export_split_whole_battles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = CardCatalog([make_card("archers"), make_card("knight")])
            self._build_dataset(root, catalog)

            audit = audit_learning_data(root, catalog, training_config())
            export = export_yolo_dataset(
                root,
                catalog,
                training_config(),
                root / "training" / "export",
            )

            self.assertTrue(audit.ready)
            self.assertEqual(audit.human_frames, 4)
            self.assertEqual(audit.human_boxes, 4)
            self.assertEqual(audit.human_battles, 2)
            self.assertEqual(audit.distinct_cards, 2)
            self.assertTrue(export["validation_is_human_only"])
            self.assertTrue(
                set(export["train_battles"]).isdisjoint(export["validation_battles"])
            )
            self.assertEqual(export["human_train_frames"], 2)
            self.assertEqual(export["human_validation_frames"], 2)
            self.assertEqual(len(list((root / "training/export/labels/train").glob("*.txt"))), 2)
            self.assertEqual(len(list((root / "training/export/labels/val").glob("*.txt"))), 2)

    def test_pseudo_labels_stay_separate_and_never_replace_human(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = CardCatalog([make_card("archers"), make_card("knight")])
            recorder, store, sample_ids = self._build_dataset(root, catalog)
            extra = recorder.record_battle_sample(
                Image.new("RGB", (320, 480), (90, 30, 60)),
                battle_index=3,
                observed_at_monotonic=10.0,
            )
            assert extra is not None
            pseudo = {
                "schema_version": 1,
                "sample_id": extra.sample_id,
                "source": "model_pseudo",
                "model_version": "champion-1",
                "human_ground_truth": False,
                "objects": [
                    {
                        "card_id": "archers",
                        "team": "enemy",
                        "bbox": [0.1, 0.2, 0.3, 0.4],
                        "confidence": 0.98,
                    }
                ],
            }
            with (recorder.run_dir / "pseudo_annotations.jsonl").open(
                "w", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(pseudo) + "\n")

            audit = audit_learning_data(
                root,
                catalog,
                training_config(),
                champion_version="champion-1",
            )
            exported = export_yolo_dataset(
                root,
                catalog,
                training_config(),
                root / "training" / "pseudo-export",
                champion_version="champion-1",
            )

            self.assertEqual(audit.human_frames, 4)
            self.assertEqual(audit.pseudo_frames, 1)
            self.assertEqual(len(store.latest_annotations()), len(sample_ids))
            self.assertEqual(exported["pseudo_train_frames"], 1)
            self.assertTrue(exported["validation_is_human_only"])


class ModelRegistryTests(unittest.TestCase):
    def test_only_valid_and_better_candidate_is_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "candidate.pt"
            model.write_bytes(b"test model")
            registry = ModelRegistry(root)
            thresholds = {
                "minimum_precision": 0.75,
                "minimum_recall": 0.65,
                "minimum_map50": 0.70,
                "minimum_map50_95": 0.40,
                "minimum_score_improvement": 0.005,
            }

            rejected = registry.register_candidate(
                model,
                {"precision": 0.5, "recall": 0.5, "map50": 0.5, "map50_95": 0.2},
                {"validation_is_human_only": True},
                thresholds,
            )
            promoted = registry.register_candidate(
                model,
                {"precision": 0.9, "recall": 0.85, "map50": 0.88, "map50_95": 0.65},
                {"validation_is_human_only": True},
                thresholds,
            )
            worse = registry.register_candidate(
                model,
                {"precision": 0.8, "recall": 0.7, "map50": 0.75, "map50_95": 0.45},
                {"validation_is_human_only": True},
                thresholds,
            )

            self.assertFalse(rejected["promoted"])
            self.assertTrue(rejected["rejection_reasons"])
            self.assertTrue(promoted["promoted"])
            self.assertFalse(worse["promoted"])
            self.assertEqual(registry.champion()["version"], promoted["version"])
            self.assertTrue(registry.champion_model_path().is_file())


class LearnedPerceptionTests(unittest.TestCase):
    def test_validated_detector_enriches_lane_with_exact_enemy_card(self) -> None:
        catalog = CardCatalog(
            [
                CardDefinition(
                    **{
                        **make_card("giant").__dict__,
                        "roles": ("troop", "tank", "win_condition"),
                    }
                )
            ]
        )

        class Vector:
            def __init__(self, values: list) -> None:
                self.values = values

            def cpu(self) -> "Vector":
                return self

            def tolist(self) -> list:
                return self.values

        class Boxes:
            cls = Vector([0.0])
            conf = Vector([0.95])
            xyxyn = Vector([[0.58, 0.48, 0.72, 0.68]])

        class Result:
            boxes = Boxes()

        class Model:
            @staticmethod
            def predict(**_kwargs: object) -> list[Result]:
                return [Result()]

        detector = LearnedBattlefieldDetector.__new__(LearnedBattlefieldDetector)
        detector.catalog = catalog
        detector.training = {"runtime_confidence": 0.7, "image_size": 640, "device": "cpu"}
        detector.registry = None
        detector.champion = {"version": "test"}
        detector.model = Model()
        detector.class_names = ["enemy__giant"]
        detector.error = ""
        fallback = {
            "left": LaneThreat("left", 0.0, 0, 0.0, "none", ()),
            "right": LaneThreat("right", 0.0, 0, 0.0, "none", ()),
        }

        threats = detector.detect(Image.new("RGB", (600, 1000), "black"), fallback)

        self.assertEqual(threats["right"].enemy_cards, ("giant",))
        self.assertEqual(threats["right"].threat, "heavy")
        self.assertGreater(threats["right"].score, 0.2)

        from crbot.knowledge import KnowledgeBase
        detector.catalog = CardCatalog.load(Path(__file__).resolve().parents[1] / "data/cards.json")
        detector.knowledge = KnowledgeBase.load(Path(__file__).resolve().parents[1] / "data/battle_knowledge.json")
        for cid, layer in (("giant", "ground"), ("balloon", "air"), ("minions", "air"), ("musketeer", "ground")):
            detector.class_names = ["enemy__" + cid]
            result = detector.detect(Image.new("RGB", (600, 1000), "black"), fallback)["right"]
            self.assertEqual(result.unit_layers, (layer,))
            self.assertGreaterEqual(result.layer_confidence, .7)
        fallback["right"] = LaneThreat("right", .8, 2, .7, "heavy", ((.65, .58), (.8, .7)))
        result = detector.detect(Image.new("RGB", (600, 1000), "black"), fallback)["right"]
        self.assertIn((.8, .7), result.centers)
        self.assertEqual(result.layer_confidence, 0)
        self.assertEqual(result.unit_layers, ())


if __name__ == "__main__":
    unittest.main()
