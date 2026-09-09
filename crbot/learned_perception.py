from __future__ import annotations

from pathlib import Path
from typing import Any
from PIL import Image

from .battle_perception import LaneThreat
from .cards import CardCatalog
from .gpu import vision_device
from .learning import ModelRegistry


class LearnedBattlefieldDetector:
    """Optional exact-card detector backed only by a validated champion model."""

    def __init__(
        self,
        project_root: Path,
        catalog: CardCatalog,
        training_config: dict[str, Any],
    ):
        self.catalog = catalog
        self.training = training_config
        self.registry = ModelRegistry(project_root)
        self.champion = self.registry.champion()
        self.model = None
        self.class_names: list[str] = []
        self.error = ""
        self.observed_allies: list[dict[str, Any]] = []
        self.allies_observed = False
        model_path = self.registry.champion_model_path()
        if self.champion is None or model_path is None:
            return
        try:
            from ultralytics import YOLO

            self.model = YOLO(str(model_path))
            self.class_names = list(
                self.champion.get("dataset_manifest", {}).get("classes", [])
            )
        except (ImportError, OSError, ValueError) as exc:
            self.error = str(exc)

    @property
    def available(self) -> bool:
        return self.model is not None and bool(self.class_names)

    def detect(
        self,
        image: Image.Image,
        fallback: dict[str, LaneThreat],
    ) -> dict[str, LaneThreat]:
        self.observed_allies = []
        self.allies_observed = False
        if not self.available:
            return fallback
        try:
            result = self.model.predict(
                source=image,
                conf=float(self.training.get("runtime_confidence", 0.70)),
                imgsz=int(self.training.get("image_size", 640)),
                device=vision_device(self.training.get("device", "auto")),
                verbose=False,
            )[0]
        except Exception as exc:  # pragma: no cover - backend/runtime dependent
            self.error = str(exc)
            return fallback
        self.allies_observed = any(str(name).startswith("ally__") for name in self.class_names)
        by_lane: dict[str, list[tuple[str, float, float, float]]] = {
            "left": [],
            "right": [],
        }
        if result.boxes is not None:
            classes = result.boxes.cls.cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()
            boxes = result.boxes.xyxyn.cpu().tolist()
            for class_index, confidence, bbox in zip(classes, confidences, boxes):
                index = int(class_index)
                if index < 0 or index >= len(self.class_names):
                    continue
                class_name = str(self.class_names[index])
                if class_name.startswith("ally__"):
                    card_id = class_name.removeprefix("ally__")
                    x1, y1, x2, y2 = [float(value) for value in bbox]
                    x, y = (x1 + x2) / 2, (y1 + y2) / 2
                    if self.catalog.get(card_id) is not None and 0.20 <= y <= 0.80:
                        self.observed_allies.append({
                            "card_id": card_id, "x": round(x, 4), "y": round(y, 4),
                            "lane": "left" if x < 0.5 else "right",
                            "confidence": float(confidence),
                        })
                    continue
                if not class_name.startswith("enemy__"):
                    continue
                card_id = class_name.removeprefix("enemy__")
                if self.catalog.get(card_id) is None:
                    continue
                x1, y1, x2, y2 = [float(value) for value in bbox]
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                if not 0.20 <= center_y <= 0.80:
                    continue
                lane = "left" if center_x < 0.5 else "right"
                by_lane[lane].append((card_id, center_x, center_y, float(confidence)))

        merged: dict[str, LaneThreat] = {}
        for lane in ("left", "right"):
            predictions = by_lane[lane]
            base = fallback[lane]
            if not predictions:
                merged[lane] = base
                continue
            enemy_cards = tuple(value[0] for value in predictions)
            enemy_roles = {
                role
                for card_id in enemy_cards
                for role in (self.catalog.get(card_id).roles if self.catalog.get(card_id) else ())
            }
            proximity = max(value[2] for value in predictions)
            learned_score = min(
                1.0,
                len(predictions) * 0.20
                + max(0.0, (proximity - 0.20) / 0.52) * 0.62
                + max(value[3] for value in predictions) * 0.15,
            )
            if "swarm" in enemy_roles or len(predictions) >= 4:
                threat_type = "swarm"
            elif enemy_roles.intersection({"tank", "win_condition"}) or proximity >= 0.63:
                threat_type = "heavy"
            else:
                threat_type = "single"
            centers = tuple((round(value[1], 4), round(value[2], 4)) for value in predictions)
            merged[lane] = LaneThreat(
                lane=lane,
                score=round(max(base.score, learned_score), 4),
                unit_count=max(base.unit_count, len(predictions)),
                proximity=round(max(base.proximity, proximity), 4),
                threat=threat_type,
                centers=centers,
                enemy_cards=enemy_cards,
                approach_rate=base.approach_rate,
                unit_layers=base.unit_layers,
                layer_confidence=base.layer_confidence,
            )
        return merged
