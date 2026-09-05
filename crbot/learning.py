from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from .cards import CardCatalog
from .dataset import DatasetStore, HUMAN_LABEL_SOURCES, SampleRecord, list_dataset_runs


LEARNING_SCHEMA_VERSION = 1


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_number} 行不是有效 JSON") from exc
            if isinstance(value, dict):
                yield value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


@dataclass(frozen=True)
class LabeledFrame:
    run_dir: Path
    sample: SampleRecord
    frame_path: Path
    source: str
    objects: tuple[dict[str, Any], ...]
    model_version: str = ""

    @property
    def group_id(self) -> str:
        return f"{self.run_dir.name}:battle-{self.sample.battle_index}"


@dataclass(frozen=True)
class LearningAudit:
    runs: int
    captured_frames: int
    human_frames: int
    human_boxes: int
    human_battles: int
    pseudo_frames: int
    pseudo_boxes: int
    distinct_cards: int
    class_counts: dict[str, int]
    invalid_objects: int
    ready: bool
    blocking_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["blocking_reasons"] = list(self.blocking_reasons)
        return value


def _valid_objects(
    raw_objects: Iterable[dict[str, Any]],
    catalog: CardCatalog,
    *,
    minimum_confidence: float | None = None,
) -> tuple[list[dict[str, Any]], int]:
    valid: list[dict[str, Any]] = []
    invalid = 0
    for raw in raw_objects:
        card_id = str(raw.get("card_id", "")).strip()
        team = str(raw.get("team", "")).strip()
        bbox = raw.get("bbox", [])
        confidence = float(raw.get("confidence", 1.0))
        if (
            catalog.get(card_id) is None
            or team not in {"enemy", "ally"}
            or not isinstance(bbox, list)
            or len(bbox) != 4
            or any(not isinstance(value, (int, float)) for value in bbox)
            or any(float(value) < 0.0 or float(value) > 1.0 for value in bbox)
            or float(bbox[2]) <= float(bbox[0])
            or float(bbox[3]) <= float(bbox[1])
            or (minimum_confidence is not None and confidence < minimum_confidence)
        ):
            invalid += 1
            continue
        valid.append(
            {
                "card_id": card_id,
                "team": team,
                "bbox": [round(float(value), 6) for value in bbox],
                "confidence": round(confidence, 6),
            }
        )
    return valid, invalid


def collect_human_frames(
    project_root: Path,
    catalog: CardCatalog,
) -> tuple[list[LabeledFrame], int, int]:
    frames: list[LabeledFrame] = []
    invalid_objects = 0
    captured_frames = 0
    for run_dir in list_dataset_runs(project_root):
        store = DatasetStore(run_dir)
        samples = {sample.sample_id: sample for sample in store.samples()}
        captured_frames += len(samples)
        for sample_id, annotation in store.latest_annotations().items():
            sample = samples.get(sample_id)
            if sample is None:
                continue
            if (
                str(annotation.get("source", "")) not in HUMAN_LABEL_SOURCES
                or not bool(annotation.get("usable", True))
            ):
                continue
            objects, invalid = _valid_objects(annotation.get("objects", []), catalog)
            invalid_objects += invalid
            if not objects:
                continue
            frames.append(
                LabeledFrame(
                    run_dir=run_dir,
                    sample=sample,
                    frame_path=store.frame_path(sample),
                    source=str(annotation["source"]),
                    objects=tuple(objects),
                )
            )
    return frames, invalid_objects, captured_frames


def latest_pseudo_annotations(run_dir: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for value in _read_jsonl(run_dir / "pseudo_annotations.jsonl"):
        sample_id = str(value.get("sample_id", ""))
        if sample_id:
            latest[sample_id] = value
    return latest


def collect_pseudo_frames(
    project_root: Path,
    catalog: CardCatalog,
    *,
    model_version: str,
    minimum_confidence: float,
    human_sample_ids: set[str],
) -> tuple[list[LabeledFrame], int]:
    frames: list[LabeledFrame] = []
    invalid_objects = 0
    if not model_version:
        return frames, invalid_objects
    for run_dir in list_dataset_runs(project_root):
        store = DatasetStore(run_dir)
        samples = {sample.sample_id: sample for sample in store.samples()}
        for sample_id, annotation in latest_pseudo_annotations(run_dir).items():
            if sample_id in human_sample_ids or sample_id not in samples:
                continue
            if str(annotation.get("model_version", "")) != model_version:
                continue
            objects, invalid = _valid_objects(
                annotation.get("objects", []),
                catalog,
                minimum_confidence=minimum_confidence,
            )
            invalid_objects += invalid
            if not objects:
                continue
            sample = samples[sample_id]
            frames.append(
                LabeledFrame(
                    run_dir=run_dir,
                    sample=sample,
                    frame_path=store.frame_path(sample),
                    source="model_pseudo",
                    objects=tuple(objects),
                    model_version=model_version,
                )
            )
    return frames, invalid_objects


def audit_learning_data(
    project_root: Path,
    catalog: CardCatalog,
    training_config: dict[str, Any],
    *,
    champion_version: str = "",
) -> LearningAudit:
    human, invalid, captured = collect_human_frames(project_root, catalog)
    human_ids = {frame.sample.sample_id for frame in human}
    pseudo, pseudo_invalid = collect_pseudo_frames(
        project_root,
        catalog,
        model_version=champion_version,
        minimum_confidence=float(training_config.get("pseudo_confidence", 0.94)),
        human_sample_ids=human_ids,
    )
    invalid += pseudo_invalid
    class_counts: Counter[str] = Counter()
    for frame in human:
        for obj in frame.objects:
            class_counts[str(obj["card_id"])] += 1
    human_boxes = sum(len(frame.objects) for frame in human)
    pseudo_boxes = sum(len(frame.objects) for frame in pseudo)
    human_battles = len({frame.group_id for frame in human})
    minimum_frames = int(training_config.get("minimum_human_frames", 200))
    minimum_boxes = int(training_config.get("minimum_human_boxes", 300))
    minimum_battles = int(training_config.get("minimum_human_battles", 10))
    minimum_cards = int(training_config.get("minimum_distinct_cards", 12))
    reasons: list[str] = []
    if len(human) < minimum_frames:
        reasons.append(f"人工框选帧 {len(human)}/{minimum_frames}")
    if human_boxes < minimum_boxes:
        reasons.append(f"人工目标框 {human_boxes}/{minimum_boxes}")
    if human_battles < minimum_battles:
        reasons.append(f"已覆盖对局 {human_battles}/{minimum_battles}")
    if len(class_counts) < minimum_cards:
        reasons.append(f"不同卡牌 {len(class_counts)}/{minimum_cards}")
    return LearningAudit(
        runs=len(list_dataset_runs(project_root)),
        captured_frames=captured,
        human_frames=len(human),
        human_boxes=human_boxes,
        human_battles=human_battles,
        pseudo_frames=len(pseudo),
        pseudo_boxes=pseudo_boxes,
        distinct_cards=len(class_counts),
        class_counts=dict(sorted(class_counts.items())),
        invalid_objects=invalid,
        ready=not reasons,
        blocking_reasons=tuple(reasons),
    )


def _group_split(
    frames: list[LabeledFrame],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[LabeledFrame], list[LabeledFrame]]:
    groups = sorted({frame.group_id for frame in frames})
    if len(groups) < 2:
        raise ValueError("至少需要来自两局不同对局的人工框选数据")
    ranked = sorted(
        groups,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )
    validation_count = max(1, min(len(groups) - 1, math.ceil(len(groups) * validation_fraction)))
    validation_groups = set(ranked[:validation_count])
    training = [frame for frame in frames if frame.group_id not in validation_groups]
    validation = [frame for frame in frames if frame.group_id in validation_groups]
    return training, validation


def _class_name(obj: dict[str, Any]) -> str:
    return f"{obj['team']}__{obj['card_id']}"


def export_yolo_dataset(
    project_root: Path,
    catalog: CardCatalog,
    training_config: dict[str, Any],
    output_dir: Path,
    *,
    champion_version: str = "",
) -> dict[str, Any]:
    human, invalid_objects, _captured = collect_human_frames(project_root, catalog)
    if not human:
        raise ValueError("没有可导出的人工目标框；请先在“数据标注”中框选敌我单位")
    train_human, validation = _group_split(
        human,
        validation_fraction=float(training_config.get("validation_fraction", 0.2)),
        seed=int(training_config.get("seed", 20260902)),
    )
    class_names = sorted({_class_name(obj) for frame in human for obj in frame.objects})
    class_ids = {name: index for index, name in enumerate(class_names)}

    pseudo: list[LabeledFrame] = []
    if bool(training_config.get("include_pseudo_labels", True)) and champion_version:
        pseudo, pseudo_invalid = collect_pseudo_frames(
            project_root,
            catalog,
            model_version=champion_version,
            minimum_confidence=float(training_config.get("pseudo_confidence", 0.94)),
            human_sample_ids={frame.sample.sample_id for frame in human},
        )
        invalid_objects += pseudo_invalid
        pseudo = [
            frame
            for frame in pseudo
            if all(_class_name(obj) in class_ids for obj in frame.objects)
        ]
        maximum_ratio = min(
            1.0, max(0.0, float(training_config.get("maximum_pseudo_frame_ratio", 0.5)))
        )
        maximum_pseudo = max(0, int(len(train_human) * maximum_ratio))
        maximum_box_ratio = min(
            1.0, max(0.0, float(training_config.get("maximum_pseudo_box_ratio", 0.5)))
        )
        maximum_pseudo_boxes = int(
            sum(len(frame.objects) for frame in train_human) * maximum_box_ratio
        )
        selected_pseudo: list[LabeledFrame] = []
        selected_boxes = 0
        for frame in sorted(pseudo, key=lambda value: value.sample.sample_id):
            if len(selected_pseudo) >= maximum_pseudo:
                break
            if selected_boxes + len(frame.objects) > maximum_pseudo_boxes:
                continue
            selected_pseudo.append(frame)
            selected_boxes += len(frame.objects)
        pseudo = selected_pseudo

    if output_dir.exists():
        raise ValueError(f"训练集输出目录已存在：{output_dir}")
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def write_frame(frame: LabeledFrame, split: str) -> None:
        image_name = f"{frame.sample.sample_id}{frame.frame_path.suffix.casefold()}"
        target_image = output_dir / "images" / split / image_name
        shutil.copy2(frame.frame_path, target_image)
        label_path = output_dir / "labels" / split / f"{frame.sample.sample_id}.txt"
        rows: list[str] = []
        for obj in frame.objects:
            x1, y1, x2, y2 = [float(value) for value in obj["bbox"]]
            class_id = class_ids[_class_name(obj)]
            rows.append(
                f"{class_id} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} "
                f"{x2 - x1:.6f} {y2 - y1:.6f}"
            )
        label_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    for frame in train_human + pseudo:
        write_frame(frame, "train")
    for frame in validation:
        write_frame(frame, "val")

    data_yaml = output_dir / "data.yaml"
    yaml_names = json.dumps(class_names, ensure_ascii=False)
    data_yaml.write_text(
        f"path: {json.dumps(output_dir.resolve().as_posix())}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names: {yaml_names}\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": LEARNING_SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "human_train_frames": len(train_human),
        "pseudo_train_frames": len(pseudo),
        "human_validation_frames": len(validation),
        "train_battles": sorted({frame.group_id for frame in train_human}),
        "validation_battles": sorted({frame.group_id for frame in validation}),
        "classes": class_names,
        "invalid_objects_skipped": invalid_objects,
        "validation_is_human_only": True,
        "pseudo_model_version": champion_version if pseudo else "",
    }
    _write_json(output_dir / "manifest.json", manifest)
    return {**manifest, "output_dir": str(output_dir.resolve()), "data_yaml": str(data_yaml.resolve())}


class ModelRegistry:
    def __init__(self, project_root: Path):
        self.root = project_root.resolve() / "models" / "battlefield"
        self.registry_path = self.root / "registry.json"

    def load(self) -> dict[str, Any]:
        if not self.registry_path.is_file():
            return {
                "schema_version": LEARNING_SCHEMA_VERSION,
                "champion": None,
                "candidates": [],
            }
        with self.registry_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError("模型注册表格式无效")
        return value

    def champion(self) -> dict[str, Any] | None:
        value = self.load().get("champion")
        return value if isinstance(value, dict) else None

    def champion_model_path(self) -> Path | None:
        champion = self.champion()
        if champion is None:
            return None
        path = self.root / str(champion.get("model_path", ""))
        return path.resolve() if path.is_file() else None

    @staticmethod
    def _score(metrics: dict[str, float]) -> float:
        return (
            0.35 * float(metrics.get("precision", 0.0))
            + 0.25 * float(metrics.get("recall", 0.0))
            + 0.25 * float(metrics.get("map50", 0.0))
            + 0.15 * float(metrics.get("map50_95", 0.0))
        )

    def register_candidate(
        self,
        model_path: Path,
        metrics: dict[str, float],
        manifest: dict[str, Any],
        promotion_config: dict[str, Any],
    ) -> dict[str, Any]:
        if not model_path.is_file():
            raise ValueError(f"候选模型不存在：{model_path}")
        version = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}"
        relative_path = Path("candidates") / f"battlefield_{version}.pt"
        target = self.root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, target)
        normalized_metrics = {
            key: round(float(metrics.get(key, 0.0)), 6)
            for key in ("precision", "recall", "map50", "map50_95")
        }
        candidate = {
            "version": version,
            "created_at_unix": time.time(),
            "model_path": relative_path.as_posix(),
            "metrics": normalized_metrics,
            "score": round(self._score(normalized_metrics), 6),
            "dataset_manifest": manifest,
            "promoted": False,
            "rejection_reasons": [],
        }
        minimums = {
            "precision": float(promotion_config.get("minimum_precision", 0.75)),
            "recall": float(promotion_config.get("minimum_recall", 0.65)),
            "map50": float(promotion_config.get("minimum_map50", 0.70)),
            "map50_95": float(promotion_config.get("minimum_map50_95", 0.40)),
        }
        rejection_reasons = [
            f"{key}={normalized_metrics[key]:.3f}<{minimum:.3f}"
            for key, minimum in minimums.items()
            if normalized_metrics[key] < minimum
        ]
        if not bool(manifest.get("validation_is_human_only", False)):
            rejection_reasons.append("验证集不是纯人工真值")
        registry = self.load()
        champion = registry.get("champion")
        minimum_improvement = float(promotion_config.get("minimum_score_improvement", 0.005))
        if isinstance(champion, dict):
            champion_score = float(champion.get("score", 0.0))
            if candidate["score"] < champion_score + minimum_improvement:
                rejection_reasons.append(
                    f"综合分 {candidate['score']:.3f} 未超过当前模型 {champion_score:.3f}"
                )
        candidate["rejection_reasons"] = rejection_reasons
        if not rejection_reasons:
            candidate["promoted"] = True
            registry["champion"] = dict(candidate)
        registry.setdefault("candidates", []).append(candidate)
        _write_json(self.registry_path, registry)
        return candidate


def train_detector(
    project_root: Path,
    catalog: CardCatalog,
    training_config: dict[str, Any],
) -> dict[str, Any]:
    registry = ModelRegistry(project_root)
    champion = registry.champion()
    champion_version = str(champion.get("version", "")) if champion else ""
    audit = audit_learning_data(
        project_root,
        catalog,
        training_config,
        champion_version=champion_version,
    )
    if not audit.ready:
        raise ValueError("训练数据尚未达标：" + "；".join(audit.blocking_reasons))
    version = time.strftime("%Y%m%d_%H%M%S")
    export_dir = project_root / "training" / "exports" / version
    manifest = export_yolo_dataset(
        project_root,
        catalog,
        training_config,
        export_dir,
        champion_version=champion_version,
    )
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - optional heavyweight dependency
        raise ValueError(
            "缺少训练依赖 ultralytics；请先安装 requirements-training.txt"
        ) from exc

    model = YOLO(str(training_config.get("base_model", "yolo11n.pt")))
    results = model.train(
        data=manifest["data_yaml"],
        epochs=int(training_config.get("epochs", 50)),
        imgsz=int(training_config.get("image_size", 640)),
        batch=int(training_config.get("batch", 8)),
        device=str(training_config.get("device", "cpu")),
        project=str((project_root / "models" / "battlefield" / "training_runs").resolve()),
        name=version,
        exist_ok=False,
        seed=int(training_config.get("seed", 20260902)),
    )
    save_dir = Path(str(results.save_dir))
    best_model = save_dir / "weights" / "best.pt"
    validation_model = YOLO(str(best_model))
    metrics_result = validation_model.val(
        data=manifest["data_yaml"],
        split="val",
        imgsz=int(training_config.get("image_size", 640)),
        device=str(training_config.get("device", "cpu")),
    )
    box = metrics_result.box
    metrics = {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
    }
    candidate = registry.register_candidate(
        best_model,
        metrics,
        manifest,
        dict(training_config.get("promotion", {})),
    )
    return {"audit": audit.to_dict(), "manifest": manifest, "candidate": candidate}


def auto_label_with_champion(
    project_root: Path,
    catalog: CardCatalog,
    training_config: dict[str, Any],
    *,
    limit: int = 0,
) -> dict[str, Any]:
    registry = ModelRegistry(project_root)
    champion = registry.champion()
    model_path = registry.champion_model_path()
    if champion is None or model_path is None:
        raise ValueError("还没有通过验证并晋级的战场识别模型")
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - optional heavyweight dependency
        raise ValueError(
            "缺少推理依赖 ultralytics；请先安装 requirements-training.txt"
        ) from exc
    model = YOLO(str(model_path))
    class_names = list(champion.get("dataset_manifest", {}).get("classes", []))
    threshold = float(training_config.get("pseudo_confidence", 0.94))
    written_frames = 0
    written_boxes = 0
    inspected = 0
    for run_dir in list_dataset_runs(project_root):
        store = DatasetStore(run_dir)
        human_ids = set(store.latest_annotations())
        already = latest_pseudo_annotations(run_dir)
        for sample in store.samples():
            existing_pseudo = already.get(sample.sample_id)
            if sample.sample_id in human_ids or (
                existing_pseudo is not None
                and str(existing_pseudo.get("model_version", ""))
                == str(champion["version"])
            ):
                continue
            if limit > 0 and inspected >= limit:
                break
            inspected += 1
            result = model.predict(
                source=str(store.frame_path(sample)),
                conf=threshold,
                imgsz=int(training_config.get("image_size", 640)),
                device=str(training_config.get("device", "cpu")),
                verbose=False,
            )[0]
            objects: list[dict[str, Any]] = []
            if result.boxes is not None:
                classes = result.boxes.cls.cpu().tolist()
                confidences = result.boxes.conf.cpu().tolist()
                boxes = result.boxes.xyxyn.cpu().tolist()
                for class_index, confidence, bbox in zip(classes, confidences, boxes):
                    index = int(class_index)
                    if index < 0 or index >= len(class_names):
                        continue
                    class_name = str(class_names[index])
                    if "__" not in class_name:
                        continue
                    team, card_id = class_name.split("__", 1)
                    if catalog.get(card_id) is None or team not in {"enemy", "ally"}:
                        continue
                    objects.append(
                        {
                            "card_id": card_id,
                            "team": team,
                            "bbox": [round(float(value), 6) for value in bbox],
                            "confidence": round(float(confidence), 6),
                        }
                    )
            if not objects:
                continue
            payload = {
                "schema_version": LEARNING_SCHEMA_VERSION,
                "sample_id": sample.sample_id,
                "created_at_unix": time.time(),
                "source": "model_pseudo",
                "model_version": champion["version"],
                "minimum_confidence": threshold,
                "objects": objects,
                "human_ground_truth": False,
            }
            pseudo_path = run_dir / "pseudo_annotations.jsonl"
            with pseudo_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written_frames += 1
            written_boxes += len(objects)
        if limit > 0 and inspected >= limit:
            break
    return {
        "model_version": champion["version"],
        "inspected_frames": inspected,
        "pseudo_frames_written": written_frames,
        "pseudo_boxes_written": written_boxes,
        "confidence_threshold": threshold,
        "human_annotations_overwritten": 0,
    }


def run_learning_cycle(
    project_root: Path,
    catalog: CardCatalog,
    training_config: dict[str, Any],
) -> dict[str, Any]:
    """Run the safe self-training loop, or stop before mutation when seed data is insufficient."""
    registry = ModelRegistry(project_root)
    champion = registry.champion()
    champion_version = str(champion.get("version", "")) if champion else ""
    audit = audit_learning_data(
        project_root,
        catalog,
        training_config,
        champion_version=champion_version,
    )
    if not audit.ready:
        return {
            "status": "waiting_for_human_seed_labels",
            "audit": audit.to_dict(),
            "auto_label": None,
            "training": None,
        }
    auto_label_result = None
    if champion is not None:
        auto_label_result = auto_label_with_champion(
            project_root,
            catalog,
            training_config,
            limit=max(0, int(training_config.get("cycle_auto_label_limit", 500))),
        )
    training_result = train_detector(project_root, catalog, training_config)
    return {
        "status": (
            "new_champion_promoted"
            if training_result["candidate"].get("promoted")
            else "candidate_rejected"
        ),
        "audit": audit.to_dict(),
        "auto_label": auto_label_result,
        "training": training_result,
    }
