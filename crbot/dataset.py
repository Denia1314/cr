from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


DATASET_SCHEMA_VERSION = 1
HUMAN_LABEL_SOURCES = {"human", "imported_expert"}


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name} 第 {line_number} 行不是有效 JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path.name} 第 {line_number} 行必须是 JSON 对象")
            yield value


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    timestamp_unix: float
    frame: str
    screen_size: tuple[int, int]
    battle_index: int
    source: str = "continuous_capture"
    label_status: str = "unlabeled"
    bot_action_is_ground_truth: bool = False
    event_context: str = "battle_frame"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SampleRecord":
        raw_size = value.get("screen_size", [0, 0])
        return cls(
            sample_id=str(value["sample_id"]),
            timestamp_unix=float(value.get("timestamp_unix", 0.0)),
            frame=str(value["frame"]),
            screen_size=(int(raw_size[0]), int(raw_size[1])),
            battle_index=int(value.get("battle_index", 0)),
            source=str(value.get("source", "continuous_capture")),
            label_status=str(value.get("label_status", "unlabeled")),
            bot_action_is_ground_truth=bool(
                value.get("bot_action_is_ground_truth", False)
            ),
            event_context=str(value.get("event_context", "battle_frame")),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["screen_size"] = list(self.screen_size)
        return value


def list_dataset_runs(project_root: Path) -> list[Path]:
    runs_dir = project_root / "runs"
    if not runs_dir.is_dir():
        return []
    candidates = [
        path
        for path in runs_dir.iterdir()
        if path.is_dir()
        and ((path / "samples.jsonl").is_file() or (path / "events.jsonl").is_file())
    ]
    return sorted(candidates, key=lambda path: path.name, reverse=True)


class DatasetStore:
    """Read captured battle samples and append human-authored annotations."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir.resolve()
        self.samples_path = self.run_dir / "samples.jsonl"
        self.events_path = self.run_dir / "events.jsonl"
        self.annotations_path = self.run_dir / "annotations.jsonl"

    def samples(self) -> list[SampleRecord]:
        if self.samples_path.is_file():
            records = [SampleRecord.from_dict(value) for value in _read_jsonl(self.samples_path)]
        else:
            records = self._legacy_event_samples()
        return [record for record in records if self.frame_path(record).is_file()]

    def _legacy_event_samples(self) -> list[SampleRecord]:
        records: list[SampleRecord] = []
        seen_frames: set[str] = set()
        battle_index = 0
        for value in _read_jsonl(self.events_path):
            event = str(value.get("event", ""))
            if event == "battle_started_auto":
                battle_index += 1
            frame = value.get("frame")
            if not frame or str(frame) in seen_frames:
                continue
            if event not in {"battle_started_auto", "battle_action", "battle_ended_auto"}:
                continue
            frame_text = str(frame)
            seen_frames.add(frame_text)
            sequence = int(value.get("sequence", len(records) + 1))
            size = value.get("screen_size", [0, 0])
            records.append(
                SampleRecord(
                    sample_id=f"{self.run_dir.name}-legacy-{sequence:06d}",
                    timestamp_unix=float(value.get("timestamp_unix", 0.0)),
                    frame=frame_text,
                    screen_size=(int(size[0]), int(size[1])),
                    battle_index=max(1, battle_index),
                    source="legacy_event_frame",
                    event_context=event,
                )
            )
        return records

    def frame_path(self, sample: SampleRecord) -> Path:
        path = (self.run_dir / sample.frame).resolve()
        if path != self.run_dir and self.run_dir not in path.parents:
            raise ValueError(f"样本帧路径越界：{sample.frame}")
        return path

    def latest_annotations(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for value in _read_jsonl(self.annotations_path):
            sample_id = str(value.get("sample_id", ""))
            if sample_id:
                latest[sample_id] = value
        return latest

    def save_annotation(self, annotation: dict[str, Any]) -> dict[str, Any]:
        samples = {sample.sample_id: sample for sample in self.samples()}
        sample_id = str(annotation.get("sample_id", "")).strip()
        if sample_id not in samples:
            raise ValueError("标注对应的样本不存在")

        source = str(annotation.get("source", "human"))
        if source not in HUMAN_LABEL_SOURCES:
            raise ValueError("训练标签只能来自人工或导入的专家标注")

        hand_cards = list(annotation.get("hand_cards", []))
        if len(hand_cards) != 4:
            raise ValueError("hand_cards 必须包含四个卡槽")
        normalized_hand = [str(value).strip() if value else None for value in hand_cards]

        objects: list[dict[str, Any]] = []
        for raw_object in annotation.get("objects", []):
            card_id = str(raw_object.get("card_id", "")).strip()
            team = str(raw_object.get("team", "enemy"))
            bbox = [float(value) for value in raw_object.get("bbox", [])]
            if not card_id:
                raise ValueError("每个框选目标都必须填写 card_id")
            if team not in {"enemy", "ally"}:
                raise ValueError("team 必须是 enemy 或 ally")
            if len(bbox) != 4 or any(value < 0.0 or value > 1.0 for value in bbox):
                raise ValueError("bbox 必须是 0 到 1 之间的四个归一化坐标")
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                raise ValueError("bbox 的右下角必须位于左上角之后")
            objects.append(
                {
                    "card_id": card_id,
                    "team": team,
                    "bbox": [round(value, 6) for value in bbox],
                }
            )

        previous = self.latest_annotations().get(sample_id, {})
        saved: dict[str, Any] = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "sample_id": sample_id,
            "revision": int(previous.get("revision", 0)) + 1,
            "updated_at_unix": time.time(),
            "source": source,
            "usable": bool(annotation.get("usable", True)),
            "battle_phase": str(annotation.get("battle_phase", "unknown")),
            "hand_cards": normalized_hand,
            "objects": objects,
            "notes": str(annotation.get("notes", "")).strip(),
        }
        self.annotations_path.parent.mkdir(parents=True, exist_ok=True)
        with self.annotations_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(saved, ensure_ascii=False) + "\n")
        return saved

    def summary(self) -> dict[str, int]:
        samples = self.samples()
        annotated = self.latest_annotations()
        usable = sum(
            1
            for sample in samples
            if sample.sample_id in annotated and bool(annotated[sample.sample_id].get("usable"))
        )
        return {
            "samples": len(samples),
            "annotated": sum(1 for sample in samples if sample.sample_id in annotated),
            "usable": usable,
        }
