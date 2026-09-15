from __future__ import annotations

import io
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from .dataset import DATASET_SCHEMA_VERSION, SampleRecord


class TrainingRecorder:
    def __init__(
        self,
        project_root: Path,
        dataset_config: dict[str, Any] | None = None,
    ):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = project_root / "runs" / stamp
        collision = 1
        while self.run_dir.exists():
            self.run_dir = project_root / "runs" / f"{stamp}_{collision:02d}"
            collision += 1
        self.frames_dir = self.run_dir / "frames"
        self.samples_dir = self.run_dir / "samples"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.samples_path = self.run_dir / "samples.jsonl"
        self.manifest_path = self.run_dir / "dataset.json"
        self.sequence = 0
        self.sample_sequence = 0
        self.dataset_config = dataset_config or {}
        self.dataset_enabled = bool(self.dataset_config.get("enabled", True))
        self.sample_interval_s = max(
            0.1, float(self.dataset_config.get("battle_frame_interval_s", 2.0))
        )
        self.sample_jpeg_quality = max(
            40, min(100, int(self.dataset_config.get("jpeg_quality", 65)))
        )
        self.event_jpeg_quality = max(
            40,
            min(
                100,
                int(
                    self.dataset_config.get(
                        "event_jpeg_quality", self.sample_jpeg_quality
                    )
                ),
            ),
        )
        self.max_long_edge = max(
            320, min(3840, int(self.dataset_config.get("max_long_edge", 960)))
        )
        event_frame_policy = str(
            self.dataset_config.get("event_frame_policy", "key_events")
        ).strip().lower()
        self.event_frame_policy = (
            event_frame_policy
            if event_frame_policy in {"all", "key_events", "none"}
            else "key_events"
        )
        max_run_image_mb = max(
            0.0, float(self.dataset_config.get("max_run_image_mb", 200))
        )
        self.image_quota_bytes = int(max_run_image_mb * 1024 * 1024)
        self.image_bytes_written = 0
        self.image_quota_exhausted = False
        self.last_sample_monotonic: float | None = None
        self._write_dataset_manifest()

    def _write_dataset_manifest(self) -> None:
        manifest = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "created_at_unix": time.time(),
            "continuous_capture_enabled": self.dataset_enabled,
            "battle_frame_interval_s": self.sample_interval_s,
            "jpeg_quality": self.sample_jpeg_quality,
            "event_jpeg_quality": self.event_jpeg_quality,
            "max_long_edge": self.max_long_edge,
            "event_frame_policy": self.event_frame_policy,
            "max_run_image_mb": self.image_quota_bytes / (1024 * 1024),
            "annotation_policy": {
                "bot_actions_are_ground_truth": False,
                "accepted_label_sources": ["human", "imported_expert"],
            },
        }
        with self.manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    @staticmethod
    def _safe_name(value: str) -> str:
        return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        prepared = image.convert("RGB") if image.mode != "RGB" else image.copy()
        if max(prepared.size) > self.max_long_edge:
            prepared.thumbnail(
                (self.max_long_edge, self.max_long_edge),
                Image.Resampling.LANCZOS,
            )
        return prepared

    def _save_jpeg(
        self,
        image: Image.Image,
        path: Path,
        *,
        quality: int,
    ) -> tuple[int, int] | None:
        if self.image_quota_exhausted:
            return None

        prepared = self._prepare_image(image)
        output = io.BytesIO()
        prepared.save(
            output,
            format="JPEG",
            quality=quality,
            optimize=True,
            progressive=True,
        )
        encoded = output.getvalue()
        if (
            self.image_quota_bytes > 0
            and self.image_bytes_written + len(encoded) > self.image_quota_bytes
        ):
            self.image_quota_exhausted = True
            return None

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
        self.image_bytes_written += len(encoded)
        return prepared.size

    def _should_save_event_frame(self, event_type: str) -> bool:
        if self.event_frame_policy == "all":
            return True
        if self.event_frame_policy == "none":
            return False
        # Continuous battle samples already preserve the surrounding image.  Saving
        # every bot action again adds no new ground-truth labels and was the largest
        # source of duplicated run data.
        return event_type != "battle_action"

    def record(
        self,
        event_type: str,
        image: Image.Image | None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.sequence += 1
        frame_relative: str | None = None
        stored_frame_size: tuple[int, int] | None = None
        frame_capture: str | None = None
        if image is not None and self._should_save_event_frame(event_type):
            filename = f"{self.sequence:06d}_{self._safe_name(event_type)}.jpg"
            frame_path = self.frames_dir / filename
            stored_frame_size = self._save_jpeg(
                image,
                frame_path,
                quality=self.event_jpeg_quality,
            )
            if stored_frame_size is not None:
                frame_relative = str(frame_path.relative_to(self.run_dir)).replace("\\", "/")
            else:
                frame_capture = "quota_exceeded"
        elif image is not None:
            frame_capture = "policy_skipped"

        event: dict[str, Any] = {
            "sequence": self.sequence,
            "timestamp_unix": time.time(),
            "event": event_type,
            "frame": frame_relative,
        }
        if image is not None:
            event["screen_size"] = [image.width, image.height]
        if stored_frame_size is not None:
            event["stored_frame_size"] = list(stored_frame_size)
        if frame_capture is not None:
            event["frame_capture"] = frame_capture
        if payload:
            event.update(payload)

        with self.events_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event

    def record_battle_sample(
        self,
        image: Image.Image,
        *,
        battle_index: int,
        observed_at_monotonic: float | None = None,
        observed_at_unix: float | None = None,
    ) -> SampleRecord | None:
        """Persist an unlabeled battle frame at a controlled sampling rate."""
        if not self.dataset_enabled or self.image_quota_exhausted:
            return None
        observed_at = (
            time.monotonic()
            if observed_at_monotonic is None
            else float(observed_at_monotonic)
        )
        if (
            self.last_sample_monotonic is not None
            and observed_at - self.last_sample_monotonic < self.sample_interval_s
        ):
            return None

        self.last_sample_monotonic = observed_at
        self.sample_sequence += 1
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.sample_sequence:06d}_battle.jpg"
        frame_path = self.samples_dir / filename
        stored_frame_size = self._save_jpeg(
            image,
            frame_path,
            quality=self.sample_jpeg_quality,
        )
        if stored_frame_size is None:
            self.sample_sequence -= 1
            return None
        sample = SampleRecord(
            sample_id=(
                f"{self.run_dir.name}-b{max(1, int(battle_index)):03d}"
                f"-s{self.sample_sequence:06d}"
            ),
            timestamp_unix=time.time() if observed_at_unix is None else float(observed_at_unix),
            frame=str(frame_path.relative_to(self.run_dir)).replace("\\", "/"),
            screen_size=stored_frame_size,
            battle_index=max(1, int(battle_index)),
        )
        with self.samples_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
        return sample
