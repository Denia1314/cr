"""Local-only, atomic records for resumable candidate jobs (no model activation)."""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def read_json(path: Path, default=None):
    if not path.exists():
        return {} if default is None else default
    # A corrupt ledger must not silently reset and retrain/promote duplicates.
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


class LearningStore:
    def __init__(self, root: Path):
        self.root = root.resolve() / "training" / "self_learning"

    def state_path(self, identity: str) -> Path:
        if len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
            raise ValueError("无效学习身份")
        return self.root / "identities" / (identity + ".json")

    def load(self, identity: str) -> dict:
        return read_json(self.state_path(identity), {"schema": 1, "identity": identity,
                         "consumed": [], "jobs": [], "last_training_at": 0})

    def save(self, state: dict) -> None:
        atomic_json(self.state_path(state["identity"]), state)

    def status(self, phase: str, reason: str, **details) -> dict:
        value = {"schema": 1, "phase": phase, "reason": reason,
                 "updated_at": time.time(), **details}
        atomic_json(self.root / "status.json", value)
        print(f"[自主学习] {phase}：{reason}", flush=True)
        return value

    def current_status(self) -> dict:
        return read_json(self.root / "status.json", {"phase": "idle", "reason": "尚未开始"})
