"""Read-only, deterministic group splits for replay evaluation.

The split is based on whole battle IDs, never individual transitions.  A stable
hash makes the frozen set reproducible on both training machines without
copying raw screenshots or writing a local registry during evaluation.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Iterable, Mapping


EVALUATION_PROTOCOL_VERSION = "frozen_group_v1"


def content_fingerprint(rows: Iterable[Any]) -> str:
    """Fingerprint normalized row content, independent of input ordering."""
    normalized: list[str] = []
    for row in rows:
        if is_dataclass(row):
            value = asdict(row)
        elif isinstance(row, Mapping):
            value = dict(row)
        else:
            raise TypeError("实验数据行必须是 dataclass 或 mapping")
        normalized.append(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    payload = "\n".join(sorted(normalized))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def experiment_protocol_fingerprint(
    *,
    data_fingerprint: str,
    split: Mapping[str, object],
    filter_config: Mapping[str, object],
    model_config: Mapping[str, object],
) -> str:
    payload = {
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "data_fingerprint": str(data_fingerprint),
        "split": dict(split),
        "filter_config": dict(filter_config),
        "model_config": dict(model_config),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def champion_frozen_exposure(
    champion: Mapping[str, Any] | None,
    frozen_groups: Iterable[str],
) -> dict[str, object]:
    """Report whether an existing champion previously saw frozen evaluation games."""
    frozen = set(map(str, frozen_groups))
    if not champion:
        return {"status": "no_champion", "overlap_groups": [], "checked": True}
    manifest = champion.get("manifest")
    if not isinstance(manifest, Mapping):
        return {"status": "unknown_legacy_manifest", "overlap_groups": [], "checked": False}
    seen: set[str] = set()
    for key in ("train_battles", "validation_battles", "temporal_validation_battles", "transfer_battles"):
        values = manifest.get(key, [])
        if isinstance(values, list):
            seen.update(map(str, values))
    overlap = sorted(frozen & seen)
    return {
        "status": "overlap" if overlap else "unseen",
        "overlap_groups": overlap,
        "checked": True,
        "champion_version": champion.get("version"),
    }


def group_fingerprint(group_id: str) -> str:
    return hashlib.sha256(str(group_id).encode("utf-8")).hexdigest()


def group_set_fingerprint(groups: Iterable[str]) -> str:
    payload = "\n".join(sorted({str(group) for group in groups if str(group)}))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EvaluationGroups:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    frozen: tuple[str, ...]
    protocol_version: str = EVALUATION_PROTOCOL_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "train_groups": list(self.train),
            "validation_groups": list(self.validation),
            "frozen_groups": list(self.frozen),
            "train_fingerprints": [group_fingerprint(value) for value in self.train],
            "validation_fingerprints": [
                group_fingerprint(value) for value in self.validation
            ],
            "frozen_fingerprints": [
                group_fingerprint(value) for value in self.frozen
            ],
            "set_fingerprint": group_set_fingerprint(
                (*self.train, *self.validation, *self.frozen)
            ),
            "groups_are_disjoint": not bool(
                set(self.train)
                & (set(self.validation) | set(self.frozen))
                or set(self.validation) & set(self.frozen)
            ),
        }


def reserve_frozen_groups(
    groups: Iterable[str],
    *,
    validation_fraction: float,
    freeze_fraction: float,
    seed: int,
) -> EvaluationGroups:
    """Assign whole groups to train/validation/frozen deterministically."""
    unique = sorted({str(group) for group in groups if str(group)})
    if not 0.0 <= float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction 必须在 [0, 1) 内")
    if not 0.0 <= float(freeze_fraction) < 1.0:
        raise ValueError("freeze_fraction 必须在 [0, 1) 内")
    if float(validation_fraction) + float(freeze_fraction) >= 1.0:
        raise ValueError("validation_fraction 与 freeze_fraction 之和必须小于 1")
    ranked = sorted(
        unique,
        key=lambda group: hashlib.sha256(f"{int(seed)}:{group}".encode("utf-8")).hexdigest(),
    )
    freeze_count = int(len(ranked) * float(freeze_fraction))
    if freeze_fraction > 0 and len(ranked) >= 10:
        freeze_count = max(1, freeze_count)
    frozen = tuple(sorted(ranked[:freeze_count]))
    remaining = ranked[freeze_count:]
    validation_count = int(len(remaining) * float(validation_fraction))
    if validation_fraction > 0 and len(remaining) >= 4:
        validation_count = max(1, validation_count)
    validation = tuple(sorted(remaining[:validation_count]))
    train = tuple(sorted(remaining[validation_count:]))
    return EvaluationGroups(train=train, validation=validation, frozen=frozen)
