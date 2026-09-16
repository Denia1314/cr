"""Pinned, checksummed public battlefield baseline; never a trained champion."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from .atomic_file import atomic_write

REVISION = "f89ecbf309e698cb9fccdfb64a6ff2a930e6541e"
VERSION = "pbatch-yolov10m-20240831"
SOURCE = "https://github.com/Pbatch/ClashRoyaleBuildABot"
RELATIVE_ROOT = "models/battlefield/pretrained/pbatch"
ASSETS = {
    "units_M_480x352.onnx": (30896276, "4cf422370650b1d18647cffc05e02bf75fdc2fbdb1f8ef983f4b8d6f8ea908b3"),
    "side.onnx": (68762, "686bc75eae2ddc872b556534aeac8b4f5c6e891df76eef12e9fc91d406ab2583"),
}


def manifest() -> dict:
    return dict(version=VERSION, source=SOURCE, revision=REVISION,
                kind="public_pretrained_baseline", local_quality_validated=False,
                battle_acceptance=False, promoted=False,
                model_sha256=ASSETS["units_M_480x352.onnx"][1],
                files={name: dict(size=size, sha256=digest) for name, (size, digest) in ASSETS.items()},
                licenses={"repository_and_side": "MIT (Peter Batchelor, 2022)",
                          "detector_metadata": "AGPL-3.0 (Ultralytics)"})


def verify_asset(path: Path, name: str) -> bool:
    size, digest = ASSETS[name]
    return path.is_file() and path.stat().st_size == size and hashlib.sha256(path.read_bytes()).hexdigest() == digest


def assets_ready(root: Path) -> bool:
    return all(verify_asset(root / RELATIVE_ROOT / name, name) for name in ASSETS)


def install_assets(root: Path) -> dict:
    """Explicit setup operation; the battle loop never downloads weights."""
    directory = root / RELATIVE_ROOT
    for name, (size, digest) in ASSETS.items():
        path = directory / name
        if verify_asset(path, name):
            continue
        url = f"https://raw.githubusercontent.com/Pbatch/ClashRoyaleBuildABot/{REVISION}/clashroyalebuildabot/models/{name}"
        with urlopen(url, timeout=60) as response:
            data = response.read(size + 1)
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"战场权重校验失败：{name}")
        atomic_write(path, data)
    result = manifest()
    atomic_write(directory / "manifest.json", (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode())
    notice = root / "THIRD_PARTY_NOTICES.md"
    if notice.is_file():
        atomic_write(directory / "THIRD_PARTY_NOTICES.md", notice.read_bytes())
    return result


def sync_assets(root: Path, checkout: Path) -> dict:
    """Exchange only this allowlisted baseline through the private data repo."""
    published = received = 0
    local = root / RELATIVE_ROOT
    remote = checkout / "models/battlefield_pretrained" / VERSION
    for name in ASSETS:
        source, target = local / name, remote / name
        if verify_asset(source, name):
            if not verify_asset(target, name):
                atomic_write(target, source.read_bytes())
                published += 1
        elif target.is_file():
            if not verify_asset(target, name):
                raise ValueError(f"共享战场权重校验失败：{name}")
            atomic_write(source, target.read_bytes())
            received += 1
    if assets_ready(root):
        data = (json.dumps(manifest(), ensure_ascii=False, indent=2) + "\n").encode()
        atomic_write(local / "manifest.json", data)
        atomic_write(remote / "manifest.json", data)
        notice = root / "THIRD_PARTY_NOTICES.md"
        if notice.is_file():
            atomic_write(local / "THIRD_PARTY_NOTICES.md", notice.read_bytes())
            atomic_write(remote / "THIRD_PARTY_NOTICES.md", notice.read_bytes())
    return dict(published_files=published, received_files=received,
                ready=assets_ready(root), version=VERSION, local_quality_validated=False)
