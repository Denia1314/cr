"""Separate bundled read-only resources from persistent user data."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys


def frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def resource_root() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))


def data_root() -> Path:
    if os.environ.get("CRBOT_DATA_DIR"):
        return Path(os.environ["CRBOT_DATA_DIR"]).expanduser().resolve()
    if not frozen():
        return resource_root()
    beside = Path(sys.executable).resolve().parent
    if (beside / "config.json").is_file():
        return beside
    return Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "RoyalLab"


def initialize_data() -> Path:
    root = data_root()
    root.mkdir(parents=True, exist_ok=True)
    if frozen():
        seed = resource_root() / "defaults"
        for source in seed.rglob("*"):
            if not source.is_file():
                continue
            target = root / source.relative_to(seed)
            target.parent.mkdir(parents=True, exist_ok=True)
            from .battlefield_assets import ASSETS, RELATIVE_ROOT, verify_asset
            if source.relative_to(seed).parent.as_posix() == RELATIVE_ROOT and source.name in ASSETS:
                if not verify_asset(target, source.name):
                    from .atomic_file import atomic_write
                    atomic_write(target, source.read_bytes())
                continue
            # Preserve calibration, local catalogs, and all user edits on upgrades.
            if not target.exists():
                try:
                    with source.open("rb") as src, target.open("xb") as dst:
                        shutil.copyfileobj(src, dst)
                except FileExistsError:
                    pass
    return root


def configure_bundled_tools() -> None:
    if not frozen():
        return
    vendor = resource_root() / "vendor"
    paths = [vendor / "git/cmd", vendor / "git/mingw64/bin", vendor / "gh"]
    os.environ["PATH"] = os.pathsep.join([str(p) for p in paths] + [os.environ.get("PATH", "")])


def learning_command(root: Path, request: Path, log: Path) -> list[str]:
    if frozen():
        return [sys.executable, "--self-learning", "--worker-log", str(log),
                "--root", str(root), "--request", str(request)]
    return [sys.executable, "-X", "utf8", "-m", "crbot.self_learning",
            "--root", str(root), "--request", str(request)]
