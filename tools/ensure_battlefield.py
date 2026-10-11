"""Install the pinned battlefield baseline and the matching ONNX runtime."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ORT_VERSION = "1.23.2"  # CUDA 12 / cuDNN 9, matching the project's PyTorch cu128.


def ensure(root=ROOT):
    from crbot.battlefield_assets import install_assets
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    training = config.get("training", {})
    if not training.get("pretrained_enabled", True):
        return {"enabled": False}
    use_cuda = False
    if os.environ.get("CRBOT_COMPUTE_DEVICE", str(training.get("device", "auto"))).lower() != "cpu":
        try:
            import torch
            use_cuda = torch.cuda.is_available()
        except ImportError:
            pass
    try:
        import onnxruntime as ort
        ready = ort.__version__ == ORT_VERSION and (not use_cuda or "CUDAExecutionProvider" in ort.get_available_providers())
    except ImportError:
        ready = False
    if not ready:
        # CPU and GPU wheels share an import namespace; never leave both installed.
        from importlib.metadata import PackageNotFoundError, version
        package = "onnxruntime-gpu" if use_cuda else "onnxruntime"
        other = "onnxruntime" if use_cuda else "onnxruntime-gpu"
        try:
            version(other)
        except PackageNotFoundError:
            pass
        else:
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", other], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                        f"{package}=={ORT_VERSION}"], check=True)
    # Cache only the fixed public baseline, never personal models or replays.
    from tools.ensure_gpu import shared_home
    from crbot.battlefield_assets import ASSETS, RELATIVE_ROOT, verify_asset
    from crbot.atomic_file import atomic_write
    cache = shared_home() / 'public-assets'
    for name in ASSETS:
        source = root / RELATIVE_ROOT / name
        target = cache / RELATIVE_ROOT / name
        if verify_asset(source, name) and not verify_asset(target, name):
            atomic_write(target, source.read_bytes())
    install_assets(cache)
    for name in ASSETS:
        source = cache / RELATIVE_ROOT / name
        target = root / RELATIVE_ROOT / name
        if not verify_asset(target, name):
            atomic_write(target, source.read_bytes())
    result = install_assets(root)
    print(f"Battlefield baseline ready: {result['version']} (local accuracy not yet certified)", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    ensure()
    if args.verify:
        # A fresh interpreter is necessary if setup replaced an imported runtime.
        subprocess.run([sys.executable, "-m", "tools.verify_battlefield"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
