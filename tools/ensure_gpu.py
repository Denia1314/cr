"""Prepare CUDA in the project environment before a Windows launcher starts."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CUDA_CHECK = (
    "import torch; "
    "x=torch.ones((32,32),device='cuda:0'); "
    "assert (x@x).sum().item()==32768; "
    "print('GPU verified:',torch.cuda.get_device_name(0),'torch='+torch.__version__)"
)


def run(args, *, capture=False):
    return subprocess.run(args, cwd=ROOT, text=True,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.STDOUT if capture else None,
                          timeout=60 if capture else None,
                          creationflags=(getattr(subprocess, 'CREATE_NO_WINDOW', 0)
                                         if os.environ.get('CRBOT_NO_CONSOLE') == '1' else 0))


def has_nvidia():
    candidates = [shutil.which('nvidia-smi'),
                  str(Path(os.environ.get('WINDIR', r'C:\Windows')) / 'System32/nvidia-smi.exe'),
                  str(Path(os.environ.get('ProgramFiles', r'C:\Program Files')) /
                      'NVIDIA Corporation/NVSMI/nvidia-smi.exe')]
    for path in candidates:
        if not path or not Path(path).is_file():
            continue
        result = run([path, '--query-gpu=name', '--format=csv,noheader'], capture=True)
        if result.returncode == 0 and result.stdout.strip():
            print('NVIDIA detected:', result.stdout.strip(), flush=True)
            return True
    return False


def ensure_gpu(force=False):
    if not force and os.environ.get('CRBOT_COMPUTE_DEVICE', '').lower() == 'cpu':
        print('GPU setup skipped: CRBOT_COMPUTE_DEVICE=cpu.', flush=True)
        return 0
    if not force and not has_nvidia():
        print('No NVIDIA driver detected; using existing CPU environment. '
              'For an NVIDIA PC, install its driver and run setup_gpu.bat.', flush=True)
        return 0
    python = ROOT / '.venv/Scripts/python.exe'
    if not python.exists():
        if run([sys.executable, '-m', 'venv', str(ROOT / '.venv')]).returncode:
            return 1
    checked = run([str(python), '-c', CUDA_CHECK], capture=True)
    if checked.returncode:
        print(checked.stdout, flush=True)
        print('Installing CUDA PyTorch into .venv (large download on first launch)...', flush=True)
        # Explicit local versions also replace an already installed CPU build.
        args = [str(python), '-m', 'pip', 'install', '--upgrade',
                'torch==2.11.0+cu128', 'torchvision==0.26.0+cu128',
                '--index-url', 'https://download.pytorch.org/whl/cu128']
        if run(args).returncode:
            return 1
    else:
        print(checked.stdout, flush=True)
    dependencies = run([str(python), '-c', 'import cv2,numpy,PIL; from ultralytics import YOLO'], capture=True)
    if dependencies.returncode:
        if run([str(python), '-m', 'pip', 'install', '-r', 'requirements-training.txt']).returncode:
            return 1
    if run([str(python), '-m', 'pip', 'check'], capture=True).returncode:
        run([str(python), '-m', 'pip', 'check'])
        return 1
    return run([str(python), '-c', CUDA_CHECK]).returncode


def main():
    try:
        result = ensure_gpu('--force' in sys.argv)
        if result == 0:
            python = ROOT / '.venv/Scripts/python.exe'
            result = run([str(python) if python.is_file() else sys.executable,
                          str(ROOT / 'tools/ensure_battlefield.py')]).returncode
    except (OSError, subprocess.SubprocessError) as exc:
        print(f'GPU setup error: {exc}', flush=True)
        result = 1
    if result:
        print('GPU setup failed. Check the driver/network/Python error above, then retry '
              'setup_gpu.bat. To explicitly use CPU, set CRBOT_COMPUTE_DEVICE=cpu.', flush=True)
    return result


if __name__ == '__main__':
    raise SystemExit(main())
