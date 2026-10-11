"""Prepare reusable per-user Python/CUDA before a Windows launcher starts."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import hashlib
import time
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
CUDA_CHECK = (
    "import torch; "
    "x=torch.ones((32,32),device='cuda:0'); "
    "assert (x@x).sum().item()==32768; "
    "print('GPU verified:',torch.cuda.get_device_name(0),'torch='+torch.__version__)"
)


def shared_home():
    return Path(os.environ.get('LOCALAPPDATA') or Path.home() / 'AppData/Local') / 'RoyalLab'


def environment_root():
    custom = os.environ.get('CRBOT_PYTHON_ENV')
    if custom:
        path = Path(custom).expanduser()
        if not path.is_absolute():
            raise ValueError('CRBOT_PYTHON_ENV must be an absolute path.')
        return path
    return shared_home() / 'python-envs' / f'python-{sys.version_info.major}.{sys.version_info.minor}'


def environment_python():
    return environment_root() / 'Scripts/python.exe'


@contextmanager
def environment_lock():
    import msvcrt
    root = environment_root()
    root.parent.mkdir(parents=True, exist_ok=True)
    with root.with_suffix(root.suffix + '.setup.lock').open('a+b') as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        deadline = time.monotonic() + 1800
        announced = False
        while True:
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                if not announced:
                    print('Another launcher is preparing the shared environment; waiting...', flush=True)
                    announced = True
                if time.monotonic() >= deadline:
                    raise TimeoutError('Shared environment setup is still busy; retry after it finishes.')
                time.sleep(.25)
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def create_environment():
    python = environment_python()
    if python.exists():
        return 0
    # A local venv may be the bootstrap interpreter; keep the shared environment's
    # base outside that project so deleting a checkout cannot break it.
    base = getattr(sys, '_base_executable', sys.executable)
    return run([base, '-m', 'venv', str(environment_root())]).returncode


def ensure_requirements(cuda=False):
    requirement = 'requirements-training.txt' if cuda else 'requirements.txt'
    files = [ROOT / 'requirements.txt'] + ([ROOT / requirement] if cuda else [])
    signature = hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()
    marker = environment_root() / ('.requirements-cuda' if cuda else '.requirements-cpu')
    checked = run([str(environment_python()), '-c',
                   'import cv2,numpy,PIL' + ('; from ultralytics import YOLO; import cupy' if cuda else '')], capture=True)
    if checked.returncode or not marker.is_file() or marker.read_text() != signature:
        if run([str(environment_python()), '-m', 'pip', 'install', '--disable-pip-version-check',
                '-r', str(ROOT / requirement)]).returncode:
            return 1
        if run([str(environment_python()), '-m', 'pip', 'check'], capture=True).returncode:
            return 1
        marker.write_text(signature)
    return 0


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
    python = environment_python()
    if not python.exists():
        if create_environment():
            return 1
    checked = run([str(python), '-c', CUDA_CHECK], capture=True)
    if checked.returncode:
        print(checked.stdout, flush=True)
        print(f'Installing CUDA PyTorch into {environment_root()} (one-time preparation)...', flush=True)
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
        with environment_lock():
            print(f'Shared Python environment: {environment_root()}', flush=True)
            result = create_environment()
            if result == 0 and '--environment-only' not in sys.argv:
                cuda = '--force' in sys.argv or (os.environ.get('CRBOT_COMPUTE_DEVICE', '').lower() != 'cpu' and has_nvidia())
                result = ensure_gpu(force=cuda)
                if result == 0:
                    result = ensure_requirements(cuda)
                if result == 0:
                    result = run([str(environment_python()), str(ROOT / 'tools/ensure_battlefield.py')]).returncode
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f'GPU setup error: {exc}', flush=True)
        result = 1
    if result:
        print('GPU setup failed. Check the driver/network/Python error above, then retry '
              'setup_gpu.bat. To explicitly use CPU, set CRBOT_COMPUTE_DEVICE=cpu.', flush=True)
    return result


if __name__ == '__main__':
    if '--print-environment' in sys.argv:
        print(environment_root())
    else:
        raise SystemExit(main())
