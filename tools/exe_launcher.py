"""Windowed launcher for the external project Python environment."""
from pathlib import Path
import ctypes
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from tkinter import ttk

REQUIRED = ('tools/ensure_gpu.py', 'launch_ui.pyw', 'crbot/gui.py', 'requirements.txt')
NO_CONSOLE = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def project_root() -> Path:
    return (Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False)
            else Path(__file__).resolve().parent.parent)


def child_environment() -> dict[str, str]:
    env = os.environ.copy()
    # PyInstaller's Tk paths refer to its disposable extraction directory.
    # External Python must discover its own Tcl/Tk and Python libraries.
    for key in list(env):
        if key in {'TCL_LIBRARY', 'TK_LIBRARY', 'PYTHONHOME', 'PYTHONPATH'} or key.startswith('_PYI_'):
            env.pop(key, None)
    bundle = str(getattr(sys, '_MEIPASS', ''))
    if bundle:
        env['PATH'] = os.pathsep.join(part for part in env.get('PATH', '').split(os.pathsep)
                                      if not part.casefold().startswith(bundle.casefold()))
    env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', CRBOT_NO_CONSOLE='1')
    return env


def reset_dll_search() -> None:
    # All launcher imports/Tk initialization precede this. No bundled helpers run later.
    if sys.platform == 'win32' and getattr(sys, 'frozen', False):
        ctypes.windll.kernel32.SetDllDirectoryW(None)


def find_python(root: Path) -> Path:
    candidates = [root / '.venv/Scripts/python.exe',
                  Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe']
    if not getattr(sys, 'frozen', False):
        candidates.append(Path(sys.executable))
    system = shutil.which('python')
    if system and 'windowsapps' not in system.casefold():
        candidates.append(Path(system))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError('未找到 Python 3.10 或更高版本。请先安装 Python，或恢复项目 .venv 环境。')


class Cancelled(Exception):
    pass


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if sys.platform == 'win32':
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=NO_CONSOLE, timeout=15)
    else:
        process.terminate()
    process.wait(timeout=15)


def run_hidden(args, root, env, log, cancel, timeout=1800) -> int:
    log.write('\n> ' + subprocess.list2cmdline([str(arg) for arg in args]) + '\n')
    log.flush()
    process = subprocess.Popen([str(arg) for arg in args], cwd=root, env=env,
                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                               creationflags=NO_CONSOLE)
    deadline = time.monotonic() + timeout
    while process.poll() is None:
        if cancel.wait(.1):
            stop_process(process)
            raise Cancelled()
        if time.monotonic() > deadline:
            stop_process(process)
            raise RuntimeError('准备环境超时，请查看启动日志后重试。')
    return process.returncode


def prepare_and_launch(root, events, cancel) -> None:
    try:
        with (root / 'launcher.log').open('w', encoding='utf-8') as log:
            env = child_environment()
            missing = [name for name in REQUIRED if not (root / name).is_file()]
            if missing:
                raise RuntimeError('请将 EXE 放在完整项目根目录。缺少：' + ', '.join(missing))
            python = find_python(root)
            events.put(('status', '正在检查 Python 和 GPU 环境…'))
            if run_hidden([python, root / 'tools/ensure_gpu.py'], root, env, log, cancel):
                raise RuntimeError('GPU 环境准备失败，详细原因见下方日志。')
            # The bootstrap may have just created a local CUDA environment.
            python = find_python(root)
            events.put(('status', '正在检查界面依赖…'))
            check = [python, '-c', 'import cv2,numpy,PIL,tkinter']
            if run_hidden(check, root, env, log, cancel, 90):
                events.put(('status', '首次使用：正在安装依赖，请保持网络连接…'))
                if run_hidden([python, '-m', 'pip', 'install', '-r', root / 'requirements.txt'], root, env, log, cancel):
                    raise RuntimeError('依赖安装失败，详细原因见下方日志。')
                if run_hidden(check, root, env, log, cancel, 90):
                    raise RuntimeError('依赖仍不完整，请查看日志。')
            if cancel.is_set():
                raise Cancelled()
            events.put(('status', '正在打开 Royal Lab…'))
            pythonw = python.with_name('pythonw.exe')
            executable = pythonw if pythonw.is_file() else python
            with tempfile.TemporaryDirectory(prefix='royal-lab-ready-') as directory:
                ready = Path(directory) / 'ready.txt'
                with (root / 'ui-process.log').open('w', encoding='utf-8') as output:
                    process = subprocess.Popen(
                        [str(executable), str(root / 'launch_ui.pyw'), '--ready-file', str(ready)],
                        cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=output,
                        stderr=subprocess.STDOUT, creationflags=NO_CONSOLE)
                    deadline = time.monotonic() + 90
                    while not ready.exists():
                        if cancel.wait(.1):
                            stop_process(process)
                            raise Cancelled()
                        if process.poll() is not None:
                            raise RuntimeError('主界面启动失败，请查看 ui-process.log 和 startup-error.log。')
                        if time.monotonic() > deadline:
                            stop_process(process)
                            raise RuntimeError('主界面未能在 90 秒内就绪，请查看 ui-process.log。')
                    if process.poll() is not None:
                        raise RuntimeError('主界面已提前退出，请查看 ui-process.log。')
                    log.write(f'UI ready: pid={process.pid}\n')
        events.put(('ready', None))
    except Cancelled:
        events.put(('cancelled', None))
    except Exception as exc:
        events.put(('error', str(exc)))


def main() -> int:
    root = project_root()
    if '--check' in sys.argv:
        missing = [name for name in REQUIRED if not (root / name).is_file()]
        (root / 'launcher-check.log').write_text(f'Project: {root}\nMissing: {missing}\n', encoding='utf-8')
        return int(bool(missing))
    window = tk.Tk()
    window.title('Royal Lab · 正在启动')
    window.configure(bg='#0B1120')
    window.geometry('620x240')
    window.minsize(620, 240)
    tk.Label(window, text='Royal Lab', font=('Microsoft YaHei UI', 24, 'bold'),
             bg='#0B1120', fg='#F4F7FF').pack(anchor='w', padx=28, pady=(24, 8))
    status = tk.StringVar(value='正在准备启动…')
    tk.Label(window, textvariable=status, font=('Microsoft YaHei UI', 11),
             wraplength=560, justify='left', bg='#0B1120', fg='#A6B7CE').pack(anchor='w', padx=28)
    progress = ttk.Progressbar(window, mode='indeterminate')
    progress.pack(fill='x', padx=28, pady=18)
    progress.start()
    tk.Label(window, text='首次使用可能需要下载依赖，完成后自动进入控制台。',
             bg='#0B1120', fg='#8195B0').pack(anchor='w', padx=28)
    events = queue.Queue()
    cancel = threading.Event()
    state = {'finished': False, 'exit': 0}

    def close():
        if state['finished']:
            window.destroy()
        else:
            cancel.set()
            status.set('正在取消启动…')

    def poll():
        try:
            while True:
                kind, value = events.get_nowait()
                if kind == 'status':
                    status.set(value)
                elif kind in ('ready', 'cancelled'):
                    window.destroy()
                    return
                elif kind == 'error':
                    state.update(finished=True, exit=1)
                    progress.stop()
                    status.set(value)
                    window.title('Royal Lab · 启动失败')
                    window.geometry('720x500')
                    details = tk.Text(window, bg='#101B2E', fg='#F4F7FF', wrap='word', height=12)
                    details.pack(fill='both', expand=True, padx=28, pady=16)
                    for filename in ('launcher.log', 'ui-process.log'):
                        path = root / filename
                        if path.exists():
                            details.insert('end', f'{path}\n' + path.read_text(encoding='utf-8', errors='replace')[-6000:] + '\n')
                    details.configure(state='disabled')
                    return
        except queue.Empty:
            pass
        window.after(100, poll)

    window.protocol('WM_DELETE_WINDOW', close)
    reset_dll_search()
    threading.Thread(target=prepare_and_launch, args=(root, events, cancel), daemon=True).start()
    window.after(100, poll)
    window.mainloop()
    return state['exit']


if __name__ == '__main__':
    raise SystemExit(main())
