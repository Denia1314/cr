"""Same-process packaged desktop application, including background worker dispatch."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
import traceback

from .app_paths import configure_bundled_tools, initialize_data, resource_root


def main() -> int:
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--smoke-ui", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--self-learning", action="store_true")
    parser.add_argument("--worker-log", type=Path)
    parser.add_argument("--cli", action="store_true")
    raw = sys.argv[1:]
    if '--cli' in raw:
        split = raw.index('--cli')
        args = parser.parse_args(raw[:split])
        args.cli, rest = True, raw[split+1:]
    else:
        args, rest = parser.parse_known_args(raw)
    if args.data_dir:
        os.environ["CRBOT_DATA_DIR"] = str(args.data_dir.resolve())
    if args.report:
        args.report = args.report.resolve()
    root_path = initialize_data()
    configure_bundled_tools()
    os.chdir(root_path)
    log_path = args.worker_log or root_path / "desktop.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = log
    if args.self_learning:
        sys.argv = [sys.argv[0], *rest]
        from .self_learning import main as learn
        learn()
        return 0
    if args.cli:
        sys.argv = [sys.argv[0], *rest]
        from .cli import main as cli
        return cli()
    if args.self_test:
        from .package_check import run_checks
        return run_checks(args.report or root_path / "package-check.json")
    import tkinter as tk
    from .animations import StartupView
    from . import release_label
    if sys.platform == 'win32':
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    window = tk.Tk()
    icon = resource_root() / 'royal-lab.ico'
    if icon.is_file():
        window.iconbitmap(str(icon))
    startup = StartupView(window)
    window.update_idletasks()
    results = queue.Queue()

    def load():
        try:
            from .gui import RoyalTrainerApp
            results.put((RoyalTrainerApp, None))
        except Exception:
            results.put((None, traceback.format_exc()))

    def finish():
        try:
            app_class, error = results.get_nowait()
        except queue.Empty:
            window.after(40, finish)
            return
        if error:
            print(error, flush=True)
            startup.status.set("加载失败，详情已写入：" + str(log_path))
            startup.animator.cancel()
            return
        try:
            startup.finish()
            app = app_class(window, root_path / "config.json", start_services=not args.smoke_ui)
            ready_seconds = round(time.perf_counter() - started, 3)
            if args.smoke_ui:
                def complete():
                    report = {"ok": True, "title": window.title(), "frozen": bool(getattr(sys, "frozen", False)),
                              "executable": sys.executable, "data_root": str(root_path),
                              "width": window.winfo_width(), "height": window.winfo_height(),
                              "ready_seconds": ready_seconds,
                              "version_options": list(app.engine_combo['values'])}
                    (args.report or root_path / "ui-check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                    app._on_close()
                window.after(1500, complete)
            print(release_label() + f" · packaged UI ready in {ready_seconds:.3f}s", flush=True)
        except Exception:
            details = traceback.format_exc()
            print(details, flush=True)
            from tkinter import messagebox
            messagebox.showerror("Royal Lab 启动失败", f"{details}\n日志：{log_path}", parent=window)
            window.destroy()

    threading.Thread(target=load, name="desktop-import", daemon=True).start()
    window.after(40, finish)
    window.mainloop()
    return 0
