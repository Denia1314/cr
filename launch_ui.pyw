from __future__ import annotations

import sys
import argparse
import traceback
from datetime import datetime
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent
ERROR_LOG = APP_ROOT / "startup-error.log"


def report_startup_error(show_dialog: bool = True) -> None:
    details = traceback.format_exc()
    report = (
        f"[{datetime.now().isoformat(timespec='seconds')}]\n"
        f"Python: {sys.executable}\n\n{details}"
    )
    try:
        ERROR_LOG.write_text(report, encoding="utf-8")
    except OSError:
        pass

    if not show_dialog:
        if sys.stderr is not None:
            print(report, file=sys.stderr, flush=True)
        return

    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Royal Lab 启动失败",
            "控制台启动失败，详细错误已写入：\n"
            f"{ERROR_LOG}\n\n"
            "通常是 Python 依赖不完整。请在项目目录执行：\n"
            "python -m pip install -r requirements.txt",
            parent=root,
        )
        root.destroy()
    except Exception:
        pass


def run() -> None:
    ready_file = None
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--ready-file', type=Path)
        args = parser.parse_args()
        ready_file = args.ready_file
        from crbot.gui import main

        def ready() -> None:
            if ready_file:
                ready_file.write_text('ready', encoding='utf-8')
        main(on_ready=ready)
    except Exception:
        report_startup_error(show_dialog=ready_file is None)
        raise SystemExit(1)


if __name__ == "__main__":
    run()
