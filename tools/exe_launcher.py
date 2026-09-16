"""Small Windows launcher; the project owns its Python and CUDA environment."""
from pathlib import Path
import os
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox


def project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def main() -> int:
    root = project_root()
    required = ("start_ui.bat", "tools/python_env.bat", "launch_ui.pyw", "crbot/gui.py")
    missing = [name for name in required if not (root / name).is_file()]
    if "--check" in sys.argv:
        (root / "launcher-check.log").write_text(
            f"Project: {root}\nMissing: {missing}\n", encoding="utf-8")
        return int(bool(missing))
    window = tk.Tk()
    window.withdraw()
    try:
        if missing:
            raise RuntimeError("请将 RoyalLab.exe 放在完整项目的根目录，与 start_ui.bat 放在一起。\n缺少：" + ", ".join(missing))
        # A fixed relative command avoids quoting project paths through cmd.exe.
        # Keep the preparation console visible for downloads and actionable errors.
        subprocess.Popen(
            [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", "start_ui.bat"],
            cwd=root, creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
    except (OSError, RuntimeError) as exc:
        messagebox.showerror("Royal Lab 启动失败", str(exc), parent=window)
        return 1
    finally:
        window.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
