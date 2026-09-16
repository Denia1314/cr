"""Small Tk animations; a single owner cancels callbacks on widget destruction."""
from __future__ import annotations

import math
import os
import time
import tkinter as tk


def motion_enabled() -> bool:
    return os.environ.get("CRBOT_REDUCED_MOTION", "0") != "1"


def mix(start: str, end: str, fraction: float) -> str:
    fraction = max(0.0, min(1.0, fraction))
    a = tuple(int(start[i:i+2], 16) for i in (1, 3, 5))
    b = tuple(int(end[i:i+2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{round(x+(y-x)*fraction):02x}" for x, y in zip(a, b))


class Animator:
    def __init__(self, widget: tk.Misc):
        self.widget = widget
        self.jobs = {}
        widget.bind("<Destroy>", self._destroy, add="+")

    def _destroy(self, event):
        if event.widget == self.widget:
            self.cancel()

    def cancel(self, key=None):
        for name in list(self.jobs):
            if key is None or name == key:
                job = self.jobs.pop(name)
                try:
                    self.widget.after_cancel(job)
                except tk.TclError:
                    pass

    def tween(self, key, update, duration=180, complete=None):
        self.cancel(key)
        if not motion_enabled():
            update(1.0)
            if complete:
                complete()
            return
        started = time.perf_counter()
        def tick():
            self.jobs.pop(key, None)
            if not self.widget.winfo_exists():
                return
            progress = min(1.0, (time.perf_counter()-started)*1000/duration)
            update(1-(1-progress)**3)
            if progress < 1:
                self.jobs[key] = self.widget.after(16, tick)
            elif complete:
                complete()
        tick()

    def pulse(self, key, update, period=1800):
        self.cancel(key)
        if not motion_enabled():
            update(1.0)
            return
        started = time.perf_counter()
        def tick():
            if not self.widget.winfo_exists():
                return
            phase = (time.perf_counter()-started)*1000/period
            update((1-math.cos(phase*math.tau))/2)
            self.jobs[key] = self.widget.after(40, tick)
        tick()


def fade_in(root: tk.Tk):
    animator = Animator(root)
    if motion_enabled():
        root.attributes("-alpha", 0.0)
        animator.tween("fade", lambda t: root.attributes("-alpha", t), 260)
    return animator


class StartupView:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Royal Lab · 正在加载")
        root.configure(bg="#0B1120")
        width, height = 620, 360
        root.geometry(f"{width}x{height}+{max(0,(root.winfo_screenwidth()-width)//2)}+{max(0,(root.winfo_screenheight()-height)//2)}")
        root.resizable(False, False)
        self.frame = tk.Frame(root, bg="#0B1120")
        self.frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(self.frame, width=100, height=100, bg="#0B1120", highlightthickness=0)
        self.canvas.pack(pady=(24, 10))
        self.canvas.create_oval(10, 10, 90, 90, outline="#1D304B", width=4)
        self.arc = self.canvas.create_arc(10, 10, 90, 90, outline="#44D7E8", width=4, style="arc", extent=90)
        self.canvas.create_text(50, 50, text="RL", fill="#F4F7FF", font=("Segoe UI", 20, "bold"))
        tk.Label(self.frame, text="Royal Lab", bg="#0B1120", fg="#F4F7FF", font=("Segoe UI", 24, "bold")).pack()
        self.status = tk.StringVar(value="正在加载内置运行环境…")
        tk.Label(self.frame, textvariable=self.status, bg="#0B1120", fg="#A6B7CE", font=("Microsoft YaHei UI", 10), wraplength=550).pack(pady=12)
        tk.Label(self.frame, text="本地运行 · 离线人机训练", bg="#0B1120", fg="#8195B0", font=("Microsoft YaHei UI", 9)).pack()
        self.animator = Animator(self.canvas)
        self._angle = 0
        self._spin()

    def _spin(self):
        if motion_enabled():
            self._angle = (self._angle+6) % 360
            self.canvas.itemconfigure(self.arc, start=-self._angle, extent=90+35*math.sin(math.radians(self._angle)))
            self.animator.jobs['spin'] = self.canvas.after(25, self._spin)

    def finish(self):
        self.animator.cancel()
        self.frame.destroy()
        self.root.resizable(True, True)
