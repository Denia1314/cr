from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

from PIL import Image, ImageTk

from .adb import MumuDevice
from .config import resolve_project_path, save_config


class CalibrationWindow:
    def __init__(
        self,
        root: tk.Tk,
        device: MumuDevice,
        config: dict[str, Any],
        config_path: Path,
    ):
        self.root = root
        self.device = device
        self.config = config
        self.config_path = config_path
        self.image: Image.Image | None = None
        self.display_image: Image.Image | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.scale = 1.0
        self.start: tuple[int, int] | None = None
        self.selection: tuple[int, int, int, int] | None = None
        self.rectangle: int | None = None

        self.background = "#090C17"
        self.surface = "#101522"
        self.card = "#151B2B"
        self.border = "#273149"
        self.text = "#F4F7FF"
        self.muted = "#8F9BB3"
        self.blue = "#4D8DFF"
        self.green = "#43D6A0"
        self.font = "Microsoft YaHei UI"

        root.title("Royal Lab · 画面标定")
        root.geometry("1180x820")
        root.minsize(920, 680)
        root.configure(background=self.background)

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(
            "Royal.TCombobox",
            fieldbackground=self.surface,
            background=self.card,
            foreground=self.text,
            arrowcolor=self.muted,
            bordercolor=self.border,
            lightcolor=self.border,
            darkcolor=self.border,
            padding=7,
        )
        style.map(
            "Royal.TCombobox",
            fieldbackground=[("readonly", self.surface)],
            foreground=[("readonly", self.text)],
            selectbackground=[("readonly", self.surface)],
            selectforeground=[("readonly", self.text)],
        )

        header = tk.Frame(root, background=self.surface, height=74)
        header.pack(fill="x")
        header.pack_propagate(False)
        title_box = tk.Frame(header, background=self.surface)
        title_box.pack(side="left", padx=24, pady=12)
        tk.Label(
            title_box,
            text="画面标定",
            font=(self.font, 17, "bold"),
            foreground=self.text,
            background=self.surface,
        ).pack(anchor="w")
        tk.Label(
            title_box,
            text="框选稳定 UI，让训练器只在已确认的离线人机页面工作",
            font=(self.font, 8),
            foreground=self.muted,
            background=self.surface,
        ).pack(anchor="w")

        controls = tk.Frame(
            root,
            background=self.card,
            highlightbackground=self.border,
            highlightthickness=1,
        )
        controls.pack(fill="x", padx=20, pady=(18, 10))
        top = tk.Frame(controls, background=self.card)
        top.pack(fill="x", padx=15, pady=13)
        tk.Label(
            top,
            text="标定目标",
            font=(self.font, 9, "bold"),
            foreground=self.muted,
            background=self.card,
        ).pack(side="left", padx=(0, 10))
        self.target_var = tk.StringVar()
        targets = [
            target
            for target in config["workflow"]
            if target["name"] == "offline_ai_marker"
        ]
        target_values = [target["name"] for target in targets]
        self.target_combo = ttk.Combobox(
            top,
            textvariable=self.target_var,
            values=target_values,
            state="readonly",
            width=28,
            style="Royal.TCombobox",
        )
        self.target_combo.pack(side="left", padx=(0, 12))
        if target_values:
            self.target_combo.current(0)
        self.target_combo.bind("<<ComboboxSelected>>", lambda _event: self._update_description())
        tk.Button(
            top,
            text="刷新截图",
            command=self.refresh,
            font=(self.font, 9, "bold"),
            foreground=self.text,
            background="#252E45",
            activeforeground=self.text,
            activebackground="#303B56",
            relief="flat",
            borderwidth=0,
            padx=15,
            pady=8,
            cursor="hand2",
        ).pack(side="left", padx=4)
        tk.Button(
            top,
            text="保存当前框",
            command=self.save_selection,
            font=(self.font, 9, "bold"),
            foreground=self.text,
            background=self.blue,
            activeforeground=self.text,
            activebackground="#6AA0FF",
            relief="flat",
            borderwidth=0,
            padx=15,
            pady=8,
            cursor="hand2",
        ).pack(side="left", padx=4)

        self.description_var = tk.StringVar()
        tk.Label(
            controls,
            textvariable=self.description_var,
            font=(self.font, 9),
            foreground=self.green,
            background=self.card,
        ).pack(fill="x", padx=16, pady=(0, 6))
        tk.Label(
            controls,
            text="切到 MuMu 对应页面 → 刷新截图 → 围绕稳定文字或图标拖框 → 保存当前框",
            font=(self.font, 8),
            foreground=self.muted,
            background=self.card,
        ).pack(fill="x", padx=16, pady=(0, 13))

        canvas_frame = tk.Frame(
            root,
            background="#070A12",
            highlightbackground=self.border,
            highlightthickness=1,
        )
        canvas_frame.pack(fill="both", expand=True, padx=20, pady=(0, 18))
        self.canvas = tk.Canvas(canvas_frame, background="#070A12", cursor="cross", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=3, pady=3)
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self._update_description()
        root.after(120, self.refresh)

    def _target(self) -> dict[str, Any]:
        name = self.target_var.get()
        return next(target for target in self.config["workflow"] if target["name"] == name)

    def _update_description(self) -> None:
        target = self._target()
        calibrated = "已标定" if target.get("roi") else "未标定"
        self.description_var.set(f"●  {target['description']}   ·   {calibrated}")

    def refresh(self) -> None:
        try:
            self.image = self.device.screenshot()
        except Exception as exc:
            messagebox.showerror("截图失败", str(exc))
            return
        self.root.update_idletasks()
        max_width = max(760, self.canvas.winfo_width())
        max_height = max(480, self.canvas.winfo_height())
        self.scale = min(max_width / self.image.width, max_height / self.image.height, 1.0)
        display_size = (
            max(1, round(self.image.width * self.scale)),
            max(1, round(self.image.height * self.scale)),
        )
        self.display_image = self.image.resize(display_size, Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(self.display_image)
        self.canvas.delete("all")
        self.canvas.config(scrollregion=(0, 0, *display_size))
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.selection = None
        self.rectangle = None

    def _press(self, event: tk.Event[Any]) -> None:
        self.start = (event.x, event.y)
        if self.rectangle is not None:
            self.canvas.delete(self.rectangle)
        self.rectangle = self.canvas.create_rectangle(
            event.x,
            event.y,
            event.x,
            event.y,
            outline=self.green,
            width=3,
        )

    def _drag(self, event: tk.Event[Any]) -> None:
        if self.start and self.rectangle is not None:
            self.canvas.coords(self.rectangle, self.start[0], self.start[1], event.x, event.y)

    def _release(self, event: tk.Event[Any]) -> None:
        if not self.start:
            return
        x1, x2 = sorted((self.start[0], event.x))
        y1, y2 = sorted((self.start[1], event.y))
        self.selection = (x1, y1, x2, y2)
        self.start = None

    def save_selection(self) -> None:
        if self.image is None or self.selection is None:
            messagebox.showwarning("没有框选", "请先刷新截图并拖出一个矩形区域。")
            return
        x1, y1, x2, y2 = self.selection
        if x2 - x1 < 12 or y2 - y1 < 12:
            messagebox.showwarning("区域太小", "请框选至少 12×12 像素的稳定区域。")
            return

        original = (
            max(0, round(x1 / self.scale)),
            max(0, round(y1 / self.scale)),
            min(self.image.width, round(x2 / self.scale)),
            min(self.image.height, round(y2 / self.scale)),
        )
        target = self._target()
        template_path = resolve_project_path(self.config_path, target["template"])
        template_path.parent.mkdir(parents=True, exist_ok=True)
        self.image.crop(original).save(template_path, format="PNG")

        ox1, oy1, ox2, oy2 = original
        target["roi"] = [
            round(ox1 / self.image.width, 6),
            round(oy1 / self.image.height, 6),
            round(ox2 / self.image.width, 6),
            round(oy2 / self.image.height, 6),
        ]
        if target["kind"] == "tap":
            target["click"] = [
                round(((ox1 + ox2) / 2) / self.image.width, 6),
                round(((oy1 + oy2) / 2) / self.image.height, 6),
            ]
        else:
            target["click"] = None
        save_config(self.config, self.config_path)
        self._update_description()
        messagebox.showinfo("已保存", f"已标定 {target['name']}\n{template_path}")


def run_calibration(device: MumuDevice, config: dict[str, Any], config_path: Path) -> None:
    root = tk.Tk()
    CalibrationWindow(root, device, config, config_path)
    root.mainloop()
