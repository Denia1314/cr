from __future__ import annotations

import contextlib
import json
import os
import queue
import sys
import threading
import time
import math
import traceback
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk
import tkinter as tk
from typing import Any, Callable

from PIL import Image, ImageTk

from . import implemented_stages_label, release_label
from .app_paths import data_root
from .animations import Animator, fade_in, mix
from .ui_theme import FONT, MONO, Palette
from .ui_widgets import HoverButton, Surface
from .ui_select import AnimatedSelect
from .adb import MumuDevice
from .annotate import AnnotationWindow
from .calibrate import CalibrationWindow
from .cards import CardCatalog
from .config import load_config, resolve_project_path, save_config
from .demonstration import DemonstrationRecorder
from .engine import BotEngine
from .learning import ModelRegistry, audit_learning_data
from .model_status import model_status
from .imitation import (
    ImitationRegistry,
    audit_demonstrations,
    train_imitation_policy,
)
from .replay import audit_replay
from .replay_learning import (
    ReplayPolicyRegistry,
    audit_replay_learning,
    train_replay_policy,
)
from .runtime_model import (
    apply_decision_engine,
    load_decision_engine,
    DEFAULT_RUNTIME_MODEL,
    apply_runtime_model,
    load_runtime_model,
    runtime_model_label,
)
from .vision import WorkflowRecognizer
from .training_sync import ReplaySync, SyncWorker, training_allowed


APP_ROOT = data_root()
DEFAULT_CONFIG = APP_ROOT / "config.json"


class QueueWriter:
    """Line-buffered stream that forwards worker output to Tk's main thread."""

    def __init__(self, messages: queue.Queue[tuple[str, Any]]):
        self.messages = messages
        self.buffer = ""

    def write(self, value: str) -> int:
        self.buffer += value.replace("\r\n", "\n").replace("\r", "\n")
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line.strip():
                self.messages.put(("log", line))
        return len(value)

    def flush(self) -> None:
        if self.buffer.strip():
            self.messages.put(("log", self.buffer.strip()))
        self.buffer = ""


class RoyalTrainerApp:
    def __init__(self, root: tk.Tk, config_path: Path = DEFAULT_CONFIG, *, start_services: bool = True):
        self.root = root
        self.config_path = config_path.resolve()
        self.messages: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.stop_event = threading.Event()
        self.bot_thread: threading.Thread | None = None
        self.engine: BotEngine | DemonstrationRecorder | None = None
        self.run_mode = "automation"
        self.background_busy = False
        self.current_image: Image.Image | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.last_frame_identity: int | None = None
        self.closing = False
        self.model_details_window: tk.Toplevel | None = None
        self.model_details_text: tk.Text | None = None
        self.model_details = ""
        self.card_library_window = None
        try:
            initial_config, _ = load_config(self.config_path)
            self.runtime_model = load_runtime_model(self.config_path, initial_config)
            self.decision_engine = load_decision_engine(self.config_path, initial_config)
        except (OSError, ValueError):
            self.runtime_model = DEFAULT_RUNTIME_MODEL
            self.decision_engine = "legacy"

        self._configure_window()
        self._build_interface()
        self._bind_shortcuts()
        self._compact = None
        self.root.bind("<Configure>", self._adapt_density, add="+")
        self.root.after_idle(self._adapt_density)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll_messages)
        if start_services:
            self.root.after(700, self._passive_status_check)
        self._refresh_model_status()
        self.sync_worker = None
        if start_services:
            self.sync_worker = SyncWorker(
                self.config_path.parent,
                notify=lambda message: self.messages.put(("log", message)),
            ).start()
        self.window_animation = fade_in(root)

    def _configure_window(self) -> None:
        self.root.title(release_label())
        self.root.configure(background=Palette.BG)
        width, height = 1280, 860
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        width = min(width, max(1060, screen_w - 100))
        height = min(height, max(720, screen_h - 100))
        x = max(0, (screen_w - width) // 2)
        y = max(0, (screen_h - height) // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.minsize(1040, 700)
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Vertical.TScrollbar", background=Palette.CARD_ALT,
                        troughcolor=Palette.LOG, borderwidth=0, arrowsize=12,
                        bordercolor=Palette.LOG, lightcolor=Palette.CARD_ALT,
                        darkcolor=Palette.CARD_ALT,
                        arrowcolor=Palette.MUTED)
        style.map("Vertical.TScrollbar", background=[("active", Palette.BORDER),
                                                     ("disabled", Palette.LOG)])

    def _build_interface(self) -> None:
        shell = tk.Frame(self.root, background=Palette.BG)
        shell.pack(fill="both", expand=True)
        workspace = tk.Frame(shell, background=Palette.BG)
        workspace.pack(side="left", fill="both", expand=True, padx=22, pady=18)
        header = tk.Frame(workspace, background=Palette.BG)
        header.pack(fill="x", pady=(0, 16))
        self.check_button = HoverButton(header, text="连接并检测", background=Palette.CARD,
            hover=Palette.SELECTION, foreground=Palette.BLUE, command=self.run_doctor)
        self.check_button.pack(side="right", padx=(12, 0))
        self.connection_button = HoverButton(header, text="自定义 ADB",
            background=Palette.CARD, hover=Palette.CARD_ALT, foreground=Palette.MUTED,
            command=self.open_adb_connection_settings, padx=14)
        self.connection_button.pack(side="right", padx=(12, 0))
        self.card_library_button = HoverButton(header, text="卡牌库",
            background=Palette.CARD, hover=Palette.SELECTION, foreground=Palette.MUTED,
            command=self.open_card_library, font=(FONT, 9), padx=12)
        self.card_library_button.pack(side="right", padx=(12, 0))
        self._refresh_connection_button()
        status = tk.Frame(header, background=Palette.BG)
        status.pack(side="right")
        self.activity_dot = tk.Canvas(status, width=12, height=12, bg=Palette.BG, highlightthickness=0)
        self.activity_dot.pack(side="left", padx=4)
        self.activity_dot.create_oval(3, 3, 9, 9, fill=Palette.FAINT, outline="", tags="dot")
        self.activity_animation = Animator(self.activity_dot)
        self.header_badge = tk.Label(status, text="尚未连接", font=(FONT, 9),
                                    foreground=Palette.MUTED, background=Palette.BG)
        self.header_badge.pack(side="left")
        brand = tk.Frame(header, background=Palette.BG)
        brand.pack(side="left")
        self.workspace_title = tk.Label(brand, text="训练工作台", font=(FONT, 20, "bold"),
                 foreground=Palette.TEXT, background=Palette.BG)
        self.workspace_title.pack(anchor="w")
        self.release_heading = tk.Label(brand, text=release_label(), font=(FONT, 8),
                                       foreground=Palette.MUTED, background=Palette.BG)
        self.release_heading.pack(anchor="w", pady=(3, 0))
        content = tk.Frame(workspace, background=Palette.BG)
        self.content = content
        content.pack(fill="both", expand=True)
        content.grid_columnconfigure(0, weight=3, uniform="workspace")
        content.grid_columnconfigure(1, weight=2, uniform="workspace")
        content.grid_rowconfigure(2, weight=1)
        stats = self.stats_card = tk.Frame(content, background=Palette.BG)
        stats.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        for column in range(2):
            stats.grid_columnconfigure(column, weight=1, uniform="stats")
        self.device_value, self.device_note = self._stat_card(stats, 0, "设备连接", "未检测", "等待连接模拟器", Palette.BLUE)
        self.battle_value, self.battle_note = self._stat_card(stats, 1, "完成对局", "0 局", "本次运行统计", Palette.GREEN)
        self._build_control_card(content)
        self._build_preview_card(content)
        self._build_log_card(content)
        footer = tk.Frame(workspace, background=Palette.BG)
        footer.pack(side="bottom", fill="x", pady=(10, 0))
        content.pack_forget()
        content.pack(fill="both", expand=True)
        tk.Label(footer, text="Ctrl + Enter 开始    ·    Esc 停止    ·    F5 刷新", font=(FONT, 8),
                 foreground=Palette.FAINT, background=Palette.BG).pack(side="left")
        tk.Label(footer, text="仅离线人机", font=(FONT, 8), foreground=Palette.FAINT,
                 background=Palette.BG).pack(side="right")

    def _adapt_density(self, event=None) -> None:
        if event is not None and event.widget is not self.root:
            return
        compact = self.root.winfo_height() <= 760
        if compact == self._compact:
            return
        self._compact = compact
        self.workspace_title.configure(font=(FONT, 16 if compact else 20, "bold"))
        self.release_heading.configure(font=(FONT, 7 if compact else 8))
        for value in (self.device_value, self.battle_value):
            value.configure(font=(FONT, 16 if compact else 20, "bold"))
        for card in self.stats_card.winfo_children():
            card.body.pack_configure(pady=8 if compact else 14)
        for note in (self.device_note, self.battle_note):
            note.configure(font=(FONT, 7 if compact else 8))
        for card in (self.control_card, self.preview_card, self.log_card):
            card.body.pack_configure(pady=10 if compact else 14)
        for combo in (self.preview_mode_combo,):
            combo.configure(font=(FONT, 8 if compact else 9), pady=5 if compact else 7)
        for button in (self.start_button, self.stop_button):
            button.configure(pady=6 if compact else 8, font=(FONT, 9 if compact else 10))
        self.model_status_button.configure(font=(FONT, 7 if compact else 8), pady=2 if compact else 4)

    def _stat_card(self, parent, column, title, value, note, accent):
        card = Surface(parent, padding=14)
        card.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 5, 0 if column == 1 else 5))
        tk.Label(card.body, text=title, font=(FONT, 9), fg=Palette.MUTED,
                 bg=Palette.CARD).pack(anchor="w")
        value_label = tk.Label(card.body, text=value, font=(FONT, 20, "bold"),
                              fg=accent, bg=Palette.CARD, anchor="w", width=1)
        value_label.pack(fill="x", pady=(4, 2))
        note_label = tk.Label(card.body, text=note, font=(FONT, 8), fg=Palette.FAINT,
                             bg=Palette.CARD, anchor="w", width=1)
        note_label.pack(fill="x")
        note_label.bind("<Configure>", lambda event: note_label.configure(wraplength=max(100,event.width)))
        return value_label, note_label

    def _build_preview_card(self, parent: tk.Frame) -> None:
        self.preview_card = Surface(parent)
        self.preview_expanded = False
        self.preview_card.grid(row=2, column=0, sticky="nsew", padx=(0, 6))
        card = self.preview_card.body
        card.grid_rowconfigure(2, weight=1)
        card.grid_columnconfigure(0, weight=1)
        top = tk.Frame(card, background=Palette.CARD)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        tk.Label(top, text="实时画面", font=(FONT, 12, "bold"), fg=Palette.TEXT,
                 bg=Palette.CARD).pack(side="left")
        self.preview_state = tk.Label(top, text="等待画面", font=(FONT, 8),
                                      fg=Palette.FAINT, bg=Palette.CARD)
        self.preview_state.pack(side="right")
        options = tk.Frame(card, background=Palette.CARD)
        options.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.preview_mode = tk.StringVar(value="同帧对照")
        self.preview_mode_combo = AnimatedSelect(options, textvariable=self.preview_mode,
            values=("游戏画面", "YOLO识别", "同帧对照", "推演快照"), state="readonly", width=10, font=(FONT, 8))
        self.preview_mode_combo.pack(side="left")
        self.preview_mode_combo.bind("<<ComboboxSelected>>", lambda _event: self._set_preview_mode())
        self.expand_preview_button = HoverButton(options, text="放大", command=self._toggle_preview_size,
            background=Palette.CARD_ALT, foreground=Palette.MUTED, hover=Palette.SELECTION, padx=10, pady=5)
        self.expand_preview_button.pack(side="right", padx=(6, 0))
        HoverButton(options, text="刷新", background=Palette.CARD_ALT, hover=Palette.SELECTION,
                    foreground=Palette.MUTED, command=self.refresh_preview, padx=10, pady=5).pack(side="right")
        preview_frame = tk.Frame(card, background=Palette.LOG)
        preview_frame.grid(row=2, column=0, sticky="nsew")
        preview_frame.grid_rowconfigure(0, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)
        self.preview_canvas = tk.Canvas(preview_frame, width=240, height=140,
                                       background=Palette.LOG, highlightthickness=0)
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        self.preview_canvas.bind("<Configure>", lambda _event: self._render_preview())
        self._draw_preview_placeholder()
        from .grid_world_view import GridWorldWindow
        self.grid_inspector = GridWorldWindow(preview_frame, self._grid_preview_source, embedded=True)
        self._set_preview_mode()

    def _build_control_card(self, parent: tk.Frame) -> None:
        self.control_card = Surface(parent, padding=14)
        self.control_card.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        controls = self.control_card.body
        bar = tk.Frame(controls, background=Palette.CARD)
        bar.pack(fill="x")
        self.run_state_label = tk.Label(bar, text="当前状态：待机", font=(FONT, 9),
            foreground=Palette.MUTED, background=Palette.CARD)
        self.run_state_label.pack(side="left")
        self.stop_button = HoverButton(bar, text="■  安全停止", background=Palette.DANGER_BG,
            hover=Palette.DANGER_HOVER, foreground=Palette.RED, command=self.stop_bot, state="disabled", pady=8)
        self.stop_button.pack(side="right")
        self.start_button = HoverButton(bar, text="▶  开始训练", background=Palette.BLUE,
            hover=Palette.BLUE_HOVER, foreground="#FFFFFF", command=self.start_bot, pady=8)
        self.start_button.pack(side="right", padx=(0, 8))
        self.model_status_button = HoverButton(controls, text="正在读取模型…", font=(FONT, 8),
            background=Palette.CARD, hover=Palette.SURFACE, foreground=Palette.MUTED,
            anchor="w", justify="left", padx=4, pady=4, command=self.show_model_versions)
        self.model_status_button.pack(fill="x", pady=(6, 0))
        self.model_status_button.bind("<Configure>",
            lambda event: self.model_status_button.configure(wraplength=max(100, event.width - 14)), add="+")

    def _build_log_card(self, parent: tk.Frame) -> None:
        self.log_card = Surface(parent)
        self.log_card.grid(row=2, column=1, sticky="nsew", padx=(6, 0))
        card = self.log_card.body
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(1, weight=1)
        top = tk.Frame(card, background=Palette.CARD)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        tk.Label(top, text="运行动态", font=(FONT, 12, "bold"), fg=Palette.TEXT,
                 bg=Palette.CARD).pack(side="left")
        self.log_counter = tk.Label(top, text="0 条事件", font=(FONT, 8), fg=Palette.FAINT, bg=Palette.CARD)
        self.log_counter.pack(side="right")
        log_wrap = tk.Frame(card, background=Palette.LOG)
        log_wrap.grid(row=1, column=0, sticky="nsew")
        log_wrap.grid_columnconfigure(0, weight=1)
        log_wrap.grid_rowconfigure(0, weight=1)
        self.log_text = tk.Text(log_wrap, height=4, width=24, spacing1=5, spacing3=5,
            background=Palette.LOG, foreground=Palette.TEXT, insertbackground=Palette.TEXT,
            selectbackground=Palette.SELECTION, selectforeground=Palette.TEXT, relief="flat",
            borderwidth=0, font=(MONO, 9), padx=10, pady=8, wrap="word", state="disabled")
        scroll = ttk.Scrollbar(log_wrap, orient="vertical", command=self.log_text.yview, style="Vertical.TScrollbar")
        self.log_text.configure(yscrollcommand=scroll.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        for tag, color in (("time", Palette.FAINT), ("success", Palette.GREEN),
            ("action", Palette.CYAN), ("warning", Palette.AMBER), ("error", Palette.RED), ("normal", Palette.TEXT)):
            self.log_text.tag_configure(tag, foreground=color)
        self.log_count = 0
        self._append_log("工作台已就绪。连接模拟器后即可开始离线训练。", "normal")

    def _refresh_model_status(self) -> None:
        if self.closing:
            return
        summary, details = model_status(
            self.config_path.parent, policy=getattr(self.engine, "policy", None),
            running=self._bot_is_running(), demonstration=self.run_mode == "demonstration",
        )
        details = (
            f"{release_label()}\n{implemented_stages_label()}\n"
            "阶段表示功能已接入，实战收益仍需单独验收。\n\n" + details
        )
        self.model_status_button.configure(text=summary)
        if details != self.model_details:
            self.model_details = details
            if self.model_details_text is not None and self.model_details_text.winfo_exists():
                position = self.model_details_text.yview()[0]
                self.model_details_text.configure(state="normal")
                self.model_details_text.delete("1.0", "end")
                self.model_details_text.insert("1.0", details)
                self.model_details_text.configure(state="disabled")
                self.model_details_text.yview_moveto(position)
        self.root.after(3000, self._refresh_model_status)

    def show_model_versions(self) -> None:
        if self.model_details_window is not None and self.model_details_window.winfo_exists():
            self.model_details_window.lift()
            return
        window = tk.Toplevel(self.root)
        self.model_details_window = window
        window.animation = fade_in(window)
        window.title("模型版本 · 自动刷新")
        window.geometry("850x500")
        window.configure(background=Palette.BG)
        text = tk.Text(
            window, font=(FONT, 9), wrap="word", background=Palette.BG,
            foreground=Palette.TEXT, relief="flat", padx=14, pady=14,
        )
        self.model_details_text = text
        scroll = tk.Scrollbar(window, command=text.yview)
        scroll.pack(side="right", fill="y")
        text.configure(yscrollcommand=scroll.set)
        text.pack(fill="both", expand=True)
        text.insert("1.0", self.model_details)
        text.configure(state="disabled")

    def _bind_shortcuts(self) -> None:
        self.root.bind("<F5>", lambda _event: self.refresh_preview())
        self.root.bind("<Control-Return>", lambda _event: self.start_bot())
        self.root.bind("<Escape>", lambda _event: self.stop_bot())

    def _set_header(self, text: str, color: str) -> None:
        self.header_badge.configure(text=text, foreground=color)

    def _set_run_state(self, text: str, color: str = Palette.MUTED) -> None:
        self.run_state_label.configure(text=f"当前状态：{text}", foreground=color)

    def _append_log(self, line: str, tag: str | None = None) -> None:
        if self.closing:
            return
        if tag is None:
            lowered = line.lower()
            if "错误" in line or "失败" in line or "error" in lowered:
                tag = "error"
            elif "[安全门]" in line or "已连接" in line or "通过" in line:
                tag = "success"
            elif any(
                marker in line
                for marker in (
                    "[战斗]",
                    "[开局]",
                    "[匹配]",
                    "[下牌]",
                    "[导航]",
                    "[宝箱]",
                    "[确定]",
                )
            ):
                tag = "action"
            elif any(marker in line for marker in ("[结算]", "警告", "未安装", "缺少")):
                tag = "warning"
            else:
                tag = "normal"
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{stamp}  ", "time")
        self.log_text.insert("end", line + "\n", tag)
        self.log_count += 1
        if int(self.log_text.index("end-1c").split(".")[0]) > 900:
            self.log_text.delete("1.0", "120.0")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        self.log_counter.configure(text=f"{self.log_count} 条事件")
        self._interpret_log(line)

    def _interpret_log(self, line: str) -> None:
        if "[安全门]" in line:
            self._set_run_state("离线入口已验证", Palette.GREEN)
        elif "[战斗]" in line:
            self._set_run_state("战斗中", Palette.CYAN)
            self.preview_state.configure(text="●  实时画面", foreground=Palette.GREEN)
        elif "[开局]" in line:
            self._set_run_state("正在进入战斗", Palette.BLUE)
        elif "[匹配]" in line:
            self._set_run_state("匹配时间较长，继续等待", Palette.BLUE)
        elif "[结算]" in line:
            self._set_run_state("正在处理结算", Palette.AMBER)
        elif "[宝箱]" in line:
            self._set_run_state("正在处理奖励宝箱", Palette.AMBER)
        elif "[确定]" in line:
            self._set_run_state("正在点击确定", Palette.AMBER)
        elif "[停止]" in line or "已达到 max_battles" in line:
            self._set_run_state("已停止", Palette.MUTED)

    def _set_running_controls(self, running: bool) -> None:
        if running:
            self.activity_animation.pulse("running", lambda t: self.activity_dot.itemconfigure(
                "dot", fill=mix(Palette.BLUE, Palette.CYAN, t)))
        else:
            self.activity_animation.cancel("running")
            self.activity_dot.itemconfigure("dot", fill=Palette.MUTED)
        self.start_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")
        self.check_button.configure(state="disabled" if running else "normal")
        self.connection_button.configure(state="disabled" if running else "normal")
        if running:
            self._set_header("任务运行中", Palette.GREEN)
        elif self.device_value.cget("text") == "已连接":
            self._set_header("设备已就绪", Palette.GREEN)

    def _refresh_connection_button(self, config: dict[str, Any] | None = None) -> None:
        try:
            if config is None:
                config, _config_path = load_config(self.config_path)
            mumu = dict(config.get("mumu", {}))
            if str(mumu.get("connection_mode", "auto")) == "custom_adb":
                port = int(mumu.get("adb_port", 16384))
                self.connection_button.configure(
                    text=f"ADB :{port}", foreground=Palette.CYAN
                )
            else:
                self.connection_button.configure(
                    text="自定义 ADB", foreground=Palette.MUTED
                )
        except (OSError, ValueError, TypeError):
            self.connection_button.configure(
                text="自定义 ADB", foreground=Palette.MUTED
            )

    def open_card_library(self) -> None:
        if self.card_library_window and self.card_library_window.window.winfo_exists():
            self.card_library_window.window.lift()
            self.card_library_window.window.focus_force()
            return
        try:
            from .card_library import CardLibrary
            from .card_library_ui import CardLibraryWindow
            config, config_path = load_config(self.config_path)
            catalog = resolve_project_path(config_path,
                config.get("dataset", {}).get("card_catalog", "data/cards.json"))
            self.card_library_window = CardLibraryWindow(self.root,
                CardLibrary(config_path.parent, catalog))
        except (OSError, ValueError) as exc:
            messagebox.showerror("卡牌库", str(exc), parent=self.root)

    def open_adb_connection_settings(self) -> None:
        if self._bot_is_running():
            messagebox.showinfo(
                "训练进行中",
                "请先安全停止训练，再修改模拟器连接。",
                parent=self.root,
            )
            return
        if self.background_busy:
            messagebox.showinfo(
                "请稍候", "正在处理设备操作，请稍候。", parent=self.root
            )
            return
        try:
            config, config_path = load_config(self.config_path)
        except (OSError, ValueError) as exc:
            messagebox.showerror("无法读取配置", str(exc), parent=self.root)
            return

        mumu = config.setdefault("mumu", {})
        dialog = tk.Toplevel(self.root)
        dialog.title("自定义模拟器连接")
        dialog.configure(background=Palette.BG)
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        panel = tk.Frame(
            dialog,
            background=Palette.CARD,
            highlightbackground=Palette.BORDER,
            highlightthickness=1,
        )
        panel.pack(fill="both", expand=True, padx=18, pady=18)
        tk.Label(
            panel,
            text="按 ADB 端口连接",
            font=(FONT, 15, "bold"),
            foreground=Palette.TEXT,
            background=Palette.CARD,
        ).pack(anchor="w", padx=18, pady=(16, 4))
        tk.Label(
            panel,
            text="适用于 MuMu 多开或其他已启用 ADB 的本地模拟器。\n"
            "将直接连接 127.0.0.1:<端口>，不再依赖 MuMu CLI 发现。",
            justify="left",
            font=(FONT, 9),
            foreground=Palette.MUTED,
            background=Palette.CARD,
        ).pack(anchor="w", padx=18, pady=(0, 14))

        port_row = tk.Frame(panel, background=Palette.CARD_ALT)
        port_row.pack(fill="x", padx=18, pady=(0, 14))
        tk.Label(
            port_row,
            text="ADB 端口",
            font=(FONT, 10, "bold"),
            foreground=Palette.TEXT,
            background=Palette.CARD_ALT,
        ).pack(side="left", padx=12, pady=12)
        port_var = tk.StringVar(value=str(mumu.get("adb_port", 16384)))
        port_entry = tk.Entry(
            port_row,
            textvariable=port_var,
            width=10,
            justify="center",
            font=("Segoe UI", 11, "bold"),
            foreground=Palette.TEXT,
            background=Palette.SURFACE,
            insertbackground=Palette.TEXT,
            relief="flat",
            highlightbackground=Palette.BORDER,
            highlightcolor=Palette.BLUE,
            highlightthickness=1,
        )
        port_entry.pack(side="right", padx=12, pady=8)

        actions = tk.Frame(panel, background=Palette.CARD)
        actions.pack(fill="x", padx=18, pady=(0, 16))

        def save_custom_connection() -> None:
            try:
                port = int(port_var.get().strip())
                if not 1 <= port <= 65535:
                    raise ValueError
            except ValueError:
                messagebox.showwarning(
                    "端口无效",
                    "请输入 1 到 65535 之间的整数端口。",
                    parent=dialog,
                )
                port_entry.focus_set()
                port_entry.selection_range(0, "end")
                return
            mumu["connection_mode"] = "custom_adb"
            mumu["adb_host"] = "127.0.0.1"
            mumu["adb_port"] = port
            try:
                save_config(config, config_path)
            except OSError as exc:
                messagebox.showerror("保存失败", str(exc), parent=dialog)
                return
            self._refresh_connection_button(config)
            dialog.destroy()
            self._append_log(
                f"已切换到自定义 ADB：127.0.0.1:{port}", "success"
            )
            self.run_doctor()

        def restore_auto_connection() -> None:
            mumu["connection_mode"] = "auto"
            try:
                save_config(config, config_path)
            except OSError as exc:
                messagebox.showerror("保存失败", str(exc), parent=dialog)
                return
            self._refresh_connection_button(config)
            dialog.destroy()
            self._append_log("已恢复 MuMu 自动发现连接。", "success")
            self.run_doctor()

        HoverButton(
            actions,
            text="保存并连接",
            background=Palette.BLUE,
            hover=Palette.BLUE_HOVER,
            command=save_custom_connection,
            padx=14,
            pady=8,
        ).pack(side="right")
        HoverButton(
            actions,
            text="恢复自动发现",
            background=Palette.CARD_ALT,
            hover=Palette.SELECTION,
            foreground=Palette.MUTED,
            command=restore_auto_connection,
            padx=14,
            pady=8,
        ).pack(side="left")

        dialog.bind("<Return>", lambda _event: save_custom_connection())
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        port_entry.focus_set()
        port_entry.selection_range(0, "end")
        dialog.animation = fade_in(dialog)

    def _start_background(self, target: Callable[[], None], name: str) -> bool:
        if self.background_busy:
            messagebox.showinfo("请稍候", "正在处理上一项操作，请稍候。", parent=self.root)
            return False
        self.background_busy = True
        thread = threading.Thread(target=target, name=name, daemon=True)
        thread.start()
        return True

    def _passive_status_check(self) -> None:
        if self.background_busy or self._bot_is_running():
            return

        def worker() -> None:
            try:
                config, config_path = load_config(self.config_path)
                device = MumuDevice(config)
                if device.uses_custom_adb:
                    serial = device.connect()
                    info = device.info()
                else:
                    info = device.info()
                    serial = device.connect() if info.get("is_android_started") else None
                payload: dict[str, Any] = {
                    "started": bool(info.get("is_android_started")),
                    "connected": serial is not None,
                    "installed": None,
                    "serial": serial,
                    "screen": None,
                    "image": None,
                }
                recognizer = WorkflowRecognizer(config, config_path)
                payload["calibrated"] = recognizer.calibrated_names()
                if payload["connected"]:
                    payload["installed"] = device.package_installed(config["game"]["package"])
                    image = device.screenshot()
                    payload["image"] = image
                    payload["screen"] = f"{image.width}×{image.height}"
                self.messages.put(("device_status", payload))
            except Exception as exc:
                self.messages.put(("passive_error", str(exc)))
            finally:
                self.messages.put(("background_done", None))

        self._start_background(worker, "passive-device-check")

    def run_doctor(self) -> None:
        if self._bot_is_running():
            messagebox.showinfo("训练进行中", "请先安全停止训练，再执行环境检测。", parent=self.root)
            return
        self._append_log("正在连接模拟器并检查运行环境…", "action")
        self.check_button.configure(state="disabled", text="检测中…")

        def worker() -> None:
            try:
                config, config_path = load_config(self.config_path)
                device = MumuDevice(config)
                serial = device.connect()
                info = device.info()
                installed = device.package_installed(config["game"]["package"])
                image = device.screenshot()
                recognizer = WorkflowRecognizer(config, config_path)
                calibrated = recognizer.calibrated_names()
                required = {"offline_ai_marker"} if config.get("automation", {}).get("single_marker_mode") else {
                    "offline_ai_marker",
                    "start_battle",
                    "battle_marker",
                }
                payload = {
                    "started": bool(info.get("is_android_started")),
                    "connected": True,
                    "installed": installed,
                    "serial": serial,
                    "screen": f"{image.width}×{image.height}",
                    "image": image,
                    "calibrated": calibrated,
                    "missing": sorted(required - set(calibrated)),
                }
                self.messages.put(("doctor_result", payload))
            except Exception as exc:
                self.messages.put(("operation_error", ("环境检测失败", str(exc))))
            finally:
                self.messages.put(("background_done", None))

        if not self._start_background(worker, "environment-check"):
            self.check_button.configure(state="normal", text="连接并检测")

    def start_bot(self) -> None:
        if self._bot_is_running():
            return
        if self.background_busy:
            messagebox.showinfo("请稍候", "设备操作正在进行，请稍候再开始。", parent=self.root)
            return
        selected_model = self.runtime_model
        selected_engine = self.decision_engine
        self.stop_event = threading.Event()
        self.engine = None
        self.run_mode = "automation"
        self.battle_value.configure(text="0 局", foreground=Palette.TEXT)
        self._append_log(
            f"准备无限运行 · 版本：{runtime_model_label(selected_model)}",
            "action",
        )
        self._set_run_state("正在连接设备", Palette.BLUE)
        self._set_running_controls(True)

        def run() -> None:
            config, config_path = load_config(self.config_path)
            config = apply_runtime_model(config, selected_model)
            config = apply_decision_engine(config, selected_engine)
            device = MumuDevice(config)
            serial = device.connect()
            print(f"[设备] 已连接模拟器：{serial}")
            if self.stop_event.is_set():
                print("[停止] 启动已取消，未打开游戏。")
                return
            self.engine = BotEngine(
                device, config, config_path, dry_run=False,
                max_battles=0, stop_event=self.stop_event,
            )
            self.engine.run()

        self._start_run_worker(run, "royal-trainer")

    def start_demonstration(self) -> None:
        if self._bot_is_running():
            messagebox.showinfo(
                "已有任务运行中",
                "请先安全停止当前任务。",
                parent=self.root,
            )
            return
        if self.background_busy:
            messagebox.showinfo("请稍候", "设备操作正在进行，请稍候。", parent=self.root)
            return
        if not messagebox.askokcancel(
            "开始示范学习",
            "请先进入已标定的离线人机入口。\n\n"
            "启动后由你手动选牌和落牌；程序只监控画面与触摸，不会自动点击。"
            "菜单操作会被忽略，只有离线人机战斗中的出牌才会成为专家示范。",
            parent=self.root,
        ):
            return
        self.stop_event = threading.Event()
        self.engine = None
        self.run_mode = "demonstration"
        self.battle_value.configure(text="0 局", foreground=Palette.TEXT)
        self._append_log("准备启动手动示范学习；等待离线入口验证…", "action")
        self._set_run_state("正在连接触摸监控", Palette.BLUE)
        self._set_running_controls(True)

        def run() -> None:
            config, config_path = load_config(self.config_path)
            device = MumuDevice(config)
            serial = device.connect()
            print(f"[设备] 已连接模拟器：{serial}")
            if self.stop_event.is_set():
                print("[停止] 启动已取消。")
                return
            self.engine = DemonstrationRecorder(
                device, config, config_path, stop_event=self.stop_event,
            )
            self.engine.run()

        self._start_run_worker(run, "human-demonstration")

    def _start_run_worker(self, target: Callable[[], None], name: str) -> None:
        def worker() -> None:
            writer = QueueWriter(self.messages)
            error: str | None = None
            try:
                with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                    target()
            except Exception as exc:
                error = str(exc)
                writer.write(f"错误：{exc}\n")
                if os.environ.get("CRBOT_DEBUG") == "1":
                    writer.write(traceback.format_exc())
            finally:
                writer.flush()
                self.messages.put(("bot_done", error))

        self.bot_thread = threading.Thread(target=worker, name=name, daemon=True)
        self.bot_thread.start()

    def stop_bot(self) -> None:
        if not self._bot_is_running():
            return
        self._append_log("正在请求安全停止，请等待当前设备操作完成…", "warning")
        self._set_run_state("正在安全停止", Palette.AMBER)
        self.stop_button.configure(state="disabled", text="正在停止…")
        self.stop_event.set()
        if self.engine is not None:
            self.engine.request_stop()

    def refresh_preview(self) -> None:
        if self.engine is not None and self.engine.latest_frame is not None:
            self._show_image(self.engine.latest_frame)
            return
        if self.background_busy:
            return
        self.preview_state.configure(text="●  获取画面中", foreground=Palette.AMBER)

        def worker() -> None:
            try:
                config, _config_path = load_config(self.config_path)
                device = MumuDevice(config)
                device.connect()
                image = device.screenshot()
                self.messages.put(("preview", image))
            except Exception as exc:
                self.messages.put(("operation_error", ("刷新画面失败", str(exc))))
            finally:
                self.messages.put(("background_done", None))

        self._start_background(worker, "preview-refresh")

    def open_calibration(self) -> None:
        if self._bot_is_running():
            messagebox.showinfo("训练进行中", "请先安全停止训练，再打开画面标定。", parent=self.root)
            return
        self._append_log("正在连接设备并准备标定窗口…", "action")

        def worker() -> None:
            try:
                config, config_path = load_config(self.config_path)
                device = MumuDevice(config)
                device.connect()
                self.messages.put(("calibration_ready", (device, config, config_path)))
            except Exception as exc:
                self.messages.put(("operation_error", ("无法打开标定", str(exc))))
            finally:
                self.messages.put(("background_done", None))

        self._start_background(worker, "calibration-connect")

    def _show_calibration(self, payload: tuple[MumuDevice, dict[str, Any], Path]) -> None:
        device, config, config_path = payload
        window = tk.Toplevel(self.root)
        window.transient(self.root)
        CalibrationWindow(window, device, config, config_path)
        window.bind(
            "<Destroy>",
            lambda event: self.root.after(300, self._passive_status_check) if event.widget is window else None,
            add="+",
        )
        self._append_log("标定窗口已打开。", "success")

    def open_runs_folder(self) -> None:
        runs = self.config_path.parent / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(runs)  # type: ignore[attr-defined]
        except OSError as exc:
            messagebox.showerror("无法打开目录", str(exc), parent=self.root)

    def open_annotation(self) -> None:
        if self._bot_is_running():
            messagebox.showinfo(
                "训练进行中",
                "请先安全停止训练，再打开数据标注工具。",
                parent=self.root,
            )
            return
        window = tk.Toplevel(self.root)
        window.transient(self.root)
        AnnotationWindow(window, self.config_path.parent)
        self._append_log("可选的精确敌军标注窗口已打开。", "success")

    def show_self_learning_status(self) -> None:
        from .learning_store import LearningStore
        try:
            status = LearningStore(self.config_path.parent).current_status()
            labels = {"idle": "尚未开始", "auditing": "审计数据", "training": "训练候选",
                      "candidate_ready": "候选已保存", "waiting_data": "等待有效数据",
                      "cooldown": "等待训练间隔", "paused": "已暂停", "failed": "任务失败",
                      "budget_exhausted": "达到时间预算", "collector": "仅采集",
                      "deferred": "等待其他训练结束", "waiting_boundary": "等待局间边界"}
            text = labels.get(status.get("phase"), status.get("phase", "未知")) + "\n" + status.get("reason", "")
            for key, label in (("target", "学习目标"), ("new_battles", "新增有效局"),
                               ("new_actions", "新增有效动作"), ("candidate_version", "候选版本")):
                if key in status:
                    text += f"\n{label}：{status[key]}"
            if "quality_passed" in status:
                text += "\n离线检查：" + ("通过" if status["quality_passed"] else "未通过")
            from .learning_store import read_json
            ledger = read_json(self.config_path.parent / "training/self_learning/trials/ledger.json")
            trial = ledger.get("active") or (ledger.get("history") or [None])[-1]
            if trial:
                phases = {"battle_trial": "实战比较中", "battle_pass": "实战通过，待部署",
                          "rejected": "验收未通过", "inconclusive": "证据不足", "interrupted": "实测中断"}
                text += "\n自动实测：" + phases.get(trial["status"], trial["status"])
                text += f"\n已记录 {len(trial.get('rows', []))} 局；候选 {trial['candidate'].get('version', '未知')}"
            else:
                text += "\n自动实测：尚无试验记录"
            from .replay_learning import ReplayPolicyRegistry
            registry = ReplayPolicyRegistry(self.config_path.parent).load()
            deployment = registry.get("deployment") or {}
            stages = {"probation": "已加载，观察中", "stable": "稳定运行", "rollback_pending": "等待局间回退",
                      "rolled_back": "已回退"}
            text += "\n自动部署：" + stages.get(deployment.get("state"), "等待通过实战验收的候选")
            if deployment:
                if deployment.get("reason"):
                    text += "\n部署说明：" + str(deployment["reason"])
                text += f"\n部署代次：{deployment.get('generation')}；观察记录：{deployment.get('total_battles', 0)} 局"
                publication = registry.get("deployment_publication", {})
                text += "\n发布：" + ("远端提交已核验" if publication.get("generation") == deployment.get("generation")
                                     and publication.get("uploaded") else "尚未确认上传")
            if registry.get("remote_deployment_error"):
                text += "\n远端部署待处理：" + registry["remote_deployment_error"].get("reason", "未知原因")
            self._append_log("自主学习状态：" + json.dumps(status, ensure_ascii=False), "action")
            messagebox.showinfo("自主学习 SL4 · 自动部署回退", text, parent=self.root)
        except (OSError, ValueError) as exc:
            messagebox.showerror("自主学习状态读取失败", str(exc), parent=self.root)

    def check_learning_status(self) -> None:
        if self._bot_is_running():
            messagebox.showinfo(
                "训练进行中",
                "请先安全停止训练，再检查或训练模型。",
                parent=self.root,
            )
            return
        self._append_log("正在检查无需标注的示范数据与模型…", "action")

        def worker() -> None:
            try:
                config, config_path = load_config(self.config_path)
                from .self_learning import effective_learning_config
                config = effective_learning_config(config, config_path)
                catalog_value = config.get("dataset", {}).get(
                    "card_catalog", "data/cards.json"
                )
                catalog = CardCatalog.load(
                    resolve_project_path(config_path, str(catalog_value))
                )
                demonstration_config = dict(config.get("demonstration", {}))
                imitation_audit = audit_demonstrations(
                    config_path.parent,
                    catalog,
                    demonstration_config,
                )
                imitation_registry = ImitationRegistry(config_path.parent)
                imitation_state = imitation_registry.load()
                trained_counts = [
                    int(
                        dict(candidate.get("manifest", {})).get(
                            "human_train_actions", 0
                        )
                    )
                    for candidate in imitation_state.get("candidates", [])
                    if isinstance(candidate, dict)
                ]
                last_trained_actions = max(trained_counts, default=0)
                imitation_training = None
                if (
                    imitation_audit.ready
                    and imitation_audit.usable_actions > last_trained_actions
                ):
                    imitation_training = train_imitation_policy(
                        config_path.parent,
                        catalog,
                        demonstration_config,
                    )

                exact_registry = ModelRegistry(config_path.parent)
                exact_champion = exact_registry.champion()
                exact_champion_version = (
                    str(exact_champion.get("version", ""))
                    if exact_champion
                    else ""
                )
                exact_audit = audit_learning_data(
                    config_path.parent,
                    catalog,
                    dict(config.get("training", {})),
                    champion_version=exact_champion_version,
                )
                replay_config = dict(config.get("replay", {}))
                replay_learning_audit = audit_replay_learning(
                    config_path.parent,
                    catalog,
                    replay_config,
                )
                replay_registry = ReplayPolicyRegistry(config_path.parent)
                replay_state = replay_registry.load()
                replay_trained_counts = [
                    int(dict(candidate.get("manifest", {})).get("total_actions", 0))
                    for candidate in replay_state.get("candidates", [])
                    if isinstance(candidate, dict)
                ]
                last_replay_trained_actions = max(replay_trained_counts, default=0)
                replay_training = None
                if (
                    replay_learning_audit.ready
                    and training_allowed(config_path.parent)
                    and replay_learning_audit.actions > last_replay_trained_actions
                ):
                    replay_training = train_replay_policy(
                        config_path.parent,
                        catalog,
                        replay_config,
                        candidate_only=True,
                    )
                payload = {
                    "imitation": imitation_audit.to_dict(),
                    "imitation_training": imitation_training,
                    "replay": audit_replay(
                        config_path.parent,
                        replay_config,
                    ).to_dict(),
                    "replay_learning": replay_learning_audit.to_dict(),
                    "replay_training": replay_training,
                    "exact_detector": exact_audit.to_dict(),
                }
                self.messages.put(("learning_audit", payload))
            except Exception as exc:
                self.messages.put(("operation_error", ("学习数据审计失败", str(exc))))
            finally:
                self.messages.put(("background_done", None))

        self._start_background(worker, "learning-audit")

    def sync_training_data(self) -> None:
        if self.background_busy:
            return
        def worker() -> None:
            try:
                service = ReplaySync(self.config_path.parent)
                if not service.config.get("enabled"):
                    raise ValueError("请先双击 setup_sync.bat 登录并配置本机")
                result = service.sync()
                role = "训练机" if service.config.get("trainer") else "采集机"
                self.messages.put(("log", f"[共享] {role}：上传 {result['exported_episodes']} 局，接收 {result['imported_episodes']} 局，接收模型 {result['received_models']} 个"))
            except Exception as exc:
                self.messages.put(("operation_error", ("数据同步未完成", str(exc))))
            finally:
                self.messages.put(("background_done", None))
        self._start_background(worker, "manual-sync")

    def _show_learning_audit(self, payload: dict[str, Any]) -> None:
        imitation = dict(payload.get("imitation", {}))
        imitation_champion = ImitationRegistry(self.config_path.parent).champion()
        training_result = payload.get("imitation_training")
        maturity_names = {"assisted": "辅助级", "full": "完整级"}
        if imitation_champion:
            maturity = str(imitation_champion.get("maturity", "full"))
            status = f"已启用（{maturity_names.get(maturity, maturity)}）"
            imitation_model_status = str(imitation_champion.get("version"))
        elif imitation.get("ready"):
            status = "数据已训练，但候选模型尚未通过安全验证"
            imitation_model_status = "尚无已验证模型"
        else:
            status = "正在积累示范操作"
            imitation_model_status = "尚无已验证模型"

        detail = (
            f"无需人工标注的自我学习：{status}\n\n"
            f"有效出牌：{imitation.get('usable_actions', 0)}\n"
            f"覆盖：{imitation.get('battles', 0)} 局 / "
            f"{imitation.get('distinct_selected_cards', 0)} 种卡牌\n"
            f"自动复原：{imitation.get('automatically_reconstructed_actions', 0)} 次\n"
            f"当前模型：{imitation_model_status}"
        )
        imitation_reasons = list(imitation.get("blocking_reasons", []))
        if imitation_reasons:
            detail += "\n\n继续进行‘示范学习’即可补足：\n- " + "\n- ".join(
                str(value) for value in imitation_reasons
            )
        if isinstance(training_result, dict):
            candidate = dict(training_result.get("candidate", {}))
            if candidate.get("promoted"):
                detail += "\n\n本次新增数据已自动训练并通过验证。"
            else:
                rejection = list(candidate.get("rejection_reasons", []))
                detail += "\n\n本次候选模型未通过验证，已保留原策略。"
                if rejection:
                    detail += "\n- " + "\n- ".join(str(value) for value in rejection)

        replay = dict(payload.get("replay", {}))
        replay_learning = dict(payload.get("replay_learning", {}))
        replay_training = payload.get("replay_training")
        replay_champion = ReplayPolicyRegistry(self.config_path.parent).champion()
        collection_status = (
            "数据量达到试验门槛"
            if replay.get("collection_ready")
            else "正在安全积累"
        )
        detail += (
            "\n\n自动战斗经验回放（无需标注）："
            f"{collection_status}\n"
            f"可信结算：{replay.get('verified_episodes', 0)} 局 "
            f"（胜 {replay.get('wins', 0)} / 负 {replay.get('losses', 0)} / "
            f"平 {replay.get('draws', 0)}）\n"
            f"可信状态—动作经验：{replay.get('verified_transitions', 0)} 条\n"
            f"出牌确认：{replay.get('confirmed_transitions', 0)} 已确认 / "
            f"{replay.get('unconfirmed_transitions', 0)} 未确认\n"
            f"同版本可训练：{replay_learning.get('verified_episodes', 0)} 局 / "
            f"{replay_learning.get('actions', 0)} 个动作\n"
            f"原始回放：{replay_learning.get('raw_episodes', 0)} 局 / "
            f"{replay_learning.get('raw_transitions', 0)} 条；去重后可用动作："
            f"{replay_learning.get('deduplicated_actions', 0)} 条\n"
        )
        if replay_champion:
            detail += f"回放冠军模型：{replay_champion.get('version')}（实际加载见主界面）"
        else:
            detail += "当前为影子模式；未通过验证的经验不会改写策略。"
        replay_reasons = list(replay.get("blocking_reasons", []))
        if replay_reasons:
            detail += "\n尚缺：\n- " + "\n- ".join(
                str(value) for value in replay_reasons
            )
        learning_reasons = list(replay_learning.get("blocking_reasons", []))
        if learning_reasons:
            detail += "\n同版本训练尚缺：\n- " + "\n- ".join(
                str(value) for value in learning_reasons
            )
        if isinstance(replay_training, dict):
            candidate = dict(replay_training.get("candidate", {}))
            if candidate.get("promoted"):
                detail += "\n\n本次回放候选已通过影子验证并低权重晋升。"
            elif candidate.get("quality_passed"):
                detail += "\n\n本次回放候选通过质量验证，但安全开关保持影子模式。"
            else:
                detail += "\n\n本次回放候选未通过验证，实战策略没有变化。"
            candidate_reasons = list(candidate.get("rejection_reasons", []))
            if candidate_reasons:
                detail += "\n- " + "\n- ".join(
                    str(value) for value in candidate_reasons
                )

        exact = dict(payload.get("exact_detector", {}))
        exact_champion = ModelRegistry(self.config_path.parent).champion()
        from .battlefield_assets import assets_ready, VERSION as battlefield_version
        baseline_ready = assets_ready(self.config_path.parent)
        exact_status = (
            f"识别冠军：{exact_champion.get('version')}（实际加载见主界面）"
            if exact_champion
            else f"公开预训练：{battlefield_version}，待本地准确率验收（实际加载见主界面）" if baseline_ready
            else "模型未安装：运行 setup_battlefield.bat；当前只能观察位置和威胁"
        )
        detail += (
            "\n\n可选增强：精确识别敌方卡名\n"
            f"状态：{exact_status}\n"
            f"人工框选：{exact.get('human_frames', 0)} 帧 / "
            f"{exact.get('human_boxes', 0)} 个目标\n"
            "公开预训练模型可直接推理；本地训练和准确率验收仍需独立单位框标注。"
        )
        self._append_log(
            f"无标注自学：{status} · 有效出牌 "
            f"{imitation.get('usable_actions', 0)} · "
            f"卡牌 {imitation.get('distinct_selected_cards', 0)}",
            "success" if imitation_champion else "warning",
        )
        messagebox.showinfo("无标注自我学习", detail, parent=self.root)

    def _apply_device_status(self, payload: dict[str, Any], announce: bool = False) -> None:
        if payload.get("connected"):
            self.device_value.configure(text="已连接", foreground=Palette.GREEN)
            serial = payload.get("serial") or "ADB 在线"
            screen = payload.get("screen")
            self.device_note.configure(text=f"{serial} · {screen}" if screen else str(serial))
            if payload.get("installed") is False:
                self.device_value.configure(text="缺少游戏", foreground=Palette.AMBER)
                self.device_note.configure(text="模拟器在线，但未检测到游戏包")
                self._set_header("等待安装游戏", Palette.AMBER)
            else:
                self._set_header("设备已就绪", Palette.GREEN)
        elif payload.get("started"):
            self.device_value.configure(text="连接失败", foreground=Palette.RED)
            self.device_note.configure(text="MuMu 已启动，ADB 尚未连接")
            self._set_header("ADB 未连接", Palette.RED)
        else:
            self.device_value.configure(text="未启动", foreground=Palette.MUTED)
            self.device_note.configure(text="点击“连接并检测”可自动启动")
            self._set_header("MuMu 未启动", Palette.MUTED)

        image = payload.get("image")
        if isinstance(image, Image.Image):
            self._show_image(image)
        if announce:
            installed_text = "游戏已安装" if payload.get("installed") else "未检测到游戏"
            missing = payload.get("missing") or []
            calibration_text = "标定完整" if not missing else "缺少标定：" + ", ".join(missing)
            self._append_log(f"环境检测完成：{installed_text} · {calibration_text}", "success" if not missing and payload.get("installed") else "warning")

    def show_grid_world(self) -> None:
        self.preview_mode.set('同帧对照')
        self._set_preview_mode()

    def _grid_preview_source(self):
        if self.preview_mode.get()!='推演快照':
            stream = getattr(self.engine, 'yolo_stream', None) if self.engine else None
            if stream is not None:
                return stream.latest() or (self.current_image, None)
            paired=getattr(getattr(self.engine,'policy',None),'yolo_frame',None) if self.engine else None
            return paired if paired is not None else (self.current_image,None)
        stream=getattr(self.engine,'live_grid_stream',None) if self.engine else None
        if stream is not None and self.preview_mode.get()!='推演快照':
            return stream.latest() or (self.current_image,None)
        paired=getattr(getattr(self.engine,'policy',None),'grid_frame',None) if self.engine else None
        # Model views always use the screenshot that produced the packet.
        return paired if paired is not None else (self.current_image,None)

    def _set_preview_mode(self):
        self.preview_card.reveal()
        if self.preview_mode.get()=='游戏画面':
            self.grid_inspector.window.grid_remove()
            self.preview_canvas.grid()
            self._render_preview()
        else:
            self.preview_canvas.grid_remove()
            self.grid_inspector.mode=('yolo' if self.preview_mode.get()=='YOLO识别' else
                                      'compare' if self.preview_mode.get()=='推演快照' else 'yolo_compare')
            self.grid_inspector.window.grid(row=0,column=0,sticky='nsew')
            self.grid_inspector.refresh(force=True)

    def _toggle_preview_size(self):
        self.preview_card.reveal()
        self.preview_expanded=not self.preview_expanded
        if self.preview_expanded:
            self.stats_card.grid_remove();self.log_card.grid_remove()
            self.preview_card.grid_configure(columnspan=2,padx=0)
            self.expand_preview_button.configure(text='还原')
        else:
            self.stats_card.grid();self.log_card.grid()
            self.preview_card.grid_configure(columnspan=1,padx=(0,6))
            self.expand_preview_button.configure(text='放大')

    def _show_image(self, image: Image.Image) -> None:
        # Capture frames are immutable snapshots. Keep their reference instead
        # of copying full-resolution pixels even when this view is hidden.
        self.current_image = image
        if getattr(self,'_preview_dimensions',None)!=image.size:
            self._preview_dimensions=image.size
            self.preview_state.configure(
                text=f"●  {image.width} × {image.height}",
                foreground=Palette.GREEN,
            )
        self._render_preview()

    def _draw_preview_placeholder(self) -> None:
        self.preview_canvas.delete("all")
        width = self.preview_canvas.winfo_width()
        height = self.preview_canvas.winfo_height()
        cx, cy = width / 2, height / 2
        self.preview_canvas.create_text(
            cx,
            cy,
            text="等待设备连接\n\n点击右上角「连接并检测」\n连接 MuMu 后在这里查看实时画面",
            justify="center",
            fill=Palette.FAINT,
            font=(FONT, 9),
        )

    def _render_preview(self) -> None:
        if hasattr(self,'grid_inspector') and self.preview_mode.get()!='游戏画面':
            return
        if self.current_image is None:
            self._draw_preview_placeholder()
            return
        canvas_w = max(100, self.preview_canvas.winfo_width())
        canvas_h = max(100, self.preview_canvas.winfo_height())
        scale = min(canvas_w / self.current_image.width, canvas_h / self.current_image.height)
        size = (
            max(1, round(self.current_image.width * scale)),
            max(1, round(self.current_image.height * scale)),
        )
        display = self.current_image.resize(size, Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(display)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.preview_photo, anchor="center")

    def _bot_is_running(self) -> bool:
        return self.bot_thread is not None and self.bot_thread.is_alive()

    def _poll_messages(self) -> None:
        if self.closing:
            return
        started=time.perf_counter()
        try:
            # Logs and status share Tk with the embedded 30 FPS renderer.
            # Leave excess messages queued for the next tick, without dropping.
            for _ in range(32):
                if time.perf_counter()-started >= .004:
                    break
                kind, payload = self.messages.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "device_status":
                    self._apply_device_status(payload)
                elif kind == "doctor_result":
                    self._apply_device_status(payload, announce=True)
                elif kind == "preview":
                    self._show_image(payload)
                elif kind == "calibration_ready":
                    self._show_calibration(payload)
                elif kind == "learning_audit":
                    self._show_learning_audit(payload)
                elif kind == "background_done":
                    self.background_busy = False
                    self.check_button.configure(state="normal", text="连接并检测")
                elif kind == "passive_error":
                    self.device_value.configure(text="不可用", foreground=Palette.RED)
                    self.device_note.configure(text=str(payload))
                    self._set_header("环境未就绪", Palette.RED)
                elif kind == "operation_error":
                    title, detail = payload
                    self._append_log(f"{title}：{detail}", "error")
                    messagebox.showerror(title, detail, parent=self.root)
                elif kind == "bot_done":
                    self._finish_bot(payload)
        except queue.Empty:
            pass

        if self.engine is not None and self._bot_is_running():
            self.battle_value.configure(text=f"{self.engine.completed_battles} 局")
            learning = getattr(self.engine, "self_learning", None)
            if learning is not None and learning.phase in {"auditing", "training"}:
                self.battle_note.configure(text="局间自主学习 · " + ("审计" if learning.phase == "auditing" else "候选训练"))
            elif self.engine.in_battle:
                self.battle_note.configure(text="当前正在战斗")
            elif self.engine.offline_verified:
                self.battle_note.configure(text="已验证离线入口")
            else:
                self.battle_note.configure(text="等待识别")
            frame = self.engine.latest_frame
            stream = getattr(self.engine, 'frame_stream', None)
            captured = stream.latest() if stream is not None else None
            if captured is not None:
                frame = captured.image
            if frame is not None and id(frame) != self.last_frame_identity:
                self.last_frame_identity = id(frame)
                self._show_image(frame)

        delay = max(1, math.ceil((1/30-(time.perf_counter()-started))*1000)) if self._bot_is_running() else 100
        self.root.after(delay, self._poll_messages)

    def _finish_bot(self, error: str | None) -> None:
        if self.sync_worker is not None:
            self.sync_worker.wake.set()
        battles = self.engine.completed_battles if self.engine is not None else 0
        self.battle_value.configure(text=f"{battles} 局")
        self.battle_note.configure(text="本次运行已结束")
        self._set_running_controls(False)
        self.stop_button.configure(text="■  安全停止")
        if error:
            self._set_run_state("发生错误", Palette.RED)
            self._set_header("训练已停止", Palette.RED)
        else:
            self._set_run_state("已停止", Palette.MUTED)
            if self.device_value.cget("text") == "已连接":
                self._set_header("设备已就绪", Palette.GREEN)
            if self.run_mode == "demonstration" and self.engine is not None:
                actions = int(getattr(self.engine, "action_count", 0))
                self._append_log(
                    f"示范学习结束：{battles} 局，记录 {actions} 次有效出牌。",
                    "success",
                )
            else:
                self._append_log(f"本次训练结束，共完成 {battles} 局。", "success")

    def _on_close(self) -> None:
        if self._bot_is_running():
            if not messagebox.askyesno(
                "训练仍在运行",
                "关闭窗口会停止本次训练。确认关闭吗？",
                parent=self.root,
            ):
                return
            self.stop_event.set()
            if self.engine is not None:
                self.engine.request_stop()
        self.closing = True
        if self.sync_worker is not None:
            self.sync_worker.close()
        self.root.destroy()


def enable_windows_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


def main(on_ready: Callable[[], None] | None = None) -> None:
    enable_windows_dpi_awareness()
    root = tk.Tk()
    RoyalTrainerApp(root)
    if on_ready is not None:
        root.after_idle(on_ready)
    root.mainloop()


if __name__ == "__main__":
    main()
