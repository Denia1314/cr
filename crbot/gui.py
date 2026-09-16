from __future__ import annotations

import contextlib
import json
import os
import queue
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk
import tkinter as tk
from typing import Any, Callable

from PIL import Image, ImageTk

from . import IMPLEMENTED_STAGES, implemented_stages_label, release_label
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
    DECISION_ENGINE_LABELS,
    apply_decision_engine,
    load_decision_engine,
    save_decision_engine,
    DEFAULT_RUNTIME_MODEL,
    RUNTIME_MODEL_LABELS,
    apply_runtime_model,
    load_runtime_model,
    runtime_model_label,
    save_runtime_model,
)
from .vision import WorkflowRecognizer
from .training_sync import ReplaySync, SyncWorker, training_allowed


APP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = APP_ROOT / "config.json"


class Palette:
    BG = "#0B1120"
    SURFACE = "#101B2E"
    CARD = "#142137"
    CARD_ALT = "#1D304B"
    BORDER = "#2B405C"
    TEXT = "#F4F7FF"
    MUTED = "#A6B7CE"
    FAINT = "#8195B0"
    BLUE = "#397CF6"
    BLUE_HOVER = "#6098FF"
    PURPLE = "#A78BFA"
    CYAN = "#44D7E8"
    GREEN = "#43D6A0"
    AMBER = "#FFBE55"
    RED = "#FF647C"
    LOG = "#0D1728"


FONT = "Microsoft YaHei UI"
MONO = "Cascadia Mono"


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


class HoverButton(tk.Button):
    def __init__(
        self,
        master: tk.Misc,
        *,
        background: str,
        hover: str,
        foreground: str = Palette.TEXT,
        **kwargs: Any,
    ):
        self.normal_background = background
        self.hover_background = hover
        kwargs.setdefault("font", (FONT, 10, "bold"))
        kwargs.setdefault("padx", 16)
        kwargs.setdefault("pady", 11)
        super().__init__(
            master,
            background=background,
            activebackground=hover,
            foreground=foreground,
            disabledforeground=Palette.FAINT,
            activeforeground=foreground,
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            cursor="hand2",
            **kwargs,
        )
        self.bind("<Enter>", self._enter, add="+")
        self.bind("<Leave>", self._leave, add="+")

    def _enter(self, _event: tk.Event[Any]) -> None:
        if str(self["state"]) != "disabled":
            self.configure(background=self.hover_background)

    def _leave(self, _event: tk.Event[Any]) -> None:
        self.configure(background=self.normal_background)


class RoyalTrainerApp:
    def __init__(self, root: tk.Tk, config_path: Path = DEFAULT_CONFIG):
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
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll_messages)
        self.root.after(700, self._passive_status_check)
        self._refresh_model_status()
        self.sync_worker = SyncWorker(
            self.config_path.parent,
            notify=lambda message: self.messages.put(("log", message)),
        ).start()

    def _configure_window(self) -> None:
        self.root.title(release_label())
        self.root.configure(background=Palette.BG)
        width, height = 1260, 860
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        width = min(width, max(1060, screen_w - 100))
        height = min(height, max(720, screen_h - 100))
        x = max(0, (screen_w - width) // 2)
        y = max(0, (screen_h - height) // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.minsize(1040, 700)
        self.root.option_add("*TCombobox*Listbox.font", (FONT, 10))
        self.root.option_add("*TCombobox*Listbox.background", Palette.SURFACE)
        self.root.option_add("*TCombobox*Listbox.foreground", Palette.TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", Palette.BLUE)
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Vertical.TScrollbar", background=Palette.CARD_ALT,
                        troughcolor=Palette.LOG, borderwidth=0, arrowsize=12,
                        bordercolor=Palette.LOG, lightcolor=Palette.CARD_ALT,
                        darkcolor=Palette.CARD_ALT,
                        arrowcolor=Palette.MUTED)
        style.map("Vertical.TScrollbar", background=[("active", Palette.BORDER),
                                                     ("disabled", Palette.LOG)])
        style.configure(
            "Model.TCombobox",
            fieldbackground=Palette.SURFACE,
            background=Palette.CARD_ALT,
            foreground=Palette.TEXT,
            arrowcolor=Palette.MUTED,
            bordercolor=Palette.BORDER,
            lightcolor=Palette.BORDER,
            darkcolor=Palette.BORDER,
            padding=7,
        )
        style.map(
            "Model.TCombobox",
            fieldbackground=[("readonly", Palette.SURFACE), ("disabled", Palette.CARD)],
            foreground=[("readonly", Palette.TEXT), ("disabled", Palette.FAINT)],
            selectbackground=[("readonly", Palette.SURFACE)],
            selectforeground=[("readonly", Palette.TEXT)],
        )

    def _build_interface(self) -> None:
        header = tk.Frame(self.root, background=Palette.SURFACE)
        header.pack(fill="x")
        brand = tk.Frame(header, background=Palette.SURFACE)
        brand.pack(side="left", padx=24, pady=12)
        self.release_heading = tk.Label(
            brand, text=release_label(), font=(FONT, 12, "bold"),
            foreground=Palette.TEXT, background=Palette.SURFACE,
        )
        self.release_heading.pack(anchor="w")
        tk.Label(
            brand, text='已接入：'+' · '.join(stage for stage,_ in IMPLEMENTED_STAGES), font=(FONT, 8),
            foreground=Palette.MUTED, background=Palette.SURFACE,
        ).pack(anchor="w", pady=(3, 0))

        tools_button = tk.Menubutton(
            header, text="工具 ▾", font=(FONT, 10),
            background=Palette.CARD_ALT, foreground=Palette.TEXT,
            activebackground=Palette.BORDER, activeforeground=Palette.TEXT,
            relief="flat", borderwidth=0, padx=14, pady=9, cursor="hand2",
        )
        tools_button.pack(side="right", padx=(8, 24))
        self.tools_menu = tk.Menu(
            tools_button, tearoff=False, font=(FONT, 10),
            background=Palette.SURFACE, foreground=Palette.TEXT,
            activebackground=Palette.CARD_ALT, activeforeground=Palette.TEXT,
        )
        for label, command in (
            ("画面标定", self.open_calibration),
            ("训练数据", self.open_runs_folder),
            ("精确标注（可选）", self.open_annotation),
            ("无标注自学", self.check_learning_status),
            ("自主学习状态", self.show_self_learning_status),
            ("示范学习", self.start_demonstration),
            ("双机数据同步", self.sync_training_data),
        ):
            self.tools_menu.add_command(label=label, command=command)
        tools_button.configure(menu=self.tools_menu)

        self.check_button = HoverButton(
            header, text="连接并检测", background=Palette.CARD_ALT,
            hover=Palette.BORDER, command=self.run_doctor,
        )
        self.check_button.pack(side="right")
        self.connection_button = HoverButton(
            header, text="自定义 ADB", background=Palette.SURFACE,
            hover=Palette.CARD_ALT, foreground=Palette.MUTED,
            command=self.open_adb_connection_settings, padx=12,
        )
        self.connection_button.pack(side="right", padx=8)
        self._refresh_connection_button()
        self.header_badge = tk.Label(
            header, text="尚未连接", font=(FONT, 9),
            foreground=Palette.MUTED, background=Palette.SURFACE,
        )
        self.header_badge.pack(side="left")

        content = tk.Frame(self.root, background=Palette.BG)
        content.pack(fill="both", expand=True, padx=24, pady=18)
        content.grid_columnconfigure(0, weight=3, uniform="workspace")
        content.grid_columnconfigure(1, weight=2, uniform="workspace")
        content.grid_rowconfigure(2, weight=1)

        stats = tk.Frame(content, background=Palette.BG)
        stats.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        for column in range(3):
            stats.grid_columnconfigure(column, weight=1, uniform="stats")

        self.device_value, self.device_note = self._stat_card(
            stats, 0, "模拟器", "未检测", "等待连接设备", Palette.BLUE
        )
        self.safety_value, self.safety_note = self._stat_card(
            stats, 1, "安全门", "待确认", "等待识别离线标志", Palette.PURPLE
        )
        self.battle_value, self.battle_note = self._stat_card(
            stats, 2, "已完成", "0 局", "本次运行统计", Palette.GREEN
        )

        self._build_control_card(content)
        self._build_preview_card(content)
        self._build_log_card(content)

    def _stat_card(
        self,
        parent: tk.Frame,
        column: int,
        title: str,
        value: str,
        note: str,
        accent: str,
    ) -> tuple[tk.Label, tk.Label]:
        card = tk.Frame(
            parent,
            background=Palette.CARD,
            highlightbackground=Palette.BORDER,
            highlightthickness=1,
        )
        card.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 7, 0 if column == 2 else 7))
        accent_bar = tk.Frame(card, background=accent, width=4, height=62)
        accent_bar.pack(side="left", fill="y")
        info = tk.Frame(card, background=Palette.CARD)
        info.pack(fill="both", expand=True, padx=15, pady=11)
        tk.Label(
            info,
            text=title,
            font=(FONT, 8, "bold"),
            foreground=Palette.MUTED,
            background=Palette.CARD,
        ).pack(anchor="w")
        value_label = tk.Label(
            info,
            text=value,
            font=(FONT, 19, "bold"),
            foreground=Palette.TEXT,
            background=Palette.CARD,
        )
        value_label.pack(anchor="w", pady=(2, 0))
        note_label = tk.Label(
            info,
            text=note,
            font=(FONT, 8),
            foreground=Palette.FAINT,
            background=Palette.CARD,
        )
        note_label.pack(anchor="w")
        return value_label, note_label

    def _build_preview_card(self, parent: tk.Frame) -> None:
        card = tk.Frame(
            parent,
            background=Palette.CARD,
            highlightbackground=Palette.BORDER,
            highlightthickness=1,
        )
        card.grid(row=2, column=0, sticky="nsew", padx=(0, 8))
        card.grid_rowconfigure(1, weight=1)
        card.grid_columnconfigure(0, weight=1)

        top = tk.Frame(card, background=Palette.CARD)
        top.grid(row=0, column=0, sticky="ew", padx=16, pady=(13, 9))
        tk.Label(
            top,
            text="实时画面",
            font=(FONT, 11, "bold"),
            foreground=Palette.TEXT,
            background=Palette.CARD,
        ).pack(side="left")
        self.preview_state = tk.Label(
            top,
            text="●  等待画面",
            font=(FONT, 8),
            foreground=Palette.FAINT,
            background=Palette.CARD,
        )
        self.preview_state.pack(side="left", padx=12)
        HoverButton(top,text='方格战场',command=self.show_grid_world,
                    background=Palette.CARD_ALT,foreground=Palette.MUTED,
                    hover='#26314B',padx=8,pady=6).pack(side='right',padx=4)
        HoverButton(
            top,
            text="刷新",
            background=Palette.CARD_ALT,
            hover="#26314B",
            foreground=Palette.MUTED,
            command=self.refresh_preview,
            padx=11,
            pady=6,
        ).pack(side="right")

        preview_frame = tk.Frame(card, background="#070A12", highlightbackground="#20293D", highlightthickness=1)
        preview_frame.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))
        preview_frame.grid_rowconfigure(0, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)
        self.preview_canvas = tk.Canvas(
            preview_frame,
            width=300,
            height=180,
            background="#070A12",
            highlightthickness=0,
        )
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        self.preview_canvas.bind("<Configure>", lambda _event: self._render_preview())
        self._draw_preview_placeholder()

    def _build_control_card(self, parent: tk.Frame) -> None:
        controls = tk.Frame(parent, background=Palette.BG)
        controls.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        strategy_bar = tk.Frame(controls, background=Palette.BG)
        strategy_bar.pack(fill="x", pady=(0, 6))
        tk.Label(strategy_bar, text="版本", font=(FONT, 9), foreground=Palette.MUTED,
                 background=Palette.BG).pack(side="left", padx=(0, 8))
        self.engine_var = tk.StringVar(value=DECISION_ENGINE_LABELS[self.decision_engine])
        self.engine_combo = ttk.Combobox(strategy_bar, textvariable=self.engine_var,
                                       values=list(DECISION_ENGINE_LABELS.values()), state="readonly",
                                       width=28, style="Model.TCombobox", font=(FONT, 9))
        self.engine_combo.pack(side="left")
        self.engine_combo.bind("<<ComboboxSelected>>", self._on_engine_selected)
        tk.Label(strategy_bar, text="停止后切换；预测与实测结果分开记录", font=(FONT, 8),
                 foreground=Palette.MUTED, background=Palette.BG).pack(side="left", padx=10)
        bar = tk.Frame(controls, background=Palette.BG)
        bar.pack(fill="x")
        tk.Label(
            bar, text="旧规则", font=(FONT, 9),
            foreground=Palette.MUTED, background=Palette.BG,
        ).pack(side="left", padx=(0, 8))
        self.model_key_by_label = {
            label: key for key, label in RUNTIME_MODEL_LABELS.items()
        }
        self.model_var = tk.StringVar(value=runtime_model_label(self.runtime_model))
        self.model_combo = ttk.Combobox(
            bar, textvariable=self.model_var,
            values=list(RUNTIME_MODEL_LABELS.values()), state="readonly",
            width=24, style="Model.TCombobox", font=(FONT, 9),
        )
        self.model_combo.pack(side="left")
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_selected)
        self.run_state_label = tk.Label(
            bar, text="当前状态：待机", font=(FONT, 8),
            foreground=Palette.MUTED, background=Palette.BG,
        )
        self.run_state_label.pack(side="left", padx=12)
        self.stop_button = HoverButton(
            bar, text="■  安全停止", background="#352033", hover="#47263B",
            foreground=Palette.RED, command=self.stop_bot, state="disabled",
        )
        self.stop_button.pack(side="right")
        self.start_button = HoverButton(
            bar, text="▶  离线训练", background=Palette.BLUE,
            hover=Palette.BLUE_HOVER, command=self.start_bot,
        )
        self.start_button.pack(side="right", padx=(0, 8))
        self.model_status_button = HoverButton(
            controls, text="正在读取模型…", font=(FONT, 8),
            background=Palette.BG, hover=Palette.SURFACE,
            foreground=Palette.MUTED, anchor="w", justify="left",
            padx=0, pady=4, command=self.show_model_versions,
        )
        self.model_status_button.pack(fill="x", pady=(5, 0))
        self.model_status_button.bind(
            "<Configure>",
            lambda event: self.model_status_button.configure(wraplength=max(100, event.width - 10)),
        )

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

    def _build_log_card(self, parent: tk.Frame) -> None:
        card = tk.Frame(
            parent,
            background=Palette.CARD,
            highlightbackground=Palette.BORDER,
            highlightthickness=1,
        )
        card.grid(row=2, column=1, sticky="nsew", padx=(8, 0))
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(1, weight=1)

        top = tk.Frame(card, background=Palette.CARD)
        top.grid(row=0, column=0, sticky="ew", padx=16, pady=(11, 7))
        tk.Label(
            top,
            text="运行日志",
            font=(FONT, 10, "bold"),
            foreground=Palette.TEXT,
            background=Palette.CARD,
        ).pack(side="left")
        self.log_counter = tk.Label(
            top,
            text="0 条事件",
            font=(FONT, 8),
            foreground=Palette.FAINT,
            background=Palette.CARD,
        )
        self.log_counter.pack(side="right")

        log_wrap = tk.Frame(card, background=Palette.LOG)
        log_wrap.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 14))
        log_wrap.grid_columnconfigure(0, weight=1)
        log_wrap.grid_rowconfigure(0, weight=1)
        self.log_text = tk.Text(
            log_wrap,
            height=5,
            width=30,
            spacing1=3,
            spacing3=3,
            background=Palette.LOG,
            foreground="#C7D1E6",
            insertbackground=Palette.TEXT,
            selectbackground="#2D4778",
            relief="flat",
            borderwidth=0,
            font=(MONO, 9),
            padx=10,
            pady=8,
            wrap="word",
            state="disabled",
        )
        scroll = ttk.Scrollbar(log_wrap, orient="vertical", command=self.log_text.yview,
                               style="Vertical.TScrollbar")
        self.log_text.configure(yscrollcommand=scroll.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.tag_configure("time", foreground=Palette.FAINT)
        self.log_text.tag_configure("success", foreground=Palette.GREEN)
        self.log_text.tag_configure("action", foreground=Palette.CYAN)
        self.log_text.tag_configure("warning", foreground=Palette.AMBER)
        self.log_text.tag_configure("error", foreground=Palette.RED)
        self.log_text.tag_configure("normal", foreground="#C7D1E6")
        self.log_count = 0
        self._append_log("控制台已就绪。可开始离线人机训练。", "normal")

    def _bind_shortcuts(self) -> None:
        self.root.bind("<F5>", lambda _event: self.refresh_preview())
        self.root.bind("<Control-Return>", lambda _event: self.start_bot())
        self.root.bind("<Escape>", lambda _event: self.stop_bot())

    def _set_header(self, text: str, color: str) -> None:
        self.header_badge.configure(text=f"●  {text}", foreground=color)

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
            self.safety_value.configure(text="验证通过", foreground=Palette.GREEN)
            self.safety_note.configure(text="已确认离线人机页面")
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
        self.start_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")
        self.model_combo.configure(state="disabled" if running else "readonly")
        self.engine_combo.configure(state="disabled" if running else "readonly")
        self.check_button.configure(state="disabled" if running else "normal")
        self.connection_button.configure(state="disabled" if running else "normal")
        if running:
            self._set_header("任务运行中", Palette.GREEN)
        elif self.device_value.cget("text") == "已连接":
            self._set_header("设备已就绪", Palette.GREEN)

    def _on_model_selected(self, _event: tk.Event[Any] | None = None) -> None:
        label = self.model_var.get()
        key = self.model_key_by_label.get(label, DEFAULT_RUNTIME_MODEL)
        try:
            self.runtime_model = save_runtime_model(self.config_path, key)
        except (OSError, ValueError) as exc:
            self.model_var.set(runtime_model_label(self.runtime_model))
            messagebox.showerror("无法保存模型选择", str(exc), parent=self.root)
            return
        self._append_log(f"运行版本已切换为：{runtime_model_label(key)}", "success")

    def _on_engine_selected(self, _event=None):
        selected = next((key for key, label in DECISION_ENGINE_LABELS.items()
                         if label == self.engine_var.get()), "legacy")
        try:
            self.decision_engine = save_decision_engine(self.config_path, selected)
        except (OSError, ValueError) as exc:
            self.engine_var.set(DECISION_ENGINE_LABELS[self.decision_engine])
            messagebox.showerror("无法保存决策模式", str(exc), parent=self.root)
            return
        self._append_log(f"决策模式：{DECISION_ENGINE_LABELS[selected]}，下次启动生效", "success")

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
            hover="#25304A",
            foreground=Palette.MUTED,
            command=restore_auto_connection,
            padx=14,
            pady=8,
        ).pack(side="left")

        dialog.bind("<Return>", lambda _event: save_custom_connection())
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        port_entry.focus_set()
        port_entry.selection_range(0, "end")

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
        selected_label = self.model_var.get()
        selected_model = self.model_key_by_label.get(
            selected_label, DEFAULT_RUNTIME_MODEL
        )
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
        calibrated = set(payload.get("calibrated") or [])
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

        if "offline_ai_marker" in calibrated:
            self.safety_value.configure(text="已标定", foreground=Palette.GREEN)
            self.safety_note.configure(text="运行时连续 3 帧校验")
        else:
            self.safety_value.configure(text="需标定", foreground=Palette.AMBER)
            self.safety_note.configure(text="工具 → 画面标定")

        image = payload.get("image")
        if isinstance(image, Image.Image):
            self._show_image(image)
        if announce:
            installed_text = "游戏已安装" if payload.get("installed") else "未检测到游戏"
            missing = payload.get("missing") or []
            calibration_text = "标定完整" if not missing else "缺少标定：" + ", ".join(missing)
            self._append_log(f"环境检测完成：{installed_text} · {calibration_text}", "success" if not missing and payload.get("installed") else "warning")

    def show_grid_world(self) -> None:
        from .grid_world_view import GridWorldWindow
        existing=getattr(self,'grid_inspector',None)
        if existing is not None and existing.window.winfo_exists():
            existing.window.lift();return
        self.grid_inspector=GridWorldWindow(self.root,lambda: getattr(
            getattr(self.engine,'policy',None),'grid_frame',None) if self.engine else None)

    def _show_image(self, image: Image.Image) -> None:
        self.current_image = image.copy()
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
        try:
            while True:
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

        self.root.after(33, self._poll_messages)

    def _finish_bot(self, error: str | None) -> None:
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
