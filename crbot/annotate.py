from __future__ import annotations

from pathlib import Path
from tkinter import messagebox, ttk
import tkinter as tk
from typing import Any

from PIL import Image, ImageTk

from .battle_perception import UniversalHandRecognizer
from .cards import CardCatalog
from .dataset import DatasetStore, SampleRecord, list_dataset_runs


FONT = "Microsoft YaHei UI"


class AnnotationWindow:
    """Small local tool for human-authored perception labels."""

    def __init__(
        self,
        root: tk.Tk | tk.Toplevel,
        project_root: Path,
        catalog_path: Path | None = None,
    ):
        self.root = root
        self.project_root = project_root.resolve()
        self.catalog_path = (
            catalog_path.resolve()
            if catalog_path is not None
            else self.project_root / "data" / "cards.json"
        )
        self.catalog = self._load_catalog()
        self.run_paths = list_dataset_runs(self.project_root)
        self.store: DatasetStore | None = None
        self.samples: list[SampleRecord] = []
        self.annotations: dict[str, dict[str, Any]] = {}
        self.sample_index = 0
        self.image: Image.Image | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.render_box = (0.0, 0.0, 1.0, 1.0)
        self.objects: list[dict[str, Any]] = []
        self.draw_start: tuple[float, float] | None = None
        self.preview_rectangle: int | None = None
        self.dirty = False
        self.hand_recognizer: UniversalHandRecognizer | None = None

        self._configure_window()
        self._build_interface()
        if self.run_paths:
            self.run_combo.current(0)
            self._load_run(0)
        else:
            self.status_var.set("还没有可标注的训练数据；先运行一局以采集连续画面。")

    def _load_catalog(self) -> CardCatalog:
        try:
            return CardCatalog.load(self.catalog_path)
        except (OSError, ValueError):
            return CardCatalog([])

    def _configure_window(self) -> None:
        self.root.title("Royal Lab · 对局数据标注")
        self.root.geometry("1280x820")
        self.root.minsize(1020, 680)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _build_interface(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)
        outer.grid_columnconfigure(0, weight=4)
        outer.grid_columnconfigure(1, weight=2)
        outer.grid_rowconfigure(1, weight=1)

        toolbar = ttk.Frame(outer)
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        ttk.Label(toolbar, text="数据批次", font=(FONT, 10, "bold")).pack(side="left")
        self.run_combo = ttk.Combobox(
            toolbar,
            state="readonly",
            width=28,
            values=[path.name for path in self.run_paths],
        )
        self.run_combo.pack(side="left", padx=(8, 18))
        self.run_combo.bind("<<ComboboxSelected>>", self._run_selected)
        ttk.Label(
            toolbar,
            text="左键拖框；这里只保存人工标签，不会把机器人动作当正确答案。",
        ).pack(side="left")

        canvas_frame = ttk.Frame(outer)
        canvas_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        canvas_frame.grid_columnconfigure(0, weight=1)
        canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            canvas_frame,
            background="#090c17",
            highlightthickness=1,
            highlightbackground="#273149",
            cursor="crosshair",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self._render())
        self.canvas.bind("<ButtonPress-1>", self._draw_begin)
        self.canvas.bind("<B1-Motion>", self._draw_move)
        self.canvas.bind("<ButtonRelease-1>", self._draw_end)

        panel = ttk.Frame(outer, padding=(8, 0, 0, 0))
        panel.grid(row=1, column=1, sticky="nsew")
        panel.grid_columnconfigure(0, weight=1)

        ttk.Label(panel, text="四张手牌", font=(FONT, 11, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )
        ttk.Button(
            panel,
            text="识别手牌建议",
            command=self._suggest_hand,
        ).grid(row=0, column=0, sticky="e", pady=(0, 6))
        ids = self.catalog.ids()
        self.hand_vars: list[tk.StringVar] = []
        hand = ttk.Frame(panel)
        hand.grid(row=1, column=0, sticky="ew")
        for index in range(4):
            hand.grid_columnconfigure(index, weight=1)
            variable = tk.StringVar()
            variable.trace_add("write", self._mark_dirty)
            self.hand_vars.append(variable)
            box = ttk.Combobox(hand, textvariable=variable, values=ids, width=14)
            box.grid(row=0, column=index, sticky="ew", padx=(0 if index == 0 else 3, 0))
            ttk.Label(hand, text=f"卡槽 {index + 1}").grid(
                row=1, column=index, sticky="w", padx=(0 if index == 0 else 3, 0)
            )

        phase_row = ttk.Frame(panel)
        phase_row.grid(row=2, column=0, sticky="ew", pady=(13, 8))
        ttk.Label(phase_row, text="局面阶段").pack(side="left")
        self.phase_var = tk.StringVar(value="unknown")
        self.phase_var.trace_add("write", self._mark_dirty)
        ttk.Combobox(
            phase_row,
            textvariable=self.phase_var,
            state="readonly",
            width=14,
            values=["unknown", "defense", "neutral", "push"],
        ).pack(side="right")

        separator = ttk.Separator(panel)
        separator.grid(row=3, column=0, sticky="ew", pady=7)
        ttk.Label(panel, text="场上单位框", font=(FONT, 11, "bold")).grid(
            row=4, column=0, sticky="w", pady=(2, 6)
        )

        target_row = ttk.Frame(panel)
        target_row.grid(row=5, column=0, sticky="ew")
        target_row.grid_columnconfigure(0, weight=1)
        self.target_card_var = tk.StringVar()
        self.target_card_combo = ttk.Combobox(
            target_row,
            textvariable=self.target_card_var,
            values=ids,
        )
        self.target_card_combo.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.team_var = tk.StringVar(value="enemy")
        ttk.Combobox(
            target_row,
            textvariable=self.team_var,
            state="readonly",
            values=["enemy", "ally"],
            width=8,
        ).grid(row=0, column=1)

        self.object_list = tk.Listbox(panel, height=9, font=("Cascadia Mono", 9))
        self.object_list.grid(row=6, column=0, sticky="nsew", pady=(7, 5))
        panel.grid_rowconfigure(6, weight=1)
        object_buttons = ttk.Frame(panel)
        object_buttons.grid(row=7, column=0, sticky="ew")
        ttk.Button(object_buttons, text="删除选中框", command=self._remove_object).pack(
            side="left"
        )
        ttk.Button(object_buttons, text="清空框", command=self._clear_objects).pack(
            side="left", padx=6
        )

        options = ttk.Frame(panel)
        options.grid(row=8, column=0, sticky="ew", pady=(12, 5))
        self.usable_var = tk.BooleanVar(value=True)
        self.usable_var.trace_add("write", self._mark_dirty)
        ttk.Checkbutton(
            options,
            text="该画面可用于训练",
            variable=self.usable_var,
        ).pack(side="left")

        ttk.Label(panel, text="备注").grid(row=9, column=0, sticky="w")
        self.notes = tk.Text(panel, height=3, wrap="word", font=(FONT, 9))
        self.notes.grid(row=10, column=0, sticky="ew", pady=(3, 8))
        self.notes.bind("<<Modified>>", self._notes_modified)

        self.save_button = ttk.Button(panel, text="保存人工标注", command=self._save)
        self.save_button.grid(row=11, column=0, sticky="ew", pady=(2, 0))
        ttk.Button(
            panel,
            text="保存并跳到下一张未标注",
            command=self._save_and_next_unannotated,
        ).grid(row=12, column=0, sticky="ew", pady=(6, 0))

        footer = ttk.Frame(outer)
        footer.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.previous_button = ttk.Button(footer, text="← 上一帧", command=self._previous)
        self.previous_button.pack(side="left")
        self.next_button = ttk.Button(footer, text="下一帧 →", command=self._next)
        self.next_button.pack(side="left", padx=6)
        ttk.Button(
            footer,
            text="下一张未标注",
            command=self._next_unannotated,
        ).pack(side="left")
        self.position_var = tk.StringVar(value="0 / 0")
        ttk.Label(footer, textvariable=self.position_var).pack(side="left", padx=10)
        self.status_var = tk.StringVar(value="准备就绪")
        ttk.Label(footer, textvariable=self.status_var).pack(side="right")

    def _load_run(self, index: int) -> None:
        if not 0 <= index < len(self.run_paths):
            return
        self.store = DatasetStore(self.run_paths[index])
        self.samples = self.store.samples()
        self.annotations = self.store.latest_annotations()
        self.sample_index = 0
        if self.samples:
            self._load_sample(0)
        else:
            self.image = None
            self.position_var.set("0 / 0")
            self.status_var.set("这个批次没有可标注的战斗帧。")
            self._render()

    def _run_selected(self, _event: tk.Event[Any]) -> None:
        if not self._confirm_discard():
            return
        self._load_run(self.run_combo.current())

    def _load_sample(self, index: int) -> None:
        if not self.store or not 0 <= index < len(self.samples):
            return
        self.sample_index = index
        sample = self.samples[index]
        with Image.open(self.store.frame_path(sample)) as image:
            self.image = image.convert("RGB").copy()
        annotation = self.annotations.get(sample.sample_id, {})
        hand_cards = list(annotation.get("hand_cards", [None, None, None, None]))
        hand_cards = (hand_cards + [None] * 4)[:4]
        for variable, value in zip(self.hand_vars, hand_cards):
            variable.set(value or "")
        self.phase_var.set(str(annotation.get("battle_phase", "unknown")))
        self.usable_var.set(bool(annotation.get("usable", True)))
        self.objects = [dict(value) for value in annotation.get("objects", [])]
        self.notes.delete("1.0", "end")
        self.notes.insert("1.0", str(annotation.get("notes", "")))
        self.notes.edit_modified(False)
        self.dirty = False
        self._refresh_object_list()
        self._render()
        summary = self.store.summary()
        self.position_var.set(f"{index + 1} / {len(self.samples)}")
        self.status_var.set(
            f"已标注 {summary['annotated']} / {summary['samples']}，可用 {summary['usable']}"
        )
        self.previous_button.configure(state="normal" if index > 0 else "disabled")
        self.next_button.configure(
            state="normal" if index + 1 < len(self.samples) else "disabled"
        )

    def _render(self) -> None:
        self.canvas.delete("all")
        if self.image is None:
            width = max(100, self.canvas.winfo_width())
            height = max(100, self.canvas.winfo_height())
            self.canvas.create_text(
                width / 2,
                height / 2,
                text="没有可显示的战斗画面",
                fill="#8f9bb3",
                font=(FONT, 13),
            )
            return
        canvas_width = max(100, self.canvas.winfo_width())
        canvas_height = max(100, self.canvas.winfo_height())
        scale = min(canvas_width / self.image.width, canvas_height / self.image.height)
        shown_width = max(1, round(self.image.width * scale))
        shown_height = max(1, round(self.image.height * scale))
        offset_x = (canvas_width - shown_width) / 2
        offset_y = (canvas_height - shown_height) / 2
        shown = self.image.resize((shown_width, shown_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(shown)
        self.canvas.create_image(offset_x, offset_y, image=self.photo, anchor="nw")
        self.render_box = (offset_x, offset_y, float(shown_width), float(shown_height))
        for index, item in enumerate(self.objects):
            x1, y1, x2, y2 = item["bbox"]
            color = "#ff647c" if item.get("team") == "enemy" else "#43d6a0"
            left = offset_x + x1 * shown_width
            top = offset_y + y1 * shown_height
            right = offset_x + x2 * shown_width
            bottom = offset_y + y2 * shown_height
            self.canvas.create_rectangle(left, top, right, bottom, outline=color, width=2)
            self.canvas.create_text(
                left + 3,
                top + 3,
                text=f"{index + 1} {item['card_id']}",
                fill="white",
                anchor="nw",
                font=(FONT, 8, "bold"),
            )

    def _normalized_canvas_point(self, x: float, y: float) -> tuple[float, float] | None:
        offset_x, offset_y, width, height = self.render_box
        if not (offset_x <= x <= offset_x + width and offset_y <= y <= offset_y + height):
            return None
        return (
            min(1.0, max(0.0, (x - offset_x) / width)),
            min(1.0, max(0.0, (y - offset_y) / height)),
        )

    def _draw_begin(self, event: tk.Event[Any]) -> None:
        if self.image is None:
            return
        point = self._normalized_canvas_point(float(event.x), float(event.y))
        if point is None:
            return
        self.draw_start = point

    def _draw_move(self, event: tk.Event[Any]) -> None:
        if self.draw_start is None:
            return
        point = self._normalized_canvas_point(float(event.x), float(event.y))
        if point is None:
            return
        if self.preview_rectangle is not None:
            self.canvas.delete(self.preview_rectangle)
        offset_x, offset_y, width, height = self.render_box
        self.preview_rectangle = self.canvas.create_rectangle(
            offset_x + self.draw_start[0] * width,
            offset_y + self.draw_start[1] * height,
            offset_x + point[0] * width,
            offset_y + point[1] * height,
            outline="#ffbe55",
            width=2,
            dash=(4, 2),
        )

    def _draw_end(self, event: tk.Event[Any]) -> None:
        if self.draw_start is None:
            return
        start = self.draw_start
        self.draw_start = None
        point = self._normalized_canvas_point(float(event.x), float(event.y))
        self.preview_rectangle = None
        if point is None:
            self._render()
            return
        card_id = self.target_card_var.get().strip()
        if not card_id:
            self.status_var.set("请先填写要框选的 card_id。")
            self._render()
            return
        x1, x2 = sorted((start[0], point[0]))
        y1, y2 = sorted((start[1], point[1]))
        if x2 - x1 < 0.005 or y2 - y1 < 0.005:
            self.status_var.set("框选区域太小，未添加。")
            self._render()
            return
        self.objects.append(
            {
                "card_id": card_id,
                "team": self.team_var.get(),
                "bbox": [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)],
            }
        )
        self.dirty = True
        self._refresh_object_list()
        self._render()

    def _refresh_object_list(self) -> None:
        self.object_list.delete(0, "end")
        for index, item in enumerate(self.objects, start=1):
            self.object_list.insert(
                "end", f"{index:02d} {item.get('team', '?'):5s} {item.get('card_id', '?')}"
            )

    def _remove_object(self) -> None:
        selected = self.object_list.curselection()
        if not selected:
            return
        del self.objects[int(selected[0])]
        self.dirty = True
        self._refresh_object_list()
        self._render()

    def _clear_objects(self) -> None:
        if self.objects:
            self.objects.clear()
            self.dirty = True
            self._refresh_object_list()
            self._render()

    def _suggest_hand(self) -> None:
        if self.image is None or not self.catalog.cards:
            self.status_var.set("当前画面或卡牌目录不可用。")
            return
        if self.hand_recognizer is None:
            vision = {
                "card_slot_centers": [[0.3, 0.89], [0.5, 0.89], [0.69, 0.89], [0.875, 0.89]],
                "card_roi_half_width": 0.09,
                "card_roi_top": 0.825,
                "card_roi_bottom": 0.955,
                "card_match_ratio": 0.78,
                "card_match_min_good": 35,
                "card_match_min_margin": 1.45,
                "card_empty_max_keypoints": 150,
            }
            self.hand_recognizer = UniversalHandRecognizer(self.catalog, vision)
        matches = self.hand_recognizer.recognize(self.image)
        for variable, match in zip(self.hand_vars, matches):
            variable.set(match.card_id or "")
        recognized = sum(match.card_id is not None for match in matches)
        self.status_var.set(
            f"已给出 {recognized}/4 个手牌建议；请人工确认后再保存。"
        )

    def _save(self) -> bool:
        if not self.store or not self.samples:
            return False
        sample = self.samples[self.sample_index]
        try:
            saved = self.store.save_annotation(
                {
                    "sample_id": sample.sample_id,
                    "source": "human",
                    "usable": self.usable_var.get(),
                    "battle_phase": self.phase_var.get(),
                    "hand_cards": [variable.get().strip() or None for variable in self.hand_vars],
                    "objects": self.objects,
                    "notes": self.notes.get("1.0", "end-1c"),
                }
            )
        except ValueError as exc:
            messagebox.showerror("无法保存标注", str(exc), parent=self.root)
            return False
        self.annotations[sample.sample_id] = saved
        self.dirty = False
        self.status_var.set(f"已保存人工标注，版本 {saved['revision']}")
        return True

    def _save_and_next_unannotated(self) -> None:
        if self._save():
            self._next_unannotated()

    def _next_unannotated(self) -> None:
        if not self.samples or not self._confirm_discard():
            return
        total = len(self.samples)
        for offset in range(1, total + 1):
            index = (self.sample_index + offset) % total
            if self.samples[index].sample_id not in self.annotations:
                self._load_sample(index)
                return
        self.status_var.set("当前批次的画面已经全部标注。")

    def _previous(self) -> None:
        if self.sample_index > 0 and self._confirm_discard():
            self._load_sample(self.sample_index - 1)

    def _next(self) -> None:
        if self.sample_index + 1 < len(self.samples) and self._confirm_discard():
            self._load_sample(self.sample_index + 1)

    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        return messagebox.askyesno(
            "尚未保存",
            "当前标注尚未保存，确定放弃修改吗？",
            parent=self.root,
        )

    def _mark_dirty(self, *_args: Any) -> None:
        self.dirty = True

    def _notes_modified(self, _event: tk.Event[Any]) -> None:
        if self.notes.edit_modified():
            self.dirty = True
            self.notes.edit_modified(False)

    def _close(self) -> None:
        if self._confirm_discard():
            self.root.destroy()


def run_annotation(project_root: Path, catalog_path: Path | None = None) -> None:
    root = tk.Tk()
    AnnotationWindow(root, project_root, catalog_path)
    root.mainloop()
