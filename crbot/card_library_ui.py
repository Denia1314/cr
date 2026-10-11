"""Lazy, paginated Tk card and sprite browser using the shared light controls."""
from __future__ import annotations

import math
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from .animations import Animator, fade_in, mix
from .card_library import CardLibrary
from .ui_select import AnimatedSelect as _AnimatedSelect
from .ui_theme import FONT, Palette
from .ui_widgets import HoverButton, Surface


def label(master, text="", **kwargs):
    options = dict(text=text, bg=master.cget("bg"), fg=Palette.TEXT, font=(FONT, 9))
    return tk.Label(master, **(options | kwargs))


def button(master, text, command, **kwargs):
    return HoverButton(master, text=text, command=command, background=Palette.CARD,
                       hover=Palette.SELECTION, font=(FONT, 9), padx=12, pady=7, **kwargs)


def entry(master, variable):
    return tk.Entry(master, textvariable=variable, font=(FONT, 9), relief="flat",
                    bg=Palette.CARD, fg=Palette.TEXT, insertbackground=Palette.BLUE,
                    highlightthickness=1, highlightbackground=Palette.BORDER,
                    highlightcolor=Palette.BORDER)


def select(master, variable, values, command=None, **kwargs):
    widget = _AnimatedSelect(master, textvariable=variable, values=values, **kwargs)
    if command:
        widget.bind("<<ComboboxSelected>>", command)
    return widget


def thumbnail(path, size, master):
    try:
        if path is None:
            raise ValueError("没有图标")
        with Image.open(path) as raw:
            image = raw.convert("RGBA")
            image.thumbnail(size, Image.Resampling.LANCZOS)
    except (OSError, ValueError, TypeError):
        image = Image.new("RGBA", size, Palette.CARD_ALT)
    return ImageTk.PhotoImage(image, master=master)


class ImageTile(HoverButton):
    def __init__(self, master, photo, title, subtitle, command, width=146):
        self.photo, self.title, self.subtitle = photo, title, subtitle
        super().__init__(master, text="", background=Palette.CARD, hover=Palette.SELECTION,
                         command=command)
        tk.Canvas.configure(self, width=width, height=174)

    def _draw(self):
        super()._draw()
        width = max(4, self.winfo_width())
        self.create_image(width / 2, 57 + int(self._pressed), image=self.photo)
        self.create_text(width / 2, 121, text=self.title, width=width-14,
                         font=(FONT, 8, "bold"), fill=Palette.TEXT, justify="center")
        self.create_text(width / 2, 155, text=self.subtitle, width=width-12,
                         font=(FONT, 7), fill=Palette.MUTED, justify="center")


class CardLibraryWindow:
    PAGE_SIZE = 24

    def __init__(self, parent, library: CardLibrary):
        self.library = library
        self.window = tk.Toplevel(parent)
        self.window.title("Royal Lab · 卡牌库")
        self.window.configure(bg=Palette.BG)
        self.window.geometry(f"1120x820+{parent.winfo_rootx()+24}+{parent.winfo_rooty()+20}")
        self.window.minsize(1000, 700)
        if parent.attributes("-topmost"):
            self.window.attributes("-topmost", True)
        self.card_id = None
        self.unmapped = False
        self.page = 0
        self.photos = []
        self.query = tk.StringVar()
        self.group = tk.StringVar(value="全部切片")
        self.origin = tk.StringVar(value="全部来源")
        self.affiliation = tk.StringVar(value="全部阵营")
        self.busy = False
        self.results = queue.Queue()
        self.shell = tk.Frame(self.window, bg=Palette.BG)
        self.shell.pack(fill="both", expand=True, padx=22, pady=18)
        heading = tk.Frame(self.shell, bg=Palette.BG)
        heading.pack(fill="x")
        label(heading, "卡牌库", font=(FONT, 20, "bold")).pack(side="left")
        button(heading, "添加卡牌", self.add_card_dialog).pack(side="right")
        button(heading, "导出切片库", self.export_dialog).pack(side="right", padx=10)
        button(heading, "刷新", self.refresh).pack(side="right")
        self.summary = label(self.shell, fg=Palette.MUTED, anchor="w")
        self.summary.pack(fill="x", pady=(8, 14))
        self.toolbar = tk.Frame(self.shell, bg=Palette.BG)
        self.toolbar.pack(fill="x", pady=(0, 12))
        self.search_entry = entry(self.toolbar, self.query)
        self.search_entry.pack(side="left", fill="x", expand=True, ipady=8)
        label(self.toolbar, "搜索名称 / ID / 切片类别", fg=Palette.MUTED).pack(side="left", padx=12)
        button(self.toolbar, "未对应 / 场景素材", self.show_unmapped).pack(side="right")
        self.detail = tk.Frame(self.shell, bg=Palette.BG)
        self.filters = tk.Frame(self.shell, bg=Palette.BG)
        self.grid_shell = Surface(self.shell, padding=10)
        self.grid_shell.body.configure(bg=Palette.CARD_ALT)
        self.grid_shell.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(self.grid_shell.body, bg=Palette.CARD_ALT, highlightthickness=0,
                                width=100, height=100)
        scrollbar = ttk.Scrollbar(self.grid_shell.body, orient="vertical", command=self.canvas.yview)
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.tiles = tk.Frame(self.canvas, bg=Palette.CARD_ALT)
        self.canvas_item = self.canvas.create_window(0, 0, window=self.tiles, anchor="nw")
        self.tiles.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", self._resize)
        # This toplevel binding only scrolls this library, not the main workbench.
        self.window.bind("<MouseWheel>", lambda e: self.canvas.yview_scroll(-int(e.delta/120), "units"))
        footer = tk.Frame(self.shell, bg=Palette.BG)
        footer.pack(side="bottom", fill="x", before=self.grid_shell, pady=(12, 0))
        self.page_label = label(footer, fg=Palette.MUTED)
        self.page_label.pack(side="left")
        self.next_button = button(footer, "下一页  ›", lambda: self.change_page(1))
        self.next_button.pack(side="right")
        self.prev_button = button(footer, "‹  上一页", lambda: self.change_page(-1))
        self.prev_button.pack(side="right", padx=10)
        self.query_job = None
        self.poll_job = None
        self.window.bind("<Destroy>", self._destroy, add="+")
        self.query.trace_add("write", self._search_changed)
        self.columns = 6
        self.render()
        self.animation = fade_in(self.window)

    def _destroy(self, event):
        if event.widget is self.window:
            for job in (self.query_job, self.poll_job):
                if job:
                    self.window.after_cancel(job)
            self.query_job = self.poll_job = None

    def _search_changed(self, *_):
        if self.query_job:
            self.window.after_cancel(self.query_job)
        self.query_job = self.window.after(160, self._search)

    def _search(self):
        self.query_job = None
        self.page = 0
        self.render()

    def _resize(self, event):
        self.canvas.itemconfigure(self.canvas_item, width=event.width)
        columns = max(1, event.width // 154)
        if columns != self.columns:
            self.columns = columns
            self.render()

    def _reset_view(self):
        self.page = 0
        self.query.set("")
        self.group.set("全部切片")
        self.origin.set("全部来源")
        self.affiliation.set("全部阵营")
        for frame in (self.detail, self.filters):
            for child in frame.winfo_children():
                child.destroy()
            frame.pack_forget()

    def show_cards(self):
        self.card_id, self.unmapped = None, False
        self._reset_view()
        self.render()

    def show_unmapped(self):
        self.show_card(None, unmapped=True)

    def show_card(self, card_id, *, unmapped=False):
        self.card_id, self.unmapped = card_id, unmapped
        self._reset_view()
        self.detail.pack(fill="x", before=self.grid_shell, pady=(0, 12))
        top = tk.Frame(self.detail, bg=Palette.BG)
        top.pack(fill="x")
        button(top, "‹  全部卡牌", self.show_cards).pack(side="left", padx=(0, 12))
        card = self.library.catalog.get(card_id) if card_id else None
        title = (card.name_zh or card.name_en or card.card_id) if card else "未对应 / 场景素材"
        label(top, title, font=(FONT, 14, "bold")).pack(side="left")
        if card:
            button(top, "添加切片", self.add_slices_dialog).pack(side="right")
            button(top, "卡牌资料", self.metadata_dialog).pack(side="right", padx=10)
        button(top, "编辑对应关系", self.links_dialog).pack(side="right")
        if card:
            raw = self.library.raw_cards[card_id]
            source = "网站卡牌数据" if card_id in self.library.website_cards else (
                     "本地添加" if raw.get("library_source") == "local" else "已有卡牌数据")
            label(self.detail, f"{card.card_id}  ·  {card.elixir if card.elixir is not None else '?'} 圣水"
                  f"  ·  {card.kind} / {card.rarity}  ·  {source}", fg=Palette.MUTED).pack(anchor="w", pady=(9, 0))
        self.filters.pack(fill="x", before=self.grid_shell, pady=(0, 12))
        rows = self.library.unmapped if unmapped else self.library.by_card[card_id]
        self.group_select = select(self.filters, variable=self.group,
            values=["全部切片", *sorted({r['unit'] for r in rows})], command=self.filter_changed,
            font=(FONT, 8), width=16, pady=7)
        self.group_select.pack(side="left")
        for variable, values in ((self.origin, ["全部来源", "GitHub 原始", "本地添加"]),
                                 (self.affiliation, ["全部阵营", "来源阵营 0", "来源阵营 1"])):
            select(self.filters, variable=variable, values=values, command=self.filter_changed,
                font=(FONT, 8), width=9, pady=7
            ).pack(side="left", padx=(10, 0))
        self.render()
        self.grid_shell.reveal()

    def filter_changed(self, *_):
        self.page = 0
        self.render()

    def rows(self):
        if self.card_id is None and not self.unmapped:
            return self.library.search(self.query.get())
        rows = self.library.unmapped if self.unmapped else self.library.by_card[self.card_id]
        query = self.query.get().strip().casefold()
        return [r for r in rows if (self.group.get() == "全部切片" or self.group.get() == r["unit"])
                and (self.origin.get() == "全部来源" or
                     r["origin"] == ("github" if self.origin.get() == "GitHub 原始" else "local"))
                and (self.affiliation.get() == "全部阵营" or
                     r.get("affiliation") == int(self.affiliation.get()[-1]))
                and (not query or query in (r["unit"] + " " + r["file"].name).casefold())]

    def render(self):
        for child in self.tiles.winfo_children():
            child.destroy()
        self.photos = []
        cards_view = self.card_id is None and not self.unmapped
        rows = self.rows()
        pages = max(1, math.ceil(len(rows)/self.PAGE_SIZE))
        self.page = max(0, min(self.page, pages-1))
        self.summary.configure(text=f"{len(self.library.catalog.cards)} 张卡牌  ·  "
            f"{len(self.library.records):,} 张素材  ·  本地添加 {len(self.library.local['records'])} 张"
            + ("  ·  GitHub 切片尚未导入" if not self.library.source_manifest else ""))
        self.page_label.configure(text=f"第 {self.page+1} / {pages} 页  ·  {len(rows)} {'张卡牌' if cards_view else '张切片'}")
        self.prev_button.configure(state="normal" if self.page else "disabled")
        self.next_button.configure(state="normal" if self.page+1 < pages else "disabled")
        for i, row in enumerate(rows[self.page*self.PAGE_SIZE:(self.page+1)*self.PAGE_SIZE]):
            if cards_view:
                path = self.library.catalog.local_icon_path(row)
                title = row.name_zh or row.name_en or row.card_id
                count = len(self.library.by_card[row.card_id])
                subtitle = f"{row.elixir if row.elixir is not None else '?'} 圣水 · {count} 张切片"
                command = lambda cid=row.card_id: self.show_card(cid)
            else:
                path, title = row["file"], row["unit"]
                side = f"来源 {row['affiliation']}" if row.get("affiliation") is not None else "场景素材"
                subtitle = f"{side} · {'GitHub' if row['origin']=='github' else '本地'} · {row['size'][0]}×{row['size'][1]}"
                command = lambda record=row: self.image_dialog(record)
            photo = thumbnail(path, (100, 102), self.window)
            self.photos.append(photo)
            tile = ImageTile(self.tiles, photo, title, subtitle, command)
            tile.grid(row=i//self.columns, column=i%self.columns, sticky="ew", padx=4, pady=4)
        for col in range(max(8, self.columns)):
            self.tiles.grid_columnconfigure(col, weight=1 if col < self.columns else 0,
                                            minsize=0)
        if not rows:
            text = "暂时没有对应切片，可以添加切片或编辑对应关系。" if not cards_view else "没有找到卡牌。"
            label(self.tiles, text, fg=Palette.MUTED).grid(row=0, column=0, columnspan=self.columns, pady=60)
        self.canvas.yview_moveto(0)

    def change_page(self, delta):
        self.page += delta
        self.render()
        self.grid_shell.reveal()

    def refresh(self):
        try:
            self.library.reload()
            if self.card_id or self.unmapped:
                self.show_card(self.card_id, unmapped=self.unmapped)
            else:
                self.render()
        except (ValueError, OSError) as exc:
            self.error(exc)

    def error(self, exc, parent=None):
        messagebox.showerror("卡牌库", str(exc), parent=parent or self.window)

    def dialog(self, title, width=640, height=600):
        window = tk.Toplevel(self.window)
        window.title(title)
        window.configure(bg=Palette.BG)
        window.geometry(f"{width}x{height}+{self.window.winfo_rootx()+80}+{self.window.winfo_rooty()+40}")
        window.transient(self.window)
        window.grab_set()
        body = tk.Frame(window, bg=Palette.BG)
        body.pack(fill="both", expand=True, padx=22, pady=18)
        label(body, title, font=(FONT, 16, "bold")).pack(anchor="w", pady=(0, 16))
        window.animation = fade_in(window)
        return window, body

    def image_dialog(self, record):
        window, body = self.dialog("切片预览", 720, 690)
        photo = thumbnail(record["file"], (560, 400), window)
        canvas = tk.Canvas(body, bg=Palette.CARD_ALT, highlightthickness=0, height=360)
        canvas.pack(fill="both", expand=True)
        for x in range(0, 720, 20):
            for y in range(0, 500, 20):
                canvas.create_rectangle(x, y, x+20, y+20, outline="", fill=Palette.CARD if (x+y)//20%2 else Palette.CARD_ALT)
        image_id = canvas.create_image(300, 180, image=photo)
        canvas.photo = photo
        canvas.bind("<Configure>", lambda e: canvas.coords(image_id, e.width/2, e.height/2))
        label(body, f"类别：{record['unit']}\n{record['file'].name}\n"
              f"尺寸：{record['size'][0]} × {record['size'][1]}  ·  来源阵营：{record.get('affiliation')}\n"
              f"关联卡牌：{', '.join(self.library.links[record['unit']]) or '未对应'}",
              justify="left", anchor="w", wraplength=660, fg=Palette.MUTED).pack(fill="x", pady=12)
        button(body, "打开文件位置", lambda: self.open_folder(record["file"].parent)).pack(side="right")

    @staticmethod
    def open_folder(path):
        import os
        os.startfile(path)

    def metadata_dialog(self):
        card = self.library.raw_cards[self.card_id]
        website = self.library.website_cards.get(self.card_id)
        window, body = self.dialog("卡牌资料", 790, 730)
        text = tk.Text(body, bg=Palette.CARD, fg=Palette.TEXT, font=(FONT, 9), relief="flat",
                       wrap="word", padx=12, pady=12)
        text.pack(fill="both", expand=True)
        scroll = ttk.Scrollbar(text, command=text.yview)
        scroll.pack(side="right", fill="y")
        text.configure(yscrollcommand=scroll.set)
        # Show downloaded values, never fill absent statistics with guesses.
        names = {"hitSpeed": "攻击间隔", "firstHitSpeed": "首次攻击间隔", "speed": "移动速度",
            "deployTime": "部署时间", "range": "射程", "targets": "攻击目标", "count": "数量",
            "transport": "移动方式", "lifetime": "存活时间", "projectileSpeed": "投射物速度",
            "spawnSpeed": "生成间隔", "splashRadius": "范围伤害半径", "level": "等级",
            "hitpoints": "生命值", "damage": "伤害", "damagePerSecond": "每秒伤害",
            "areaDamage": "范围伤害", "deathDamage": "死亡伤害", "towerDamage": "对塔伤害"}
        stats = (website or {}).get("battle_stats") or card.get("detail_stats", {})
        lines = [card.get("name_zh") or card.get("name_en") or self.card_id,
                 f"卡牌 ID：{self.card_id}", f"费用：{card.get('elixir', '?')} 圣水",
                 f"类型：{card.get('kind', '?')}   稀有度：{card.get('rarity', '?')}", "",
                 "网站属性" if website else "已有属性"]
        for key, value in stats.items():
            if key != "levelStats":
                lines.append(f"{names.get(key, key)}：{value}")
        if not stats:
            lines.append("尚未补充战斗属性。")
        if stats.get("levelStats"):
            lines.extend(["", "各等级数值"])
            for row in stats["levelStats"]:
                lines.append("  ·  ".join(f"{names.get(k, k)} {v}" for k, v in row.items()))
        lines.extend(["", "切片类别", *sorted({r['unit'] for r in self.library.by_card[self.card_id]}), "",
            "共享兵种与进化切片保留各自类别；素材对应关系用于管理，不替代战斗属性。"])
        text.insert("1.0", "\n".join(lines))
        text.configure(state="disabled")

    def add_card_dialog(self):
        window, body = self.dialog("添加卡牌", 640, 700)
        variables = {key: tk.StringVar(value=value) for key, value in
            {"card_id": "", "name_zh": "", "name_en": "", "elixir": "3", "kind": "troop", "rarity": "common", "icon": ""}.items()}
        fields = tk.Frame(body, bg=Palette.BG)
        fields.pack(fill="x")
        fields.grid_columnconfigure(1, weight=1)
        for row, (key, title) in enumerate((("card_id", "卡牌 ID"), ("name_zh", "中文名称"),
            ("name_en", "英文名称"), ("elixir", "圣水费用"), ("kind", "类型"), ("rarity", "稀有度"))):
            label(fields, title).grid(row=row, column=0, sticky="w", padx=(0, 14), pady=8)
            if key in {"kind", "rarity"}:
                values = ["troop", "building", "spell"] if key == "kind" else ["common", "rare", "epic", "legendary", "champion"]
                widget = select(fields, variable=variables[key], values=values,
                    font=(FONT, 9), width=14, pady=7)
            else:
                widget = entry(fields, variables[key])
            widget.grid(row=row, column=1, sticky="ew", pady=8, ipady=4)
        def choose_icon():
            path = filedialog.askopenfilename(parent=window, title="选择卡牌图标", filetypes=[("图片", "*.png *.jpg *.jpeg *.webp")])
            if path:
                variables["icon"].set(path)
        button(body, "选择卡牌图标", choose_icon).pack(anchor="w", pady=(12, 4))
        label(body, textvariable=variables["icon"], wraplength=570, fg=Palette.MUTED).pack(anchor="w")
        label(body, "保存后写入机器人使用的卡牌库，并保留备份。\n新卡的战斗属性与识别模型支持需要另外补充。",
              justify="left", fg=Palette.MUTED).pack(anchor="w", pady=14)
        def save():
            try:
                values = {key: value.get().strip() for key, value in variables.items()}
                if not values["icon"]:
                    raise ValueError("请选择卡牌图标")
                values["elixir"] = int(values["elixir"])
                values["icon"] = Path(values["icon"])
                self.library.add_card(**values)
                window.destroy()
                self.show_card(values["card_id"])
            except (OSError, ValueError) as exc:
                self.error(exc, window)
        button(body, "保存卡牌", save, foreground=Palette.BLUE).pack(side="bottom", anchor="e")

    def add_slices_dialog(self):
        window, body = self.dialog("添加切片", 690, 660)
        unit = tk.StringVar(value=self.card_id.replace("_", "-"))
        current = sorted({r["unit"] for r in self.library.by_card[self.card_id]})
        if current:
            unit.set(current[0])
        label(body, "切片类别（可选择现有类别，也可填写新类别）", fg=Palette.MUTED).pack(anchor="w")
        select(body, variable=unit, values=current or self.library.classes or [unit.get()],
            font=(FONT, 9), pady=7).pack(fill="x", pady=8)
        entry(body, unit).pack(fill="x", ipady=8)
        label(body, "原始数据中的阵营编号", fg=Palette.MUTED).pack(anchor="w", pady=(18, 8))
        side = tk.StringVar(value="0")
        select(body, variable=side, values=["0", "1"], font=(FONT, 9), width=5, pady=7).pack(anchor="w")
        paths = []
        files_label = label(body, "尚未选择图片", fg=Palette.MUTED, wraplength=630, justify="left")
        def choose():
            selected = filedialog.askopenfilenames(parent=window, title="选择透明 PNG 切片（可多选）", filetypes=[("PNG", "*.png")])
            if selected:
                paths[:] = [Path(p) for p in selected]
                files_label.configure(text=f"已选择 {len(paths)} 张图片\n" + "、".join(p.name for p in paths[:4]))
        button(body, "选择切片图片", choose).pack(anchor="w", pady=(18, 8))
        files_label.pack(anchor="w")
        label(body, "保留透明背景，批量添加时会检查图片并去重。\n现有类别的共享卡牌关系会保留；进化和召唤物使用各自类别。\n编号表示数据集的来源阵营，不替代实时画面的阵营识别。",
              justify="left", fg=Palette.MUTED, wraplength=630).pack(anchor="w", pady=20)
        def save():
            try:
                count = self.library.add_slices(paths, unit.get().strip(), int(side.get()), [self.card_id])
                window.destroy()
                self.show_card(self.card_id)
                messagebox.showinfo("添加完成", f"已添加 {count} 张新切片。\n可导出切片库供后续训练使用。", parent=self.window)
            except (ValueError, OSError) as exc:
                self.error(exc, window)
        button(body, "添加到当前卡牌", save, foreground=Palette.BLUE).pack(side="bottom", anchor="e")

    def links_dialog(self):
        window, body = self.dialog("编辑对应关系", 760, 740)
        current = self.library.unmapped if self.unmapped else self.library.by_card.get(self.card_id, [])
        unit = tk.StringVar(value=self.group.get() if self.group.get() in self.library.classes else
                            (current[0]["unit"] if current else (self.library.classes[0] if self.library.classes else "")))
        if not unit.get():
            window.destroy()
            messagebox.showinfo("对应关系", "尚未导入任何切片；先打开卡牌添加切片。", parent=self.window)
            return
        query = tk.StringVar()
        selected = set(self.library.links[unit.get()])
        choices = self.library.catalog.cards
        label(body, "选择切片类别，然后勾选所有对应卡牌。共享兵种可以关联多张卡牌。",
              fg=Palette.MUTED, wraplength=690).pack(anchor="w", pady=(0, 10))
        def load_unit(*_):
            selected.clear()
            selected.update(self.library.links[unit.get()])
            fill()
        select(body, variable=unit, values=self.library.classes, command=load_unit,
            font=(FONT, 9), pady=7).pack(fill="x")
        entry(body, query).pack(fill="x", ipady=7, pady=10)
        area = tk.Frame(body, bg=Palette.BG)
        area.pack(fill="both", expand=True)
        listing = tk.Listbox(area, selectmode="multiple", exportselection=False, font=(FONT, 9),
            relief="flat", bg=Palette.CARD, fg=Palette.TEXT, selectbackground=Palette.SELECTION,
            selectforeground=Palette.BLUE, activestyle="none", highlightthickness=0)
        listing.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(area, command=listing.yview)
        scroll.pack(side="right", fill="y")
        listing.configure(yscrollcommand=scroll.set)
        displayed = []
        def capture(*_):
            selected.difference_update(c.card_id for c in displayed)
            selected.update(displayed[i].card_id for i in listing.curselection())
        def fill(*_):
            displayed[:] = self.library.search(query.get())
            listing.delete(0, "end")
            for i, c in enumerate(displayed):
                listing.insert("end", f"{c.name_zh or c.name_en}  ·  {c.card_id}")
                if c.card_id in selected:
                    listing.selection_set(i)
        listing.bind("<<ListboxSelect>>", capture)
        query.trace_add("write", fill)
        fill()
        def save():
            try:
                capture()
                self.library.set_links(unit.get(), sorted(selected))
                window.destroy()
                self.refresh()
            except (ValueError, OSError) as exc:
                self.error(exc, window)
        button(body, "保存对应关系", save, foreground=Palette.BLUE).pack(anchor="e", pady=(14, 0))

    def export_dialog(self):
        if self.busy:
            messagebox.showinfo("切片库", "正在导出，请稍候。", parent=self.window)
            return
        path = filedialog.asksaveasfilename(parent=self.window, title="选择新切片库目录的名称",
            initialdir=self.library.root / "training/exports", initialfile="card-library-slices")
        if not path:
            return
        self.busy = True
        self.summary.configure(text="正在校验并导出切片库…")
        # A separate backend instance prevents UI reloads racing with this worker.
        def worker():
            try:
                backend = CardLibrary(self.library.root, self.library.catalog_path, self.library.slices_path)
                result = backend.export_snapshot(Path(path))
                self.results.put((True, result))
            except Exception as exc:
                self.results.put((False, str(exc)))
        threading.Thread(target=worker, name="card-library-export", daemon=True).start()
        def poll():
            try:
                ok, result = self.results.get_nowait()
            except queue.Empty:
                self.poll_job = self.window.after(100, poll)
                return
            self.busy = False
            self.render()
            if ok:
                messagebox.showinfo("导出完成", f"已导出 {result['total_images']} 张素材到：\n{path}\n\n"
                    "此目录可用于切片合成或真实战场数据导出。训练与模型加载需要另外运行。", parent=self.window)
            else:
                self.error(result)
        self.poll_job = self.window.after(100, poll)
