"""Lightweight animated select fields and rounded option flyouts."""
from __future__ import annotations

import math
import sys
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

from .animations import Animator, mix
from .ui_theme import FONT, Palette
from .ui_widgets import HoverButton, rounded_shape


class ChoiceRow(HoverButton):
    def __init__(self, master, *, selected, **kwargs):
        self.selected = selected
        super().__init__(master, background=Palette.SELECTION if selected else Palette.CARD,
                         hover="#CFE0F8" if selected else Palette.CARD_ALT,
                         foreground=Palette.BLUE if selected else Palette.TEXT,
                         anchor="w", padx=16, pady=7, **kwargs)

    def _draw(self):
        super()._draw()
        if self.selected:
            self.create_line(5, 11, 5, max(12, self.winfo_height()-11),
                             fill=Palette.BLUE, width=3, capstyle="round")


class AnimatedSelect(HoverButton):
    """Readonly Combobox-compatible selection without the native listbox.

    Highlighting with arrows is a draft. Only confirming a row emits the
    existing selection event, so opening or cancelling never writes settings.
    """
    def __init__(self, master, *, textvariable, values, width=20, state="readonly", **kwargs):
        self.variable = textvariable
        self.values = tuple(values)
        self.field_chars = width
        self.popup = None
        self.rows = []
        self.scroll_canvas = None
        self.active_index = 0
        self._closing = False
        self._close_on_release = False
        self._arrow = 0.0
        self._border = Palette.BORDER
        self._popup_position = (0, 0)
        kwargs.setdefault("font", (FONT, 9))
        kwargs.setdefault("pady", 7)
        super().__init__(master, text=textvariable.get(), state=state, padx=12,
                         background=Palette.SURFACE, hover=Palette.CARD_ALT,
                         command=self.toggle, **kwargs)
        self._trace = self.variable.trace_add("write", self._variable_changed)
        self.bind("<Destroy>", self._destroy_select, add="+")
        for key in ("<Down>", "<Alt-Down>", "<F4>", "<KeyPress-space>"):
            self.bind(key, self._open_key)
        self.bind("<Up>", lambda _e: self._open_key(move=-1))
        self.bind("<KeyRelease-space>", lambda _e: "break")
        self.bind("<Escape>", self._escape)
        self.owner = self.winfo_toplevel()
        self._owner_geometry = self._owner_bounds()
        self._owner_bindings = [
            ("<Configure>", self.owner.bind("<Configure>", self._owner_moved, add="+")),
            ("<ButtonPress-1>", self.owner.bind("<ButtonPress-1>", self._outside_click, add="+")),
        ]

    def _measure(self):
        face = tkfont.Font(self, font=self._button_options["font"])
        width = max(face.measure("0") * self.field_chars,
                    max((face.measure(str(value)) for value in self.values), default=0)) + 40
        tk.Canvas.configure(self, width=width,
                            height=face.metrics("linespace")+2*self._button_options["pady"]+4)
        self._draw()

    def _draw(self):
        if not self.winfo_exists():
            return
        self.delete("all")
        w, h = max(4, self.winfo_width()), max(4, self.winfo_height())
        disabled = self._button_options["state"] == "disabled"
        fill = Palette.CARD_ALT if disabled else self._fill
        outline = Palette.BORDER if disabled else self._border
        rounded_shape(self, w, h, 9, fill=fill, outline=outline, width=1)
        text = str(self._button_options["text"])
        face = tkfont.Font(self, font=self._button_options["font"])
        available = max(0, w-44)
        if face.measure(text) > available:
            while text and face.measure(text+"…") > available:
                text = text[:-1]
            text += "…"
        color = Palette.FAINT if disabled else Palette.TEXT
        self.create_text(13, h/2, text=text, anchor="w", font=self._button_options["font"], fill=color)
        angle = math.pi * self._arrow
        points = []
        for x, y in ((-4,-2), (0,2), (4,-2)):
            points.extend((w-19+x*math.cos(angle)-y*math.sin(angle),
                           h/2+x*math.sin(angle)+y*math.cos(angle)))
        self.create_line(*points, fill=color, width=1.5, capstyle="round", joinstyle="round")

    def configure(self, cnf=None, **kwargs):
        if isinstance(cnf, dict):
            kwargs = {**cnf, **kwargs}
            cnf = None
        changed = False
        for key in ("values", "width"):
            if key in kwargs:
                value = kwargs.pop(key)
                if key == "values":
                    self.values = tuple(value)
                else:
                    self.field_chars = int(value)
                changed = True
        if kwargs.get("state") == "disabled" or changed:
            self.close(immediate=True)
        result = super().configure(cnf, **kwargs) if kwargs or cnf is not None else None
        if changed:
            self._measure()
        return result

    config = configure

    def cget(self, key):
        if key == "values":
            return self.values
        if key == "textvariable":
            return self.variable
        if key == "width":
            return self.field_chars
        return super().cget(key)

    __getitem__ = cget

    def get(self):
        return self.variable.get()

    def set(self, value):
        self.variable.set(value)

    def _variable_changed(self, *_args):
        super().configure(text=self.variable.get())

    def _animate_to(self, target):
        super()._animate_to(target)
        start = self._border
        end = "#B8C4D3" if self._hovered or self._focused else Palette.BORDER
        def update(progress):
            self._border = mix(start, end, progress)
            self._draw()
        if self._button_options["state"] != "disabled":
            self.animator.tween("border", update, 140)

    def _animate_arrow(self, target):
        start = self._arrow
        def update(progress):
            self._arrow = start+(target-start)*progress
            self._draw()
        self.animator.tween("arrow", update, 140)

    def _open_key(self, _event=None, *, move=0):
        self.open()
        if move:
            self._move(move)
        return "break"

    def _escape(self, _event=None):
        if self.popup is not None:
            self.close(restore_focus=True)
            return "break"

    def toggle(self):
        if self.popup is None or self._closing:
            self.open()
        else:
            self.close(restore_focus=True)

    def _press(self, event=None):
        # Moving focus to the field may dismiss the flyout before mouse-up.
        # Remember the click's intent so that dismissal cannot reopen it.
        self._close_on_release = self.popup is not None and not self._closing
        super()._press(event)

    def _release(self, event):
        if self._close_on_release:
            self._close_on_release = False
            self._pressed = False
            self.close(restore_focus=True)
            self._animate_to(self.hover_background if self._hovered else self.normal_background)
        else:
            super()._release(event)

    def open(self):
        if self._button_options["state"] == "disabled" or not self.values:
            return
        self.owner.update_idletasks()
        self._owner_geometry = self._owner_bounds()
        if self.popup is not None:
            if self._closing:
                self._closing = False
                self._show_animation()
            return
        popup = self.popup = tk.Toplevel(self.owner)
        popup.withdraw()
        popup.overrideredirect(True)
        popup.transient(self.owner)
        popup.attributes("-topmost", self.owner.attributes("-topmost"))
        corner = "#FF00FF" if sys.platform == "win32" else Palette.BG
        popup.configure(background=corner)
        if sys.platform == "win32":
            popup.attributes("-transparentcolor", corner)
        backdrop = tk.Canvas(popup, bg=corner, highlightthickness=0, borderwidth=0)
        backdrop.place(relwidth=1, relheight=1)
        panel = tk.Frame(popup, bg=Palette.CARD)
        panel.pack(fill="both", expand=True, padx=8, pady=8)
        self.active_index = self.values.index(self.get()) if self.get() in self.values else 0
        self.rows = []
        self.scroll_canvas = None
        rows_parent = panel
        if len(self.values) > 10:
            self.scroll_canvas = tk.Canvas(panel, bg=Palette.CARD, highlightthickness=0)
            scrollbar = ttk.Scrollbar(panel, orient="vertical", command=self.scroll_canvas.yview)
            scrollbar.pack(side="right", fill="y")
            self.scroll_canvas.pack(side="left", fill="both", expand=True)
            self.scroll_canvas.configure(yscrollcommand=scrollbar.set)
            rows_parent = tk.Frame(self.scroll_canvas, bg=Palette.CARD)
            item = self.scroll_canvas.create_window(0, 0, window=rows_parent, anchor="nw")
            self.scroll_canvas.bind("<Configure>", lambda e: self.scroll_canvas.itemconfigure(item, width=e.width))
            popup.bind("<MouseWheel>", lambda e: self.scroll_canvas.yview_scroll(-int(e.delta/120), "units"))
        for index, value in enumerate(self.values):
            row = ChoiceRow(rows_parent, text=str(value), selected=value == self.get(),
                            font=self._button_options["font"], command=lambda i=index: self._choose(i))
            row.pack(fill="x", pady=2)
            self.rows.append(row)
        if self.scroll_canvas is not None:
            rows_parent.update_idletasks()
            self.scroll_canvas.configure(width=rows_parent.winfo_reqwidth(),
                height=min(rows_parent.winfo_reqheight(), 440, max(120, self.owner.winfo_screenheight()-160)),
                scrollregion=(0, 0, rows_parent.winfo_reqwidth(), rows_parent.winfo_reqheight()))
        popup.bind("<Configure>", lambda e: self._draw_popup(e, backdrop))
        popup.bind("<Down>", lambda _e: self._move(1))
        popup.bind("<Up>", lambda _e: self._move(-1))
        popup.bind("<Home>", lambda _e: self._activate(0))
        popup.bind("<End>", lambda _e: self._activate(len(self.rows)-1))
        popup.bind("<Escape>", lambda _e: self._cancel_key())
        popup.bind("<F4>", lambda _e: self._cancel_key())
        popup.bind("<Alt-Up>", lambda _e: self._cancel_key())
        popup.bind("<Return>", lambda _e: self._choose(self.active_index))
        popup.bind("<Tab>", lambda _e: self._tab(1))
        popup.bind("<Shift-Tab>", lambda _e: self._tab(-1))
        popup.bind("<FocusOut>", self._popup_focus_out)
        popup.bind("<Destroy>", self._popup_destroyed, add="+")
        popup.update_idletasks()
        width = max(self.winfo_width(), popup.winfo_reqwidth())
        height = popup.winfo_reqheight()
        left, top = self.winfo_vrootx(), self.winfo_vrooty()
        right, bottom = left+self.winfo_vrootwidth(), top+self.winfo_vrootheight()
        x = max(left, min(self.winfo_rootx(), right-width-6))
        below = self.winfo_rooty()+self.winfo_height()+6
        y = below if below+height <= bottom-8 else self.winfo_rooty()-height-6
        y = max(top+8, min(y, bottom-height-8))
        self._popup_position = (x, y)
        popup.geometry(f"{width}x{height}+{x}+{y}")
        popup.animator = Animator(popup)
        popup.attributes("-alpha", 0.0)
        popup.deiconify()
        self._activate(self.active_index)
        self._show_animation()

    def _draw_popup(self, event, backdrop):
        if event.widget is self.popup:
            backdrop.delete("all")
            rounded_shape(backdrop, event.width, event.height, 12,
                          fill=Palette.CARD, outline=Palette.BORDER, width=1)

    def _show_animation(self):
        popup = self.popup
        start = float(popup.attributes("-alpha"))
        x, y = self._popup_position
        def update(progress):
            popup.attributes("-alpha", start+(1-start)*progress)
            popup.geometry(f"+{x}+{y+round(6*(1-progress))}")
        popup.animator.tween("visibility", update, 170)
        self._animate_arrow(1.0)
        popup.focus_force()
        self._activate(self.active_index)

    def _activate(self, index):
        if self.popup is not None and self.rows:
            self.active_index = index % len(self.rows)
            row = self.rows[self.active_index]
            if self.scroll_canvas is not None:
                self.scroll_canvas.update_idletasks()
                total = row.master.winfo_reqheight()
                top = self.scroll_canvas.canvasy(0)
                bottom = top+self.scroll_canvas.winfo_height()
                if row.winfo_y() < top:
                    self.scroll_canvas.yview_moveto(row.winfo_y()/total)
                elif row.winfo_y()+row.winfo_height() > bottom:
                    self.scroll_canvas.yview_moveto((row.winfo_y()+row.winfo_height()-self.scroll_canvas.winfo_height())/total)
            row.focus_set()
        return "break"

    def _move(self, delta):
        return self._activate(self.active_index+delta)

    def _choose(self, index):
        if self._closing or self._button_options["state"] == "disabled":
            return "break"
        self.variable.set(self.values[index])
        self.close(restore_focus=True)
        self.event_generate("<<ComboboxSelected>>")
        return "break"

    def _cancel_key(self):
        self.close(restore_focus=True)
        return "break"

    def _tab(self, direction):
        target = self.tk.call("tk_focusNext" if direction > 0 else "tk_focusPrev", self._w)
        self.close()
        if target:
            self.owner.focus_force()
            self.nametowidget(target).focus_set()
        return "break"

    def close(self, *, immediate=False, restore_focus=False):
        popup = self.popup
        if popup is None:
            return
        if restore_focus and self.winfo_exists():
            self.owner.focus_force()
            self.focus_set()
        if immediate:
            popup.destroy()
            self.animator.cancel("arrow")
            self._arrow = 0.0
            self._draw()
            return
        if self._closing:
            return
        self._closing = True
        start = float(popup.attributes("-alpha"))
        x, y = self._popup_position
        def update(progress):
            popup.attributes("-alpha", start*(1-progress))
            popup.geometry(f"+{x}+{y-round(3*progress)}")
        popup.animator.tween("visibility", update, 110, complete=popup.destroy)
        self._animate_arrow(0.0)

    def _popup_destroyed(self, event):
        if event.widget is self.popup:
            self.popup = None
            self.rows = []
            self.scroll_canvas = None
            self._closing = False

    def _popup_focus_out(self, _event):
        if self.popup is not None:
            self.popup.after_idle(self._check_focus)

    def _check_focus(self):
        if self.popup is None or self._closing:
            return
        focus = self.owner.focus_get()
        if focus is None or focus.winfo_toplevel() is not self.popup:
            self.close()

    def _owner_moved(self, event):
        if event.widget is self.owner:
            bounds = self._owner_bounds()
            if bounds != self._owner_geometry:
                self._owner_geometry = bounds
                self.close(immediate=True)

    def _owner_bounds(self):
        return (self.owner.winfo_rootx(), self.owner.winfo_rooty(),
                self.owner.winfo_width(), self.owner.winfo_height())

    def _outside_click(self, event):
        if event.widget is not self:
            self.close()

    def _destroy_select(self, event):
        if event.widget is self:
            self.close(immediate=True)
            self.variable.trace_remove("write", self._trace)
            for sequence, binding in self._owner_bindings:
                self.owner.unbind(sequence, binding)
