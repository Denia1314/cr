"""Small canvas controls, with no browser runtime or extra UI dependency."""
from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont
from .animations import Animator, fade_in, mix
from .ui_theme import FONT, Palette


def rounded_shape(canvas, extent_x, extent_y, radius, **options):
    radius = max(0, min(radius, extent_x / 2, extent_y / 2))
    x, y, r = extent_x - 1, extent_y - 1, radius
    return canvas.create_polygon(
        1+r, 1, x-r, 1, x, 1, x, 1+r, x, y-r, x, y,
        x-r, y, 1+r, y, 1, y, 1, y-r, 1, 1+r, 1, 1,
        smooth=True, splinesteps=24, **options,
    )


class HoverButton(tk.Canvas):
    """Rounded action with reversible hover, press, focus and disabled states.

    The small Button-compatible surface keeps existing commands and callers.
    Only this control is redrawn during a transition, never the entire window.
    """
    def __init__(self, master, *, background, hover, foreground=Palette.TEXT, **kwargs):
        self.normal_background = background
        self.hover_background = hover
        self._button_options = dict(text="", font=(FONT, 10), padx=16, pady=10,
                             state="normal", command=None, anchor="center",
                             justify="center", wraplength=0, foreground=foreground)
        for key in list(kwargs):
            if key in self._button_options:
                self._button_options[key] = kwargs.pop(key)
        self._fill = background
        self._hovered = self._pressed = self._focused = False
        self._keyboard_pressed = False
        super().__init__(master, background=master.cget("background"),
                         highlightthickness=0, borderwidth=0, takefocus=True,
                         cursor="hand2", **kwargs)
        self.animator = Animator(self)
        self.bind("<Configure>", lambda _e: self._draw(), add="+")
        self.bind("<Enter>", self._enter)
        self.bind("<Leave>", self._leave)
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<FocusIn>", self._focus_in)
        self.bind("<FocusOut>", self._focus_out)
        self.bind("<KeyPress-space>", self._key_press)
        self.bind("<KeyRelease-space>", self._key_release)
        self.bind("<Return>", self._return)
        self._measure()

    def _measure(self):
        face = tkfont.Font(self, font=self._button_options["font"])
        text = str(self._button_options["text"])
        wrap = int(self._button_options["wraplength"])
        lines = []
        for paragraph in text.split("\n"):
            line = ""
            for character in paragraph:
                if wrap and line and face.measure(line + character) > wrap:
                    lines.append(line)
                    line = ""
                line += character
            lines.append(line)
        self._display_text = "\n".join(lines)
        width = max((face.measure(line) for line in lines), default=0)
        super().configure(width=width + 2 * self._button_options["padx"] + 4,
                          height=len(lines) * face.metrics("linespace") + 2 * self._button_options["pady"] + 4)
        self._draw()

    def _draw(self):
        if not self.winfo_exists():
            return
        self.delete("all")
        width, height = max(4, self.winfo_width()), max(4, self.winfo_height())
        disabled = self._button_options["state"] == "disabled"
        fill = mix(self.normal_background, Palette.CARD_ALT, .75) if disabled else self._fill
        if self._focused and not disabled:
            fill = mix(fill, self.hover_background, .35)
        rounded_shape(self, width, height, 9, fill=fill, outline=fill, width=1)
        anchor = self._button_options["anchor"]
        x = self._button_options["padx"] + 2 if anchor == "w" else width / 2
        self.create_text(x, height / 2 + int(self._pressed), text=self._display_text,
                         anchor=anchor, font=self._button_options["font"],
                         justify=self._button_options["justify"],
                         fill=Palette.FAINT if disabled else self._button_options["foreground"])

    def configure(self, cnf=None, **kwargs):
        if isinstance(cnf, dict):
            kwargs = {**cnf, **kwargs}
            cnf = None
        if not kwargs:
            return super().configure(cnf)
        measure = False
        for key in list(kwargs):
            if key in self._button_options:
                self._button_options[key] = kwargs.pop(key)
                measure |= key in {"text", "font", "padx", "pady", "wraplength"}
            elif key in {"background", "bg"}:
                self.normal_background = kwargs.pop(key)
                self._fill = self.normal_background
        if kwargs:
            super().configure(**kwargs)
        if self._button_options["state"] == "disabled":
            self.animator.cancel()
            self._pressed = self._keyboard_pressed = False
            self._fill = self.normal_background
        super().configure(cursor="arrow" if self._button_options["state"] == "disabled" else "hand2")
        super().configure(takefocus=self._button_options["state"] != "disabled")
        self._measure() if measure else self._draw()

    config = configure

    def cget(self, key):
        if key in self._button_options:
            return self._button_options[key]
        if key in {"background", "bg"}:
            return self._fill
        return super().cget(key)

    __getitem__ = cget

    def _animate_to(self, target):
        if self._button_options["state"] == "disabled":
            self.animator.cancel()
            self._fill = self.normal_background
            self._draw()
            return
        start = self._fill
        def update(progress):
            self._fill = mix(start, target, progress)
            self._draw()
        self.animator.tween("hover", update, 160)

    def _enter(self, _event=None):
        self._hovered = True
        if self._button_options["state"] != "disabled":
            self._animate_to(self.hover_background)

    def _leave(self, _event=None):
        self._hovered = self._pressed = False
        self._animate_to(self.normal_background)

    def _press(self, _event=None):
        if self._button_options["state"] != "disabled":
            self.focus_set()
            self._pressed = True
            self._animate_to(mix(self.hover_background, Palette.TEXT, .08))

    def _release(self, event):
        pressed = self._pressed
        self._pressed = False
        inside = 0 <= event.x < self.winfo_width() and 0 <= event.y < self.winfo_height()
        self._animate_to(self.hover_background if inside else self.normal_background)
        if pressed and inside:
            self.invoke()

    def _focus_in(self, _event):
        self._focused = True
        self._draw()

    def _focus_out(self, _event):
        self._focused = self._pressed = self._keyboard_pressed = False
        self._animate_to(self.hover_background if self._hovered else self.normal_background)

    def _return(self, event):
        # Keep Ctrl+Enter available to the window's Start shortcut, once.
        if event.state & 4:
            return None
        self.invoke()
        return "break"

    def _key_press(self, _event):
        if not self._keyboard_pressed:
            self._keyboard_pressed = True
            self._press()
        return "break"

    def _key_release(self, _event):
        if self._keyboard_pressed:
            self._keyboard_pressed = self._pressed = False
            self._animate_to(self.normal_background)
            self.invoke()
        return "break"

    def invoke(self):
        if self._button_options["state"] != "disabled" and self._button_options["command"]:
            return self._button_options["command"]()


class Surface(tk.Frame):
    """Rounded card with an inset content frame and animated border reveal."""
    def __init__(self, master, *, padding=14, **kwargs):
        super().__init__(master, background=master.cget("background"), **kwargs)
        self.canvas = tk.Canvas(self, background=master.cget("background"),
                                highlightthickness=0, borderwidth=0)
        self.canvas.place(relwidth=1, relheight=1)
        self.body = tk.Frame(self, background=Palette.CARD)
        self.body.pack(fill="both", expand=True, padx=padding, pady=padding)
        self._border = Palette.BORDER
        self.animator = Animator(self)
        self.bind("<Configure>", lambda _e: self._draw())

    def _draw(self):
        self.canvas.delete("all")
        rounded_shape(self.canvas, max(4, self.winfo_width()), max(4, self.winfo_height()),
                      14, fill=Palette.CARD, outline=self._border, width=1)

    def reveal(self):
        def update(progress):
            self._border = mix(Palette.SELECTION, Palette.BORDER, progress)
            self._draw()
        self.animator.tween("reveal", update, 280)


class ToolsPopover:
    """Animated tools list. Native Menu stays available as the command router."""
    def __init__(self, root, anchor, menu):
        self.root, self.anchor, self.menu = root, anchor, menu
        self.window = None
        root.bind("<Configure>", self._root_configure, add="+")

    def _root_configure(self, event):
        if event.widget is self.root:
            self.close()

    def toggle(self):
        if self.window is not None and self.window.winfo_exists():
            self.close()
            return
        window = self.window = tk.Toplevel(self.root)
        window.withdraw()
        window.overrideredirect(True)
        window.transient(self.root)
        window.configure(background=Palette.BORDER)
        panel = tk.Frame(window, background=Palette.CARD)
        panel.pack(fill="both", expand=True, padx=1, pady=1)
        tk.Label(panel, text="工具箱", bg=Palette.CARD, fg=Palette.TEXT,
                 font=(FONT, 11, "bold"), anchor="w").pack(fill="x", padx=16, pady=(14, 8))
        for index in range(self.menu.index("end") + 1):
            if self.menu.type(index) == "separator":
                tk.Frame(panel, bg=Palette.BORDER, height=1).pack(fill="x", padx=12, pady=6)
                continue
            label = self.menu.entrycget(index, "label")
            if self.menu.type(index) == "checkbutton":
                variable = self.menu.entrycget(index, "variable")
                label = ("✓  " if self.root.getvar(variable) else "    ") + label
            HoverButton(panel, text=label, anchor="w", padx=14, pady=8,
                        background=Palette.CARD, hover=Palette.CARD_ALT,
                        command=lambda i=index: self._invoke(i)).pack(fill="x", padx=6, pady=1)
        tk.Label(panel, text="Esc 关闭", bg=Palette.CARD, fg=Palette.FAINT,
                 font=(FONT, 8)).pack(anchor="w", padx=16, pady=(6, 12))
        window.update_idletasks()
        x = self.anchor.winfo_rootx() + self.anchor.winfo_width() + 10
        y = self.anchor.winfo_rooty()
        x = min(x, window.winfo_screenwidth() - window.winfo_reqwidth() - 8)
        y = min(y, window.winfo_screenheight() - window.winfo_reqheight() - 40)
        window.geometry(f"+{max(0,x)}+{max(0,y)}")
        window.bind("<Escape>", lambda _e: self.close())
        window.bind("<FocusOut>", self._lost_focus)
        window.deiconify()
        window.focus_set()
        window.animation = fade_in(window)

    def _lost_focus(self, _event):
        if self.window is None:
            return
        self.window.after_idle(self._check_focus)

    def _check_focus(self):
        if self.window is None:
            return
        focus = self.root.focus_get()
        if focus is None or focus.winfo_toplevel() is not self.window:
            self.close()

    def _invoke(self, index):
        self.close()
        self.menu.invoke(index)

    def close(self):
        window, self.window = self.window, None
        if window is not None and window.winfo_exists():
            window.destroy()
