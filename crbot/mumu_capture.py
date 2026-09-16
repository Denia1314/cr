"""Read-only MuMu renderer IPC capture using the emulator's bundled SDK."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from threading import RLock

from PIL import Image


class MumuCapture:
    def __init__(self, install_dir: Path, index: int):
        if os.name != "nt":
            raise OSError("MuMu IPC requires Windows")
        root = Path(install_dir).resolve()
        candidates = [
            root / "nx_main/sdk/external_renderer_ipc.dll",
            *sorted(root.glob("nx_device/*/shell/sdk/external_renderer_ipc.dll"), reverse=True),
            root / "shell/sdk/external_renderer_ipc.dll",
        ]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise OSError("MuMu renderer SDK not found")
        self.lock = RLock()
        self.library = ctypes.CDLL(str(path))
        self.library.nemu_connect.argtypes = [ctypes.c_wchar_p, ctypes.c_int]
        self.library.nemu_connect.restype = ctypes.c_int
        self.library.nemu_disconnect.argtypes = [ctypes.c_int]
        self.library.nemu_capture_display.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_void_p,
        ]
        self.library.nemu_capture_display.restype = ctypes.c_int
        self.display_function = getattr(self.library, "nemu_get_display_id", None)
        if self.display_function:
            self.display_function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
            self.display_function.restype = ctypes.c_int
        self.handle = self.library.nemu_connect(str(root), int(index))
        if not self.handle:
            raise OSError("MuMu renderer connection failed")
        self.buffer = None
        self.size = None

    def capture(self) -> Image.Image:
        with self.lock:
            if not self.handle:
                raise OSError("MuMu renderer disconnected")
            display = self.display_function(self.handle, b"default", 0) if self.display_function else 0
            if display < 0:
                raise OSError("MuMu display unavailable")
            for _ in range(2):
                width, height = ctypes.c_int(), ctypes.c_int()
                if self.buffer is None:
                    code = self.library.nemu_capture_display(
                        self.handle, display, 0, ctypes.byref(width), ctypes.byref(height), None,
                    )
                    if code or not (
                        0 < width.value <= 8192 and 0 < height.value <= 8192
                        and width.value * height.value <= 32 * 1024 * 1024
                    ):
                        raise OSError("Invalid MuMu capture dimensions")
                    self.size = (width.value, height.value)
                    self.buffer = ctypes.create_string_buffer(width.value * height.value * 4)
                width.value, height.value = self.size
                code = self.library.nemu_capture_display(
                    self.handle, display, len(self.buffer),
                    ctypes.byref(width), ctypes.byref(height), self.buffer,
                )
                if code == 0 and (width.value, height.value) == self.size:
                    # Copy bottom-up RGBA pixels; later captures must not mutate
                    # an image that recognition or recording is still using.
                    # Decode bottom-up RGBA directly into owned RGB storage.
                    # Avoid allocating/flipping two full RGBA images per frame.
                    return Image.frombytes("RGB", self.size, self.buffer,
                                           "raw", "RGBX", 0, -1)
                self.buffer = None
            raise OSError("MuMu capture failed or resolution changed repeatedly")

    def close(self) -> None:
        with self.lock:
            self.buffer = None
            if self.handle:
                handle, self.handle = self.handle, 0
                self.library.nemu_disconnect(handle)
