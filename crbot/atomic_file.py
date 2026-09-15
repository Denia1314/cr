"""Same-directory atomic writes tolerant of transient Windows sharing conflicts."""
from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        # Windows readers/antivirus may temporarily deny delete sharing. Keep the
        # original destination intact; never unlink it to force a replacement.
        delays = (.025, .05, .1, .2, .4, .4, .4, .4)
        for attempt in range(len(delays) + 1):
            try:
                os.replace(temporary, path)
                break
            except OSError as exc:
                if getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == len(delays):
                    raise
                time.sleep(delays[attempt])
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            # Cleanup must not hide the original write failure (or turn a
            # committed replacement into a reported failure).
            logging.getLogger(__name__).warning("Could not clean temporary file %s", temporary, exc_info=True)
