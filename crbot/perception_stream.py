"""Latest-only background visual preprocessing, independent of decisions."""
from dataclasses import dataclass
from threading import Condition, Event, Thread
import time

from .battle_perception import _level_badge_candidates


@dataclass(frozen=True)
class PreparedFrame:
    frame: object
    hand: tuple
    badges: tuple
    finished: float


class PerceptionStream:
    def __init__(self, frames, hand_recognizer):
        self.frames, self.hand_recognizer = frames, hand_recognizer
        self.condition, self.stopped = Condition(), Event()
        self.result, self.error = None, None
        self.processed = self.skipped = 0
        self.last_work_s = self.last_interval_s = None
        self.thread = Thread(target=self._run, name="battle-perception", daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        sequence, last_finished = 0, None
        try:
            while not self.stopped.is_set():
                try:
                    frame = self.frames.get(sequence=sequence, timeout=.2)
                except TimeoutError:
                    continue
                if self.stopped.is_set():
                    break
                started = time.monotonic()
                hand = tuple(self.hand_recognizer.recognize(frame.image)) if self.hand_recognizer else ()
                badges = tuple(_level_badge_candidates(frame.image))
                finished = time.monotonic()
                with self.condition:
                    self.result = PreparedFrame(frame, hand, badges, finished)
                    self.processed += 1
                    self.skipped += max(0, frame.sequence - sequence - 1)
                    self.last_work_s = finished - started
                    self.last_interval_s = None if last_finished is None else finished - last_finished
                    self.condition.notify_all()
                sequence, last_finished = frame.sequence, finished
        except Exception as exc:
            with self.condition:
                self.error = exc
                self.condition.notify_all()

    def latest(self):
        with self.condition:
            if self.error is not None:
                raise self.error
            return self.result

    def get(self, *, sequence=0, timeout=8.):
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                if self.error is not None:
                    raise self.error
                if self.stopped.is_set():
                    raise RuntimeError("perception stopped")
                if self.result is not None and self.result.frame.sequence > sequence:
                    return self.result
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("fresh perception timeout")
                self.condition.wait(min(.1, remaining))

    def status(self):
        with self.condition:
            return dict(processed=self.processed, skipped_capture_frames=self.skipped,
                        queue_capacity=1, work_s=self.last_work_s,
                        actual_fps=1/self.last_interval_s if self.last_interval_s else None,
                        frame_age_s=time.monotonic()-self.result.frame.started if self.result else None,
                        scope="hand_and_level_badges", error=str(self.error) if self.error else None)

    def close(self):
        self.stopped.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join()
