"""Single producer, latest-only screenshots with explicit capture timestamps."""
from dataclasses import dataclass
from threading import Condition, Event, Thread
import time


@dataclass(frozen=True)
class Frame:
    image: object
    started: float
    finished: float
    sequence: int


class LatestFrameStream:
    def __init__(self, capture, interval=.05):
        self.capture, self.interval = capture, max(.005, interval)
        self.condition, self.stopped = Condition(), Event()
        self.frame = None
        self.error = None
        self.produced = 0
        self.last_start_interval = None
        self.thread = Thread(target=self._run, name='battle-frame-capture', daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        while not self.stopped.is_set():
            started = time.monotonic()
            try:
                image = self.capture()
            except Exception as exc:
                with self.condition:
                    self.error = exc
                    self.condition.notify_all()
                return
            with self.condition:
                if self.frame is not None:
                    self.last_start_interval = started-self.frame.started
                self.produced += 1
                self.frame = Frame(image, started, time.monotonic(), self.produced)
                self.condition.notify_all()
            # Start-to-start target period, not an extra delay after capture.
            self.stopped.wait(max(0.,self.interval-(time.monotonic()-started)))

    def latest(self):
        with self.condition:
            return self.frame

    def get(self, *, sequence=0, after=0., timeout=8.):
        deadline = time.monotonic()+timeout
        with self.condition:
            while True:
                if self.error is not None:
                    raise self.error
                f = self.frame
                if f and f.sequence > sequence and f.started >= after:
                    return f
                if self.stopped.is_set():
                    raise RuntimeError('frame capture stopped')
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('fresh frame timeout')
                self.condition.wait(min(.1,remaining))

    def close(self):
        self.stopped.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=22)

    def status(self):
        f=self.latest()
        return dict(produced=self.produced,queue_capacity=1,
                    target_interval_s=self.interval,
                    actual_interval_s=self.last_start_interval,
                    actual_fps=None if not self.last_start_interval else 1/self.last_start_interval,
                    capture_s=None if f is None else f.finished-f.started,
                    frame_age_s=None if f is None else time.monotonic()-f.started,
                    error=None if self.error is None else str(self.error))
