"""Continuous real detections on newest captures, independent of planning."""
from collections import deque
from threading import Event, Lock, Thread
import time

from .battle_perception import LaneThreat


class YoloStream:
    def __init__(self, frames, detector_factory, catalog, target_fps=20):
        self.frames, self.factory, self.catalog = frames, detector_factory, catalog
        self.period = 1 / max(10., min(60., float(target_fps)))
        self.stopped, self.lock = Event(), Lock()
        self.result, self.error = None, ''
        self.processed = self.skipped = 0
        self.work_s = None
        self.finished_times = deque(maxlen=120)
        self.thread = Thread(target=self._run, name='battle-yolo', daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        try:
            # This model is owned exclusively by this worker. The policy's model
            # and observation lists must never be mutated while it is planning.
            detector = self.factory()
            if not detector.available:
                raise RuntimeError(detector.error or 'YOLO model unavailable')
            sequence = 0
            empty = {lane: LaneThreat(lane, 0, 0, 0, 'none', ()) for lane in ('left', 'right')}
            while not self.stopped.is_set():
                try:
                    frame = self.frames.get(sequence=sequence, timeout=.2)
                except TimeoutError:
                    continue
                if self.stopped.is_set():
                    break
                started = time.monotonic()
                # YOLO outputs identities, boxes and team evidence. Health-bar
                # reading remains in the decision path, rather than serializing
                # several extra CPU scans ahead of every preview publication.
                detector.detect(frame.image, empty, include_health=False)
                finished = time.monotonic()
                if not detector.last_detection_succeeded:
                    raise RuntimeError(detector.error or 'YOLO inference failed')
                observations = [dict(o) for o in detector.observed_enemies + detector.observed_allies]
                for observation in observations:
                    card = self.catalog.get(observation.get('card_id')) if self.catalog else None
                    if card:
                        observation['name_zh'] = card.name_zh or card.name_en
                with self.lock:
                    self.processed += 1
                    self.skipped += max(0, frame.sequence - sequence - 1)
                    self.work_s = finished - started
                    self.finished_times.append(finished)
                    packet = dict(revision=self.processed, capture_sequence=frame.sequence,
                        at=frame.started, finished_at=finished, detections=observations,
                        model=detector.status(), inference_ms=round(self.work_s * 1000, 3),
                        recognition_fps=self._fps(finished), source='continuous_yolo')
                    self.result = (frame.image, packet)
                sequence = frame.sequence
                self.stopped.wait(max(0., self.period - (time.monotonic() - started)))
        except Exception as exc:
            with self.lock:
                self.error = str(exc)

    def _fps(self, now):
        recent = [t for t in self.finished_times if now - t <= 5]
        return (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 else 0.

    def latest(self):
        with self.lock:
            # Do not present an old successful frame as a continuing live result.
            if self.error or self.result is None or time.monotonic() - self.result[1]['finished_at'] > .5:
                return None
            return self.result

    def status(self):
        with self.lock:
            now = time.monotonic()
            packet = self.result[1] if self.result else {}
            active = bool(packet and now - packet['finished_at'] <= .5 and not self.error)
            return dict(enabled=True, processed=self.processed, skipped_capture_frames=self.skipped,
                queue_capacity=1, actual_fps=self._fps(now) if active else 0.,
                inference_ms=None if self.work_s is None else round(self.work_s * 1000, 3),
                frame_age_s=now - packet['at'] if packet else None,
                model=packet.get('model'), error=self.error, scope='actual_model_and_unit_postprocessing')

    def close(self):
        self.stopped.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=22)
