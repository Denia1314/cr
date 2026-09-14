"""Temporal evidence for battle entry and reversible UI-loss handling."""
from dataclasses import dataclass


@dataclass
class BattleTransitionGuard:
    start_frames: int = 2
    stable_s: float = .7
    grace_s: float = 10.
    end_frames: int = 4
    absent_s: float = 3.
    present_frames: int = 0
    absent_frames: int = 0
    present_since: float | None = None
    absent_since: float | None = None
    started_at: float | None = None
    now: float = 0.

    @classmethod
    def from_config(cls, config):
        return cls(max(2, int(config.get('battle_start_consecutive_frames', 2))),
                   max(.1, float(config.get('battle_start_stable_s', .7))),
                   max(0., float(config.get('battle_start_grace_s', 10.))),
                   max(2, int(config.get('battle_end_consecutive_frames', 4))),
                   max(.5, float(config.get('battle_end_absent_s', 3.))))

    def reset(self):
        self.present_frames = self.absent_frames = 0
        self.present_since = self.absent_since = self.started_at = None

    def start(self, now):
        self.started_at = now

    def observe(self, present, now):
        self.now = now
        if present:
            if not self.present_frames:
                self.present_since = now
            self.present_frames += 1
            self.absent_frames = 0
            self.absent_since = None
        else:
            if not self.absent_frames:
                self.absent_since = now
            self.absent_frames += 1
            self.present_frames = 0
            self.present_since = None

    @property
    def entry_ready(self):
        return (self.present_frames >= self.start_frames and
                self.present_since is not None and
                self.now - self.present_since >= self.stable_s)

    @property
    def end_pending(self):
        return (self.started_at is not None and
                self.now - self.started_at >= self.grace_s and
                self.absent_frames >= self.end_frames and
                self.absent_since is not None and
                self.now - self.absent_since >= self.absent_s)
