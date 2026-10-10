"""Detect stationary unknown screens and pace reversible navigation taps."""
from PIL import Image, ImageChops, ImageStat


class ScreenRecovery:
    def __init__(self, config):
        self.enabled = bool(config.get("stuck_screen_recovery_enabled", True))
        self.timeout = max(1., float(config.get("stuck_screen_timeout_s", 20.)))
        self.interval = max(.2, float(config.get("stuck_screen_tap_interval_s", 1.)))
        self.threshold = max(0., float(config.get("stuck_screen_change_threshold", 2.)))
        self.reference = None
        self.since = None
        self.last_tap = None
        self.active = False

    def observe(self, image, recognized, now):
        if recognized or not self.enabled:
            self.reference = None
            self.since = self.last_tap = None
            self.active = False
            return False
        sample = image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
        if not self.active:
            changed = self.reference is None or max(ImageStat.Stat(
                ImageChops.difference(sample, self.reference)).mean) > self.threshold
            if changed:
                self.reference = sample
                self.since = now
            self.active = self.since is not None and now - self.since >= self.timeout
        return self.active and (self.last_tap is None or now - self.last_tap >= self.interval)

    def tapped(self, now):
        self.last_tap = now
