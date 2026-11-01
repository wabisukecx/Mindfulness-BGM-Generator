"""Sample-based session timing and smooth entry/exit envelopes."""
import numpy as np


class SessionEnvelope:
    def __init__(self, sample_rate, duration_seconds=0, fade_in=8, fade_out=20):
        self.sample_rate = sample_rate
        self.duration = max(0, round(duration_seconds * sample_rate))
        self.fade_in = max(1, round(fade_in * sample_rate))
        self.fade_out = max(1, round(fade_out * sample_rate))
        self.position = 0

    @property
    def finished(self):
        return self.duration > 0 and self.position >= self.duration

    @property
    def remaining_seconds(self):
        if not self.duration:
            return None
        return max(0, self.duration - self.position) / self.sample_rate

    def gains(self, frames):
        positions = self.position + np.arange(frames)
        gain = np.sin(np.clip(positions / self.fade_in, 0, 1) * np.pi/2) ** 2
        if self.duration:
            tail = np.clip((self.duration - positions) / self.fade_out, 0, 1)
            gain *= np.sin(tail * np.pi/2) ** 2
            gain[positions >= self.duration] = 0
        self.position += frames
        return gain
