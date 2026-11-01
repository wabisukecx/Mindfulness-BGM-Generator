"""Sound of the quiet presets: a drone that drifts like incense smoke, spaces
of near-silence, and a rin (Japanese bowl bell) to open and close a sitting.

Everything here is a function of absolute time, so audio blocks join seamlessly.
"""
import math
import random

import numpy as np

from src.instruments_base import BaseInstrument


RIN_FREQ = 660.0  # Fourth harmonic of the drone's fifth (165 Hz), sixth of its root


class SmokeDrone:
    """Steady low drone whose timbre drifts slowly, like rising smoke.

    Pitch never changes. Each harmonic swells and thins on its own slow,
    unsynchronised cycle, and each voice above the root is detuned by a
    fraction of a hertz so coinciding harmonics beat slowly, as a rin does.
    The harmonics carry the low fundamental on small speakers that cannot
    reproduce it.
    """

    PARTIALS = (1.0, 0.6, 0.4, 0.25, 0.15, 0.08)
    MOVEMENT = (0.1, 0.35, 0.5, 0.6, 0.7, 0.7)  # How far each harmonic drifts
    DRIFT_PERIODS = (20.0, 90.0)                # Seconds
    DETUNE_HZ = 0.08
    GAIN = 0.7

    def __init__(self, freqs, sample_rate: int):
        self.sample_rate = sample_rate
        self.voices = []
        for index, freq in enumerate(freqs):
            drifts = [tuple(1 / random.uniform(*self.DRIFT_PERIODS) for _ in range(2))
                      + tuple(random.uniform(0, 2 * math.pi) for _ in range(2))
                      for _ in self.PARTIALS]
            self.voices.append((freq + self.DETUNE_HZ * index, drifts))

    def render(self, t: np.ndarray) -> np.ndarray:
        wave = np.zeros_like(t)
        for freq, drifts in self.voices:
            for n, (level, depth, (rate1, rate2, phase1, phase2)) in enumerate(
                    zip(self.PARTIALS, self.MOVEMENT, drifts), 1):
                if freq * n >= self.sample_rate / 2:
                    continue
                # Two incommensurate slow sines: a drift in [-1, 1] that never repeats exactly
                drift = 0.5 * (np.sin(2 * np.pi * rate1 * t + phase1)
                               + np.sin(2 * np.pi * rate2 * t + phase2))
                wave += level * (1 + depth * drift) * np.sin(2 * np.pi * freq * n * t)
        return wave * self.GAIN / len(self.voices)


def _raised_cosine(x: np.ndarray) -> np.ndarray:
    """Smooth 0-to-1 ramp for x in [0, 1]."""
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(x, 0, 1))


class ShoVoice:
    """Shō-inspired voice: sustained reed clusters in a high register that
    swell with the player's breath and change one pipe at a time.

    Inspired by the gagaku mouth organ's aitake clusters and its te-utsuri,
    in which the fingers move to the next cluster one by one; the clusters
    here are built from A B D E F# and are not the historical aitake.
    """

    A4 = 440.0
    # Semitone offsets from A4: stacked fourths and seconds. Neighbouring
    # clusters share most of their pipes, so each change is a gentle shift.
    CLUSTERS = ((0, 2, 7, 9, 12, 14), (2, 5, 7, 12, 14, 19), (0, 5, 7, 9, 12, 17),
                (5, 7, 12, 14, 17, 19), (7, 9, 12, 14, 19), (0, 2, 5, 7, 12))
    REED = (1.0, 0.35, 0.3, 0.15, 0.12, 0.06, 0.05)  # Free-reed harmonic levels
    HOLD = (25.0, 45.0)    # Seconds each cluster is held
    FADE = 2.0             # Seconds for one pipe to enter or leave
    STAGGER = (0.4, 0.9)   # Seconds between successive fingers
    DETUNE_CENTS = 2.0     # Bamboo pipes are never exactly in tune: a shimmer
    BREATH = 12.0          # Seconds per inhale or exhale
    GAIN = 0.1

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.pitches = sorted({p for cluster in self.CLUSTERS for p in cluster})
        self.freqs = {p: self.A4 * 2 ** ((p + random.uniform(-1, 1) * self.DETUNE_CENTS / 100) / 12)
                      for p in self.pitches}
        self.cluster = random.randrange(len(self.CLUSTERS))
        self.spans = {p: [] for p in self.pitches}  # [start, end] seconds a pipe sounds
        for p in self.CLUSTERS[self.cluster]:
            self.spans[p].append([-self.FADE, math.inf])
        self._next_change = random.uniform(*self.HOLD)
        self.breath_offset = random.uniform(0, 1)

    def _schedule_until(self, time: float):
        while self._next_change <= time:
            current = set(self.CLUSTERS[self.cluster])
            choices = [i for i in range(len(self.CLUSTERS)) if i != self.cluster]
            # Prefer clusters sharing the most pipes with the current one
            weights = [len(current & set(self.CLUSTERS[i])) ** 2 for i in choices]
            self.cluster = random.choices(choices, weights)[0]
            target = set(self.CLUSTERS[self.cluster])
            moves = [(p, False) for p in current - target] + [(p, True) for p in target - current]
            random.shuffle(moves)
            when = self._next_change
            for pitch, entering in moves:
                if entering:
                    self.spans[pitch].append([when, math.inf])
                else:
                    self.spans[pitch][-1][1] = when
                when += random.uniform(*self.STAGGER)
            self._next_change = when + random.uniform(*self.HOLD)

    def render(self, t: np.ndarray) -> np.ndarray:
        start_t, end_t = t.min(), t.max()
        self._schedule_until(end_t)
        wave = np.zeros_like(t)
        for pitch in self.pitches:
            spans = [s for s in self.spans[pitch] if s[1] + self.FADE >= start_t and s[0] <= end_t]
            if not spans:
                continue
            gain = np.zeros_like(t)
            for start, end in spans:
                rise = (t - start) / self.FADE
                fall = (end + self.FADE - t) / self.FADE if end < math.inf else np.ones_like(t)
                gain = np.maximum(gain, _raised_cosine(np.minimum(rise, fall)))
            freq = self.freqs[pitch]
            tone = np.zeros_like(t)
            for n, level in enumerate(self.REED, 1):
                if freq * n < self.sample_rate / 2:
                    tone += level * np.sin(2 * np.pi * freq * n * t)
            wave += gain * tone
        # One swell per breath, dipping gently where inhale turns to exhale;
        # a slow wobble keeps the breathing from sounding metronomic
        phase = t / self.BREATH + self.breath_offset + 0.15 * np.sin(2 * np.pi * t / 97)
        breath = 0.75 + 0.25 * np.sin(np.pi * phase) ** 2
        return wave * breath * self.GAIN


class BreathingSpace:
    """Every few minutes the drone recedes almost to silence, then returns.

    A recess fades down over FADE seconds, holds, and fades back up, with
    raised-cosine shapes, leaving a space in which to hear the stillness.
    """

    FIRST = (180.0, 300.0)  # Seconds before the first recess
    GAP = (240.0, 420.0)    # Seconds between one recess ending and the next
    FADE = 40.0
    HOLD = (20.0, 60.0)
    FLOOR = 0.12

    def __init__(self):
        self.recesses = []  # (start, end) in seconds
        self._next_start = random.uniform(*self.FIRST)

    def gain(self, t: np.ndarray) -> np.ndarray:
        while self._next_start <= t.max():
            end = self._next_start + 2 * self.FADE + random.uniform(*self.HOLD)
            self.recesses.append((self._next_start, end))
            self._next_start = end + random.uniform(*self.GAP)
        depth = np.zeros_like(t)
        for start, end in self.recesses:
            if end < t.min() or start > t.max():
                continue
            shape = np.clip(np.minimum(t - start, end - t) / self.FADE, 0, 1)
            depth = np.maximum(depth, 0.5 - 0.5 * np.cos(np.pi * shape))
        return 1 - (1 - self.FLOOR) * depth


class Rin(BaseInstrument):
    """Japanese rin: a struck bowl bell with a long, slowly beating ring.

    Each vibration mode of a real bowl is a close pair of frequencies, which
    beat; higher modes are fainter and die away sooner.
    """

    MIN_FREQ = 200
    MAX_FREQ = 2000
    # (frequency ratio, level, decay seconds, beat Hz within the mode)
    MODES = ((1.0, 1.0, 4.5, 0.8), (2.71, 0.5, 2.5, 1.7),
             (5.15, 0.25, 1.2, 3.0), (8.3, 0.12, 0.6, 4.0))
    LEVEL = 0.18
    QUIET = 3e-4  # Stop once the ring falls this low (about -70 dBFS)

    def _on_trigger(self, freq: float):
        self.position = 0

    def generate(self, frames: int) -> np.ndarray:
        if not self.is_playing:
            return np.zeros(frames)
        t = (np.arange(frames) + self.position) / self.sample_rate
        sound = np.zeros(frames)
        for ratio, level, decay, beat in self.MODES:
            freq = self.current_freq * ratio
            if freq + beat >= self.sample_rate / 2:
                continue
            # Unequal halves: the ring swells and fades rather than vanishing
            pair = (np.sin(2 * np.pi * (freq - beat / 2) * t)
                    + 0.6 * np.sin(2 * np.pi * (freq + beat / 2) * t)) / 1.6
            sound += level * pair * np.exp(-t / decay)
        sound *= (1 - np.exp(-t / 0.002)) * self.LEVEL  # Soft mallet attack
        self.position += frames
        if t[-1] > 1 and np.max(np.abs(sound)) < self.QUIET:
            self.is_playing = False
        return sound
