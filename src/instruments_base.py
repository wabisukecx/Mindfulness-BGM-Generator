"""
Base classes and configurations for meditation instruments
"""

import numpy as np
from abc import ABC, abstractmethod
import math


class InstrumentConfig:
    """Configuration for a percussion instrument"""
    
    def __init__(self, min_interval: float, max_interval: float, enabled: bool = True):
        if (not math.isfinite(min_interval) or not math.isfinite(max_interval)
                or min_interval < 0 or max_interval < min_interval):
            raise ValueError("Intervals must be finite and satisfy 0 <= min <= max")
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.enabled = bool(enabled and min_interval > 0)
    
    def __repr__(self):
        return f"InstrumentConfig(min={self.min_interval}, max={self.max_interval}, enabled={self.enabled})"


class BaseInstrument(ABC):
    """Base class for all meditation instruments"""
    
    MIN_FREQ = 20   # Default minimum frequency
    MAX_FREQ = 20000  # Default maximum frequency
    
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.is_playing = False
        self.phase = 0
        self.start_time = 0
        self.current_freq = 0
    
    def is_freq_valid(self, freq: float) -> bool:
        """Check if frequency is within valid range"""
        return self.MIN_FREQ <= freq <= self.MAX_FREQ

    def _partial(self, freq: float, t: np.ndarray, phase: float = 0.0) -> np.ndarray:
        """Sine partial, silent at or above Nyquist so it cannot alias."""
        if freq >= self.sample_rate / 2:
            return np.zeros_like(t)
        return np.sin(2 * np.pi * freq * t + phase)

    @staticmethod
    def _bowl_fm_phase(freq: float, t: np.ndarray, rubbing_phase: float) -> np.ndarray:
        """Integrated phase of a slow, fixed-depth vibrato around freq.

        Multiplying a modulated frequency by t would make the deviation grow
        with time; integrating keeps it at freq * depth for the whole note.
        """
        depth1, rate1 = 0.0001, 0.2
        depth2, rate2 = 0.00005, 0.5
        return (2 * np.pi * freq * t
                - freq * depth1 / rate1 * np.cos(2 * np.pi * rate1 * t)
                - freq * depth2 / rate2 * np.cos(2 * np.pi * rate2 * t + rubbing_phase))

    def trigger(self, freq: float):
        """Trigger the instrument"""
        if not self.is_freq_valid(freq):
            return
        
        self.is_playing = True
        self.phase = 0
        self.start_time = 0
        self.current_freq = freq
        self._on_trigger(freq)
    
    @abstractmethod
    def _on_trigger(self, freq: float):
        """Called when instrument is triggered"""
        pass
    
    @abstractmethod
    def generate(self, frames: int) -> np.ndarray:
        """Generate audio samples"""
        pass
    
    def stop(self):
        """Stop the instrument"""
        self.is_playing = False


class VoicePool:
    """Lets a percussion instrument keep ringing while it is struck again.

    Each strike gets its own voice. When more than max_voices ring at once,
    the oldest is faded out over RELEASE_SECONDS instead of being cut off,
    which would click.
    """

    RELEASE_SECONDS = 0.05

    def __init__(self, instrument_cls, sample_rate: int, max_voices: int = 3):
        self.instrument_cls = instrument_cls
        self.sample_rate = sample_rate
        self.max_voices = max_voices
        self.voices = []  # [instrument, release gain or None while ringing]

    @property
    def is_playing(self) -> bool:
        return bool(self.voices)

    def trigger(self, freq: float):
        voice = self.instrument_cls(self.sample_rate)
        voice.trigger(freq)
        if not voice.is_playing:  # Out of the instrument's range
            return
        ringing = [entry for entry in self.voices if entry[1] is None]
        if len(ringing) >= self.max_voices:
            ringing[0][1] = 1.0
        self.voices.append([voice, None])

    def generate(self, frames: int) -> np.ndarray:
        output = np.zeros(frames)
        step = 1.0 / (self.RELEASE_SECONDS * self.sample_rate)
        for entry in self.voices:
            voice, gain = entry
            sound = voice.generate(frames)
            if gain is not None:
                ramp = np.clip(gain - step * np.arange(1, frames + 1), 0.0, 1.0)
                sound = sound * ramp
                entry[1] = ramp[-1]
                if ramp[-1] <= 0:
                    voice.stop()
            output += sound
        self.voices = [entry for entry in self.voices if entry[0].is_playing]
        return output

    def stop(self):
        for voice, _ in self.voices:
            voice.stop()
        self.voices.clear()