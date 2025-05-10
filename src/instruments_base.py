"""
Base classes and configurations for meditation instruments
"""

import numpy as np
from abc import ABC, abstractmethod


class InstrumentConfig:
    """Configuration for a percussion instrument"""
    
    def __init__(self, min_interval: float, max_interval: float, enabled: bool = True):
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.enabled = enabled
    
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