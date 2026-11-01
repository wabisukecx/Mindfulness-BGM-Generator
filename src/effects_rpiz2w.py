"""
Audio effects for Mindfulness BGM Generator - Raspberry Pi Zero 2 W Version
Simplified effects for lower computational requirements
"""

import numpy as np
from src.config_rpiz2w import REVERB_TIME, REVERB_MIX


class AudioEffectsRPiZ2W:
    """Simplified audio effect processors for RPi Zero 2 W"""
    
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        # Reverb buffer (smaller for RPi Zero 2 W)
        self.reverb_buffer = np.zeros(int(self.sample_rate * REVERB_TIME))
        self.reverb_index = 0
        self.stereo_history = np.zeros(0)

    def apply_reverb(self, signal: np.ndarray) -> np.ndarray:
        """Simple reverb effect"""
        output = np.copy(signal)
        
        for i in range(len(signal)):
            reverb_sample = self.reverb_buffer[self.reverb_index] * REVERB_MIX
            output[i] += reverb_sample
            self.reverb_buffer[self.reverb_index] = signal[i]
            self.reverb_index = (self.reverb_index + 1) % len(self.reverb_buffer)
        
        return output
    
    @staticmethod
    def apply_soft_limiting(signal: np.ndarray) -> np.ndarray:
        """Apply soft limiting to prevent clipping"""
        return np.tanh(signal * 0.8) * 0.9
    
    def create_stereo(self, signal: np.ndarray, delay: int = 2, mix: float = 0.05) -> np.ndarray:
        """Simple stereo effect with delay history kept across blocks.

        Left hears a slightly delayed copy; right hears the direct sound
        delayed by the same amount with an undelayed copy, which keeps the
        original left/right timing without reading future samples.
        """
        if len(self.stereo_history) != delay:
            self.stereo_history = np.zeros(delay)
        source = np.concatenate([self.stereo_history, signal])
        delayed = source[:len(signal)]
        if delay:
            self.stereo_history = source[-delay:].copy()
        sig_left = signal + delayed * mix
        sig_right = delayed + signal * mix
        return np.column_stack([sig_left, sig_right])