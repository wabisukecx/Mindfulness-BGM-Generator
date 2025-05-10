"""
Audio effects for Mindfulness BGM Generator
"""

import numpy as np
from config import REVERB_TIME, REVERB_MIX


class AudioEffects:
    """Audio effect processors"""
    
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        # Reverb buffer
        self.reverb_buffer = np.zeros(int(self.sample_rate * REVERB_TIME))
        self.reverb_index = 0
    
    def apply_reverb(self, signal: np.ndarray) -> np.ndarray:
        """Enhanced reverb effect"""
        output = np.copy(signal)
        
        for i in range(len(signal)):
            # Read from reverb buffer - Enhanced mix
            reverb_sample = self.reverb_buffer[self.reverb_index] * REVERB_MIX
            
            # Add to output
            output[i] += reverb_sample
            
            # Add current sample to buffer
            self.reverb_buffer[self.reverb_index] = signal[i]
            
            # Advance index
            self.reverb_index = (self.reverb_index + 1) % len(self.reverb_buffer)
        
        return output
    
    @staticmethod
    def apply_soft_limiting(signal: np.ndarray, threshold: float = 0.7, ceiling: float = 0.95) -> np.ndarray:
        """Apply soft limiting to prevent clipping"""
        return np.tanh(signal * threshold) * ceiling
    
    @staticmethod
    def apply_stereo_enhancement(signal: np.ndarray, 
                                delay1: int = 3, delay2: int = 6,
                                mix1: float = 0.08, mix2: float = 0.04) -> np.ndarray:
        """Apply stereo enhancement with multiple delays"""
        sig_left = signal + np.roll(signal, delay1) * mix1 + np.roll(signal, delay2) * mix2
        sig_right = signal + np.roll(signal, -delay1) * mix1 + np.roll(signal, -delay2) * mix2
        return np.column_stack([sig_left, sig_right])
    
    @staticmethod
    def apply_breathing_modulation(signal: np.ndarray, t: np.ndarray, 
                                  breath_phase: float, breath_cycle: float = 0.2) -> np.ndarray:
        """Apply breathing rhythm modulation"""
        breath_lfo = 0.97 + 0.03 * np.sin(2 * np.pi * breath_cycle * (t[0] * breath_cycle + breath_phase))
        return signal * breath_lfo