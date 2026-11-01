"""
Audio effects for Mindfulness BGM Generator
Enhanced with binaural processing and improved stereo field
"""

import numpy as np
from src.config import REVERB_TIME, REVERB_MIX, LIMITER_RELEASE


class AudioEffects:
    """Audio effect processors with enhanced binaural capabilities"""
    
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        # Reverb buffer
        self.reverb_buffer = np.zeros(int(self.sample_rate * REVERB_TIME))
        self.reverb_index = 0
        
        self.stereo_history = np.zeros(0)
        self.limiter_gain = 1.0

    def apply_reverb(self, signal: np.ndarray) -> np.ndarray:
        """Enhanced reverb effect"""
        # Retain separate delay lines for each channel.
        expected_shape = (int(self.sample_rate * REVERB_TIME),) + signal.shape[1:]
        if self.reverb_buffer.shape != expected_shape:
            self.reverb_buffer = np.zeros(expected_shape)
            self.reverb_index = 0
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
    
    def apply_soft_limiting(self, signal: np.ndarray, threshold: float = 0.5,
                            ceiling: float = 0.85, knee: float = 0.1) -> np.ndarray:
        """Soft limiter whose gain moves smoothly instead of per block.

        Gain reduction is reached within the block (fast attack) and released
        at most at LIMITER_RELEASE seconds per unit of gain; the final tanh
        stage catches any overshoot while the gain is still ramping down.
        """
        # Detect signal peak
        peak = np.max(np.abs(signal))

        # Target gain if peak exceeds threshold
        target = 1.0
        if peak > threshold:
            # Calculate soft knee region
            if peak < threshold + knee:
                # Gentle compression in soft knee region
                ratio = 1 + (peak - threshold) / knee * 0.5
                target = threshold / (threshold + (peak - threshold) / ratio)
            else:
                # Hard limiting
                target = threshold / peak

        start = self.limiter_gain
        if target < start:
            end = target
        else:
            end = min(target, start + len(signal) / (LIMITER_RELEASE * self.sample_rate))
        self.limiter_gain = end
        gain = np.linspace(start, end, len(signal) + 1)[1:]
        signal = signal * gain.reshape((-1,) + (1,) * (signal.ndim - 1))

        # Final tanh compression (softer)
        return ceiling * np.tanh(signal / ceiling * 0.9)
    
    def apply_stereo_enhancement(self, signal: np.ndarray,
                                 delay1: int = 3, delay2: int = 6,
                                 mix1: float = 0.08, mix2: float = 0.04) -> np.ndarray:
        """Causal stereo delays, preserving history across audio blocks."""
        history_size = max(delay1, delay2)
        if len(self.stereo_history) != history_size:
            self.stereo_history = np.zeros(history_size)
        source = np.concatenate([self.stereo_history, signal])
        delayed1 = source[history_size-delay1:history_size-delay1+len(signal)]
        delayed2 = source[history_size-delay2:history_size-delay2+len(signal)]
        if history_size:
            self.stereo_history = source[-history_size:].copy()
        left = signal + delayed1 * mix1 + delayed2 * mix2
        right = signal + delayed2 * mix1 + delayed1 * mix2
        return np.column_stack([left, right])

    @staticmethod
    def apply_breathing_modulation(signal: np.ndarray, t: np.ndarray, 
                                  breath_phase: float, breath_cycle: float = 0.2) -> np.ndarray:
        """Apply breathing rhythm modulation"""
        breath_lfo = 0.97 + 0.03 * np.sin(2 * np.pi * breath_cycle * (t[0] * breath_cycle + breath_phase))
        return signal * breath_lfo