"""
Audio effects for Mindfulness BGM Generator
Enhanced with binaural processing and improved stereo field
"""

import numpy as np
from src.config import REVERB_TIME, REVERB_MIX, LIMITER_KNEE


class AudioEffects:
    """Audio effect processors with enhanced binaural capabilities"""
    
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        # Reverb buffer
        self.reverb_buffer = np.zeros(int(self.sample_rate * REVERB_TIME))
        self.reverb_index = 0
        
        # Delay buffers for binaural processing
        self.binaural_left_buffer = np.zeros(256)
        self.binaural_right_buffer = np.zeros(256)
        self.binaural_left_index = 0
        self.binaural_right_index = 0
    
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
    def apply_soft_limiting(signal: np.ndarray, threshold: float = 0.5, 
                        ceiling: float = 0.85, knee: float = 0.1) -> np.ndarray:
        """Improved soft limiter"""
        # Calculate RMS for dynamic range awareness
        rms = np.sqrt(np.mean(signal ** 2))
        
        # Detect signal peak
        peak = np.max(np.abs(signal))
        
        # Adjust gain if peak exceeds threshold
        if peak > threshold:
            # Calculate soft knee region
            if peak < threshold + knee:
                # Gentle compression in soft knee region
                ratio = 1 + (peak - threshold) / knee * 0.5
                gain = threshold / (threshold + (peak - threshold) / ratio)
            else:
                # Hard limiting
                gain = threshold / peak
            
            signal = signal * gain
        
        # Final tanh compression (softer)
        return ceiling * np.tanh(signal / ceiling * 0.9)
    
    @staticmethod
    def apply_stereo_enhancement(signal: np.ndarray, 
                                delay1: int = 3, delay2: int = 6,
                                mix1: float = 0.08, mix2: float = 0.04) -> np.ndarray:
        """Apply stereo enhancement with multiple delays"""
        sig_left = signal + np.roll(signal, delay1) * mix1 + np.roll(signal, delay2) * mix2
        sig_right = signal + np.roll(signal, -delay1) * mix1 + np.roll(signal, -delay2) * mix2
        return np.column_stack([sig_left, sig_right])
    
    def apply_binaural_enhancement(self, signal: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Subtle stereo field processing inspired by binaural beats"""
        # Initialize output left and right channels
        frames = len(signal)
        output = np.zeros((frames, 2), dtype=np.float32)
        
        # Slight frequency difference for left and right channels
        freq_shift_left = 1.0 - 0.0008  # Left channel is slightly lower (-0.08%)
        freq_shift_right = 1.0 + 0.0008  # Right channel is slightly higher (+0.08%)
        
        # Subtle phase difference (equivalent to 1-4ms delay)
        phase_shift_left = int(0.001 * self.sample_rate)  # 1ms
        phase_shift_right = int(0.0015 * self.sample_rate)  # 1.5ms
        
        # Copy signal for processing
        left_signal = np.copy(signal)
        right_signal = np.copy(signal)
        
        # Apply independent reverb lines for left and right
        # Add comb filter effect (part of flanging effect)
        for i in range(frames):
            # Left channel
            self.binaural_left_buffer[self.binaural_left_index] = left_signal[i]
            comb_idx = (self.binaural_left_index - phase_shift_left) % len(self.binaural_left_buffer)
            left_signal[i] += self.binaural_left_buffer[comb_idx] * 0.06
            self.binaural_left_index = (self.binaural_left_index + 1) % len(self.binaural_left_buffer)
            
            # Right channel
            self.binaural_right_buffer[self.binaural_right_index] = right_signal[i]
            comb_idx = (self.binaural_right_index - phase_shift_right) % len(self.binaural_right_buffer)
            right_signal[i] += self.binaural_right_buffer[comb_idx] * 0.06
            self.binaural_right_index = (self.binaural_right_index + 1) % len(self.binaural_right_buffer)
        
        # Generate stereo field
        for i in range(frames):
            # Apply subtle frequency shift (mix current and slightly previous samples)
            t_pos = i / self.sample_rate
            
            # Left channel - apply phase modulation
            output[i, 0] = left_signal[i] * 0.9 + signal[i] * 0.1
            
            # Right channel - apply phase modulation
            output[i, 1] = right_signal[i] * 0.9 + signal[i] * 0.1
        
        # Normalize to maintain left-right balance
        max_amp = np.max(np.abs(output))
        if max_amp > 0:
            output /= max_amp
            output *= np.max(np.abs(signal))
        
        return output
    
    @staticmethod
    def apply_breathing_modulation(signal: np.ndarray, t: np.ndarray, 
                                  breath_phase: float, breath_cycle: float = 0.2) -> np.ndarray:
        """Apply breathing rhythm modulation"""
        breath_lfo = 0.97 + 0.03 * np.sin(2 * np.pi * breath_cycle * (t[0] * breath_cycle + breath_phase))
        return signal * breath_lfo