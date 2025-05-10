"""
Meditation instrument implementations for Mindfulness BGM Generator
"""

import numpy as np
from src.instruments_base import BaseInstrument, InstrumentConfig

# Re-export InstrumentConfig for convenience
__all__ = ['SlitDrum', 'Handpan', 'CrystalSingingBowl', 'TibetanBell', 'InstrumentConfig']


class SlitDrum(BaseInstrument):
    """Slit drum class"""
    
    MIN_FREQ = 100  # Low frequency range
    MAX_FREQ = 500  # Up to mid frequency range
    
    def _on_trigger(self, freq: float):
        """Slit drum specific trigger behavior"""
        self.drum_phase = 0
        self.drum_start_time = 0
        
    def generate(self, frames: int) -> np.ndarray:
        """Generate slit drum sound"""
        if not self.is_playing:
            return np.zeros(frames)
            
        t = (np.arange(frames) + self.drum_phase) / self.sample_rate
        
        # Fundamental tone
        fundamental = np.sin(2 * np.pi * self.current_freq * t)
        # Add harmonics for depth
        second = np.sin(2 * np.pi * self.current_freq * 2 * t) * 0.2
        third = np.sin(2 * np.pi * self.current_freq * 3 * t) * 0.08
        fourth = np.sin(2 * np.pi * self.current_freq * 4 * t) * 0.04
        
        # Envelope (representing wooden resonance)
        envelope = np.exp(-3 * t) * (1 + 0.5 * np.exp(-10 * t))
        
        # Synthesis
        drum_sound = (fundamental + second + third + fourth) * envelope * 0.35
        
        self.drum_phase += frames
        self.drum_start_time += frames / self.sample_rate
        
        # Stop after 3 seconds
        if self.drum_start_time > 3.0:
            self.is_playing = False
            
        return drum_sound


class Handpan(BaseInstrument):
    """Handpan class"""
    
    MIN_FREQ = 147  # D3
    MAX_FREQ = 587  # D5
    
    def _on_trigger(self, freq: float):
        """Handpan specific trigger behavior"""
        self.pan_phase = 0
        self.pan_start_time = 0
        
    def generate(self, frames: int) -> np.ndarray:
        """Generate handpan sound"""
        if not self.is_playing:
            return np.zeros(frames)
            
        t = (np.arange(frames) + self.pan_phase) / self.sample_rate
        
        # Fundamental with characteristic pitch bend
        pitch_bend = np.exp(-15 * t) * 0.005 + 1  # Slight pitch drop at impact
        fundamental = np.sin(2 * np.pi * self.current_freq * t * pitch_bend)
        
        # Harmonic series with slight inharmonicity
        second = np.sin(2 * np.pi * self.current_freq * 2.02 * t) * 0.3
        third = np.sin(2 * np.pi * self.current_freq * 2.99 * t) * 0.15
        fourth = np.sin(2 * np.pi * self.current_freq * 3.98 * t) * 0.08
        fifth = np.sin(2 * np.pi * self.current_freq * 4.05 * t) * 0.05
        sixth = np.sin(2 * np.pi * self.current_freq * 6.02 * t) * 0.03
        
        # Metallic resonance with modulation
        resonance_freq = self.current_freq * 1.51
        resonance_mod = 1 + 0.1 * np.exp(-5 * t)  # Modulation decreases over time
        metallic = np.sin(2 * np.pi * resonance_freq * t * resonance_mod + np.pi/4) * 0.2
        
        # Add sympathetic resonance (characteristic of handpan)
        sympathetic = np.sin(2 * np.pi * self.current_freq * 5.1 * t) * 0.05 * np.exp(-2 * t)
        
        # Complex envelope with characteristic handpan decay
        strike = np.exp(-30 * t)    # Sharp initial strike
        bloom = 1 - np.exp(-5 * t)  # Sound "blooms" after strike
        decay = np.exp(-2.5 * t)    # Main decay
        sustain = np.exp(-0.8 * t)  # Sustain
        
        envelope = strike * 0.3 + bloom * decay * 0.5 + sustain * 0.2
        
        # Add subtle beating effect
        beating = 1 + 0.02 * np.sin(2 * np.pi * 2 * t) * np.exp(-3 * t)
        
        # Synthesis
        pan_sound = (fundamental + second + third + fourth + fifth + sixth + metallic + sympathetic)
        pan_sound *= envelope * beating * 0.28
        
        self.pan_phase += frames
        self.pan_start_time += frames / self.sample_rate
        
        # Stop after 3.5 seconds or when very quiet
        if self.pan_start_time > 3.5 or np.max(np.abs(pan_sound)) < 0.0005:
            self.is_playing = False
            
        return pan_sound


class CrystalSingingBowl(BaseInstrument):
    """Crystal singing bowl class"""
    
    MIN_FREQ = 200  # From mid range
    MAX_FREQ = 2000  # To high range
    
    def _on_trigger(self, freq: float):
        """Crystal bowl specific trigger behavior"""
        self.crystal_phase = 0
        self.crystal_start_time = 0
        self.rubbing_phase = np.random.uniform(0, 2*np.pi)
        
    def generate(self, frames: int) -> np.ndarray:
        """Generate crystal singing bowl sound"""
        if not self.is_playing:
            return np.zeros(frames)
            
        t = (np.arange(frames) + self.crystal_phase) / self.sample_rate
        
        # Centered frequency modulation - very subtle
        fm_depth1 = 0.0001
        fm_depth2 = 0.00005
        fm_freq1 = 0.2
        fm_freq2 = 0.5
        
        # Centered modulation around base frequency
        freq_mod1 = np.sin(2 * np.pi * fm_freq1 * t) * fm_depth1
        freq_mod2 = np.sin(2 * np.pi * fm_freq2 * t + self.rubbing_phase) * fm_depth2
        freq_final = self.current_freq * (1 + freq_mod1 + freq_mod2)
        
        # Pure fundamental
        fundamental = np.sin(2 * np.pi * freq_final * t)
        
        # Very subtle harmonics
        second = np.sin(2 * np.pi * self.current_freq * 2 * t + np.pi/4) * 0.02
        third = np.sin(2 * np.pi * self.current_freq * 3 * t + np.pi/3) * 0.01
        fourth = np.sin(2 * np.pi * self.current_freq * 4 * t + np.pi/5) * 0.008
        fifth = np.sin(2 * np.pi * self.current_freq * 5 * t + np.pi/6) * 0.005
        
        # Ring modulation for crystalline quality
        ring_freq = self.current_freq * 7.1
        ring = np.sin(2 * np.pi * ring_freq * t) * 0.005
        
        # Centered amplitude modulation
        am_depth1 = 0.005
        am_depth2 = 0.002
        am_freq1 = 3.7
        am_freq2 = 5.3
        amplitude_mod = 1 + am_depth1 * np.sin(2 * np.pi * am_freq1 * t)
        amplitude_mod += am_depth2 * np.sin(2 * np.pi * am_freq2 * t + np.pi/2)
        
        # Time-based envelope
        t_fade = t + self.crystal_phase / self.sample_rate
        
        rub_attack = 1 - np.exp(-2 * t_fade)
        strike_attack = np.exp(-20 * t_fade)
        decay = np.exp(-0.4 * t_fade)
        
        envelope = rub_attack * 0.7 + strike_attack * 0.3
        envelope *= decay
        
        # Ensure complete fade out
        final_fade = np.exp(-0.2 * t_fade)
        envelope *= final_fade
        
        # Very subtle noise
        noise = np.random.randn(frames) * 0.0005 * np.exp(-5 * t_fade)
        
        # Synthesis
        crystal_sound = (fundamental + second + third + fourth + fifth + ring) * envelope * amplitude_mod * 0.35
        crystal_sound += noise
        
        self.crystal_phase += frames
        self.crystal_start_time += frames / self.sample_rate
        
        # Stop when very quiet or after time limit
        if self.crystal_start_time > 12.0 or np.max(np.abs(crystal_sound)) < 0.0005:
            self.is_playing = False
            
        return crystal_sound


class TibetanBell(BaseInstrument):
    """Tibetan bell (Tingsha) class"""
    
    MIN_FREQ = 1000  # High range
    MAX_FREQ = 8000  # To ultra-high range
    
    def __init__(self, sample_rate: int):
        super().__init__(sample_rate)
        self.current_freq = 2000  # Default frequency
    
    def _on_trigger(self, freq: float):
        """Tibetan bell specific trigger behavior"""
        self.bell_phase = 0
        self.bell_start_time = 0
        
    def generate(self, frames: int) -> np.ndarray:
        """Generate bell sound"""
        if not self.is_playing:
            return np.zeros(frames)
            
        t = (np.arange(frames) + self.bell_phase) / self.sample_rate
        
        # Multiple frequencies with inharmonic ratios
        f1 = self.current_freq
        f2 = self.current_freq * 1.02  # Very slight dissonance
        f3 = self.current_freq * 2.76  # Inharmonic partial
        f4 = self.current_freq * 5.43  # High inharmonic partial
        f5 = self.current_freq * 3.98  # Additional inharmonic
        f6 = self.current_freq * 6.51  # Higher partial
        
        # Bell sound components
        wave1 = np.sin(2 * np.pi * f1 * t)
        wave2 = np.sin(2 * np.pi * f2 * t) * 0.8
        wave3 = np.sin(2 * np.pi * f3 * t) * 0.3
        wave4 = np.sin(2 * np.pi * f4 * t) * 0.1
        wave5 = np.sin(2 * np.pi * f5 * t) * 0.05
        wave6 = np.sin(2 * np.pi * f6 * t) * 0.03
        
        # Time-based envelope
        t_fade = t + self.bell_phase / self.sample_rate
        
        attack = np.exp(-100 * t_fade)
        decay1 = np.exp(-5 * t_fade)
        decay2 = np.exp(-1.5 * t_fade)
        
        envelope = attack * 0.5 + decay1 * 0.3 + decay2 * 0.2
        
        # Ensure complete fade out
        final_fade = np.exp(-0.8 * t_fade)
        envelope *= final_fade
        
        # Centered amplitude modulation
        am_depth = 0.015 * np.exp(-2 * t_fade)  # Decreasing shimmer
        am_freq = 8
        amplitude_mod = 1 + am_depth * np.sin(2 * np.pi * am_freq * t)
        
        # Synthesis
        bell_sound = (wave1 + wave2 + wave3 + wave4 + wave5 + wave6) * envelope * amplitude_mod * 0.2
        
        self.bell_phase += frames
        self.bell_start_time += frames / self.sample_rate
        
        # Stop when very quiet or after time limit
        if self.bell_start_time > 4.0 or np.max(np.abs(bell_sound)) < 0.001:
            self.is_playing = False
            
        return bell_sound