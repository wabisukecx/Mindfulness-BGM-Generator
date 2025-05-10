"""
Ambient sound generation for Mindfulness BGM Generator - Raspberry Pi Zero 2 W Version
Simplified for lower computational requirements
"""

import numpy as np


class AmbientSoundGeneratorRPiZ2W:
    """Simplified ambient nature sounds for RPi Zero 2 W"""
    
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.ocean_phase = np.random.uniform(0, 2*np.pi)
        self.ocean_variation = np.random.uniform(0.8, 1.2)
        self.noise_state = 0.0
    
    def _generate_ocean_waves(self, t: np.ndarray, frames: int) -> np.ndarray:
        """Generate cleaner ocean wave sounds with less noise (simplified)"""
        t_ocean = t + self.ocean_phase
        var = self.ocean_variation
        
        # Main wave rhythm (fewer components for RPi Zero 2 W)
        wave1 = np.sin(2 * np.pi * 0.08 * var * t_ocean) * 0.8
        wave2 = np.sin(2 * np.pi * 0.06 * var * t_ocean + np.pi/3) * 0.7
        wave3 = np.sin(2 * np.pi * 0.04 * var * t_ocean + np.pi/6) * 0.6
        wave4 = np.sin(2 * np.pi * 0.03 * var * t_ocean + np.pi/4) * 0.5
        
        # Wave swells (smoother)
        swell = 0.6 + 0.4 * np.sin(2 * np.pi * 0.02 * var * t_ocean)
        main_waves = (wave1 + wave2 + wave3 + wave4) * swell
        
        # Breaking wave sounds (greatly reduced noise)
        breaking_trigger = np.sin(2 * np.pi * 0.05 * var * t_ocean)
        breaking_envelope = np.maximum(0, breaking_trigger) ** 2
        
        # Minimal noise component
        breaking_noise = np.random.randn(frames) * 0.05
        # Light low-pass filter
        for i in range(1, frames):
            breaking_noise[i] = 0.7 * breaking_noise[i-1] + 0.3 * breaking_noise[i]
        
        breaking_waves = breaking_noise * breaking_envelope * 0.3
        
        # Deep wave sounds (sine wave based)
        deep_waves = np.sin(2 * np.pi * 0.015 * t_ocean) * 0.5
        deep_waves += np.sin(2 * np.pi * 0.01 * t_ocean + np.pi/2) * 0.4
        deep_waves += np.sin(2 * np.pi * 0.007 * t_ocean + np.pi/3) * 0.3
        
        # Wave receding sounds (minimal noise)
        wash_envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.03 * t_ocean + np.pi)
        wash_sound = np.sin(2 * np.pi * 0.025 * t_ocean) * wash_envelope * 0.3
        
        # Clean synthesis
        ocean_sound = main_waves + breaking_waves + deep_waves + wash_sound
        
        # Light smoothing
        for i in range(1, frames):
            ocean_sound[i] = 0.5 * ocean_sound[i-1] + 0.5 * ocean_sound[i]
            
        return ocean_sound
    
    def generate_nature_sounds(self, frames: int, ambient_ratio: float) -> np.ndarray:
        """Generate cleaner natural sounds with minimal noise"""
        if ambient_ratio == 0:
            return np.zeros(frames)
        
        t = (np.arange(frames) + self.noise_state) / self.sample_rate
        
        # Generate only clean ocean sounds
        ocean = self._generate_ocean_waves(t, frames)
        
        # Return wave sounds as is
        nature_mix = ocean
        
        # Final smoothing (noise removal)
        for i in range(1, frames):
            nature_mix[i] = 0.7 * nature_mix[i-1] + 0.3 * nature_mix[i]
        
        self.noise_state += frames
        return nature_mix * 1.5