"""Audible ocean texture: slow wave envelopes modulate colored noise."""

import numpy as np


class AmbientSoundGenerator:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.ocean_phase = np.random.uniform(0, 2*np.pi)
        self.ocean_variation = np.random.uniform(0.8, 1.2)
        self.noise_state = 0
        self.filter_state = 0.0
        self.foam_filter_state = 0.0

    def generate_ocean_waves(self, t: np.ndarray, frames: int) -> np.ndarray:
        phase = self.ocean_phase
        # A shorter arrival and longer retreat, with smooth, quiet troughs.
        cycle = (0.075*self.ocean_variation*t + phase/(2*np.pi)) % 1.0
        shape = np.where(cycle < 0.3, cycle/0.3, (1.0-cycle)/0.7)
        swell = shape*shape*(3.0-2.0*shape)
        wash = 0.75 + 0.25 * np.sin(2*np.pi*0.027*t + phase)
        noise = np.random.randn(frames)
        colored = np.empty(frames)
        foam = np.empty(frames)
        # A sample-rate-independent low-pass filter with continuous block state.
        pole = np.exp(-2*np.pi*650/self.sample_rate)
        foam_pole = np.exp(-2*np.pi*1800/self.sample_rate)
        state = self.filter_state
        foam_state = self.foam_filter_state
        for i, sample in enumerate(noise):
            state = pole * state + (1-pole) * sample
            foam_state = foam_pole * foam_state + (1-foam_pole) * sample
            colored[i] = state
            foam[i] = foam_state
        self.filter_state = state
        self.foam_filter_state = foam_state
        colored *= np.sqrt((1+pole)/(1-pole))
        foam *= np.sqrt((1+foam_pole)/(1-foam_pole))
        # Sub-audible oscillators are envelopes, never added to the audio signal.
        return (colored * (0.018 + 0.05*swell) + foam * 0.006*swell**2) * wash

    def generate_nature_sounds(self, frames: int, ambient_ratio: float) -> np.ndarray:
        if ambient_ratio == 0:
            return np.zeros(frames)
        t = (np.arange(frames) + self.noise_state) / self.sample_rate
        result = self.generate_ocean_waves(t, frames)
        self.noise_state += frames
        return result
