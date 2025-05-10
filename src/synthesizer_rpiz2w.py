"""
Sound synthesis functions for Mindfulness BGM Generator - Raspberry Pi Zero 2 W Version
Optimized for lower computational requirements
"""

import numpy as np
from typing import List
import random

from src.config_rpiz2w import BASE_FREQS
from src.sound_types import SoundType


class SynthesizerRPiZ2W:
    """Optimized sound synthesis for Raspberry Pi Zero 2 W"""
    
    @staticmethod
    def create_chord() -> List[float]:
        """Generate chords suitable for mindfulness (simplified)"""
        root = random.choice(BASE_FREQS)
        
        # Mindfulness-oriented chord types
        chord_types = [
            ([0], "Single Note"),                     # Drone
            ([0, 12], "Octave"),                      # Octave
            ([0, 7], "Perfect Fifth"),                # Perfect fifth
            ([0, 7, 12], "Open Fifth Octave"),        # Open fifth and octave
            ([0, 5], "Perfect Fourth"),               # Perfect fourth
            ([0, 5, 10], "Quartal Stack"),            # Quartal stack
            ([0, 5, 10, 15], "Extended Quartal"),     # Extended quartal stack
            ([0, 2, 7], "Sus2"),                      # Suspended 2nd
            ([0, 7, 14], "Double Octave Fifth"),      # Two-octave fifth
            ([0, 12, 19], "Octave Plus Fifth"),       # Octave plus fifth
            ([0, 2, 9], "Add9 Open"),                 # Open 9th
            ([0, 7, 17], "Tenth"),                    # 10th
        ]
        
        intervals, chord_name = random.choice(chord_types)
        
        frequencies = []
        for interval in intervals:
            freq = root * (2 ** (interval / 12))
            frequencies.append(freq)
        
        return frequencies
    
    @staticmethod
    def generate_tone(freq: float, t: np.ndarray, sound_type: SoundType) -> np.ndarray:
        """Generate waveform according to sound type (optimized)"""
        if sound_type == SoundType.HARMONIC:
            wave = np.sin(2 * np.pi * freq * t)
            wave += 0.25 * np.sin(2 * np.pi * freq * 2 * t)
            wave += 0.12 * np.sin(2 * np.pi * freq * 3 * t)
            wave += 0.06 * np.sin(2 * np.pi * freq * 4 * t)
            return wave * 0.35
            
        elif sound_type == SoundType.PURE:
            wave = np.sin(2 * np.pi * freq * t)
            wave += 0.05 * np.sin(2 * np.pi * freq * 2 * t)
            return wave * 0.5
            
        elif sound_type == SoundType.SOFT_PAD:
            wave = np.sin(2 * np.pi * freq * t)
            wave += 0.15 * np.sin(2 * np.pi * freq * 0.5 * t)
            wave += 0.08 * np.sin(2 * np.pi * freq * 2 * t)
            attack = 1 - np.exp(-3 * t)
            return wave * attack * 0.4
            
        elif sound_type == SoundType.WARM:
            wave = np.sin(2 * np.pi * freq * t)
            wave += 0.35 * np.sin(2 * np.pi * freq * 0.5 * t)
            wave += 0.15 * np.sin(2 * np.pi * freq * 2 * t)
            wave += 0.08 * np.sin(2 * np.pi * freq * 3 * t)
            return wave * 0.35
            
        elif sound_type == SoundType.BELL_LIKE:
            wave = np.sin(2 * np.pi * freq * t)
            wave += 0.3 * np.sin(2 * np.pi * freq * 2 * t)
            wave += 0.2 * np.sin(2 * np.pi * freq * 3 * t)
            decay = np.exp(-2 * t)
            return wave * decay * 0.4