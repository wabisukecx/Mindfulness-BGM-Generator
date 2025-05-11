"""
Sound synthesis functions for Mindfulness BGM Generator
Enhanced with natural harmonic progressions
"""

import numpy as np
from typing import List
import random

from src.config import BASE_FREQS
from src.sound_types import SoundType


class Synthesizer:
    """Sound synthesis and chord generation utilities"""
    
    # Natural harmonic progressions (5度圏に基づく)
    NATURAL_PROGRESSIONS = {
        220.00: [220.00, 329.63, 146.83, 293.66],  # A -> A, E, D, D (octave below)
        261.63: [261.63, 392.00, 174.61, 196.00],  # C -> C, G, F, G (octave below)
        329.63: [329.63, 246.94, 220.00, 164.81],  # E -> E, B, A, E (octave below)
        440.00: [440.00, 659.25, 293.66, 587.33],  # A -> A, E, D, D (octave above)
        523.25: [523.25, 783.99, 349.23, 392.00],  # C -> C, G, F, G (octave above)
    }
    
    def select_next_root(self, current_root: float, mode: str) -> float:
        """Select next root based on mode and current root"""
        if current_root is None:
            return random.choice(BASE_FREQS)
        
        # Find closest base frequency
        closest_base = min(BASE_FREQS, key=lambda x: abs(x - current_root))
        
        if mode == "stable":
            # 80% stay on same root
            if random.random() < 0.8:
                return current_root
            else:
                # Move to harmonically related note
                if closest_base in self.NATURAL_PROGRESSIONS:
                    candidates = self.NATURAL_PROGRESSIONS[closest_base]
                    return random.choice(candidates)
                else:
                    return random.choice(BASE_FREQS)
        
        elif mode == "balanced":
            # 60% stay on same root
            if random.random() < 0.6:
                return current_root
            else:
                if closest_base in self.NATURAL_PROGRESSIONS:
                    candidates = self.NATURAL_PROGRESSIONS[closest_base]
                    # Prefer closer intervals
                    weights = [0.3, 0.3, 0.2, 0.2]  # Prefer root and fifth
                    return random.choices(candidates, weights=weights)[0]
                else:
                    return random.choice(BASE_FREQS)
        
        else:  # dynamic
            # 30% stay on same root
            if random.random() < 0.3:
                return current_root
            else:
                return random.choice(BASE_FREQS)
    
    @classmethod
    def create_chord(cls, previous_root: float = None, mode: str = "balanced") -> List[float]:
        """Generate chords suitable for mindfulness with mode consideration"""
        if previous_root is None:
            root = random.choice(BASE_FREQS)
        else:
            synth = cls()
            root = synth.select_next_root(previous_root, mode)
        
        # Mindfulness-oriented chord types
        if mode == "stable":
            # Prefer simpler, more stable chords
            chord_types = [
                ([0], "Single Note", 0.3),                    # Drone
                ([0, 12], "Octave", 0.25),                    # Octave
                ([0, 7], "Perfect Fifth", 0.2),               # Perfect fifth
                ([0, 7, 12], "Open Fifth Octave", 0.15),      # Open fifth and octave
                ([0, 5], "Perfect Fourth", 0.1),              # Perfect fourth
            ]
        elif mode == "balanced":
            # Balanced chord selection
            chord_types = [
                ([0], "Single Note", 0.15),                   # Drone
                ([0, 12], "Octave", 0.15),                    # Octave
                ([0, 7], "Perfect Fifth", 0.15),              # Perfect fifth
                ([0, 7, 12], "Open Fifth Octave", 0.1),       # Open fifth and octave
                ([0, 5], "Perfect Fourth", 0.1),              # Perfect fourth
                ([0, 5, 10], "Quartal Stack", 0.1),           # Quartal stack
                ([0, 2, 7], "Sus2", 0.1),                     # Suspended 2nd
                ([0, 7, 14], "Double Octave Fifth", 0.05),    # Two-octave fifth
                ([0, 2, 9], "Add9 Open", 0.05),               # Open 9th
                ([0, 7, 17], "Tenth", 0.05),                  # 10th
            ]
        else:  # dynamic
            # More variety in chord selection
            chord_types = [
                ([0], "Single Note", 0.08),                     # Drone
                ([0, 12], "Octave", 0.08),                      # Octave
                ([0, 7], "Perfect Fifth", 0.08),                # Perfect fifth
                ([0, 7, 12], "Open Fifth Octave", 0.08),        # Open fifth and octave
                ([0, 5], "Perfect Fourth", 0.08),               # Perfect fourth
                ([0, 5, 10], "Quartal Stack", 0.08),            # Quartal stack
                ([0, 5, 10, 15], "Extended Quartal", 0.08),     # Extended quartal stack
                ([0, 2, 7], "Sus2", 0.08),                      # Suspended 2nd
                ([0, 7, 14], "Double Octave Fifth", 0.08),      # Two-octave fifth
                ([0, 12, 19], "Octave Plus Fifth", 0.08),       # Octave plus fifth
                ([0, 2, 9], "Add9 Open", 0.08),                 # Open 9th
                ([0, 7, 17], "Tenth", 0.08),                    # 10th
            ]
        
        # Extract intervals, names, and weights
        chord_options = [(intervals, name) for intervals, name, _ in chord_types]
        weights = [weight for _, _, weight in chord_types]
        
        # Select chord type with weighted probability
        selected_chord = random.choices(chord_options, weights=weights)[0]
        intervals, chord_name = selected_chord
        
        frequencies = []
        for interval in intervals:
            freq = root * (2 ** (interval / 12))
            frequencies.append(freq)
        
        return frequencies
    
    @staticmethod
    def generate_tone(freq: float, t: np.ndarray, sound_type: SoundType) -> np.ndarray:
        """Generate waveform according to sound type (high quality)"""
        if sound_type == SoundType.HARMONIC:
            wave = np.sin(2 * np.pi * freq * t)
            wave += 0.25 * np.sin(2 * np.pi * freq * 2 * t)
            wave += 0.12 * np.sin(2 * np.pi * freq * 3 * t)
            wave += 0.06 * np.sin(2 * np.pi * freq * 4 * t)
            wave += 0.03 * np.sin(2 * np.pi * freq * 5 * t)  # 5th harmonic
            wave += 0.015 * np.sin(2 * np.pi * freq * 6 * t)  # 6th harmonic
            wave += 0.008 * np.sin(2 * np.pi * freq * 7 * t)  # 7th harmonic
            return wave * 0.35
            
        elif sound_type == SoundType.PURE:
            wave = np.sin(2 * np.pi * freq * t)
            wave += 0.05 * np.sin(2 * np.pi * freq * 2 * t)
            wave += 0.02 * np.sin(2 * np.pi * freq * 3 * t)  # Slight 3rd harmonic
            return wave * 0.5
            
        elif sound_type == SoundType.SOFT_PAD:
            wave = np.sin(2 * np.pi * freq * t)
            wave += 0.15 * np.sin(2 * np.pi * freq * 0.5 * t)
            wave += 0.08 * np.sin(2 * np.pi * freq * 2 * t)
            wave += 0.04 * np.sin(2 * np.pi * freq * 3 * t)  # Additional harmonic
            # Soft attack
            attack = 1 - np.exp(-3 * t)
            return wave * attack * 0.4
            
        elif sound_type == SoundType.WARM:
            wave = np.sin(2 * np.pi * freq * t)
            wave += 0.35 * np.sin(2 * np.pi * freq * 0.5 * t)
            wave += 0.15 * np.sin(2 * np.pi * freq * 2 * t)
            wave += 0.08 * np.sin(2 * np.pi * freq * 3 * t)
            wave += 0.04 * np.sin(2 * np.pi * freq * 4 * t)  # 4th harmonic
            wave += 0.02 * np.sin(2 * np.pi * freq * 0.25 * t)  # Sub-harmonic
            return wave * 0.35
            
        elif sound_type == SoundType.BELL_LIKE:
            wave = np.sin(2 * np.pi * freq * t)
            wave += 0.3 * np.sin(2 * np.pi * freq * 2 * t)
            wave += 0.2 * np.sin(2 * np.pi * freq * 3 * t)
            wave += 0.1 * np.sin(2 * np.pi * freq * 5.7 * t)  # Inharmonic partial
            wave += 0.05 * np.sin(2 * np.pi * freq * 7.3 * t)  # Another inharmonic
            # Bell-like decay
            decay = np.exp(-2 * t)
            return wave * decay * 0.4