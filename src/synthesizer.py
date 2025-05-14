"""
Sound synthesis functions for Mindfulness BGM Generator
Enhanced with natural harmonic progressions and extended chord variations
"""

import numpy as np
from typing import List
import random

from src.config import BASE_FREQS
from src.sound_types import SoundType


class Synthesizer:
    """Sound synthesis and chord generation utilities"""
    
    # Natural harmonic progressions based on the circle of fifths
    NATURAL_PROGRESSIONS = {
        220.00: [220.00, 329.63, 146.83, 293.66],  # A -> A, E, D, D (octave below)
        261.63: [261.63, 392.00, 174.61, 196.00],  # C -> C, G, F, G (octave below)
        329.63: [329.63, 246.94, 220.00, 164.81],  # E -> E, B, A, E (octave below)
        440.00: [440.00, 659.25, 293.66, 587.33],  # A -> A, E, D, D (octave above)
        523.25: [523.25, 783.99, 349.23, 392.00],  # C -> C, G, F, G (octave above)
    }
    
    # Pentatonic-based progressions
    PENTATONIC_PROGRESSIONS = {
        220.00: [220.00, 261.63, 329.63, 174.61],  # A -> A, C, E, F (below)
        261.63: [261.63, 329.63, 392.00, 220.00],  # C -> C, E, G, A (below)
        329.63: [329.63, 392.00, 523.25, 261.63],  # E -> E, G, C, C (below)
        440.00: [440.00, 523.25, 659.25, 349.23],  # A -> A, C, E, F (above)
        523.25: [523.25, 659.25, 784.00, 440.00],  # C -> C, E, G, A (above)
    }
    
    # Modal progressions (Dorian, Mixolydian, Aeolian, etc.)
    MODAL_PROGRESSIONS = {
        220.00: [220.00, 246.94, 293.66, 196.00, 174.61],  # A -> A, B, D, G, F
        261.63: [261.63, 293.66, 349.23, 233.08, 196.00],  # C -> C, D, F, Bb, G
        329.63: [329.63, 369.99, 293.66, 440.00, 246.94],  # E -> E, F#, D, A, B
        440.00: [440.00, 493.88, 587.33, 392.00, 349.23],  # A -> A, B, D, G, F
        523.25: [523.25, 587.33, 698.46, 466.16, 392.00],  # C -> C, D, F, Bb, G
    }
    
    def select_next_root(self, current_root: float, mode: str) -> float:
        """Select the next root note based on the circle of fifths"""
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
    
    def select_next_pentatonic_root(self, current_root: float) -> float:
        """Select the next root note based on the pentatonic scale"""
        if current_root is None:
            return random.choice(BASE_FREQS)
        
        # Find closest base frequency
        closest_base = min(BASE_FREQS, key=lambda x: abs(x - current_root))
        
        # 70% stay on same root - pentatonic emphasizes stability
        if random.random() < 0.7:
            return current_root
        else:
            if closest_base in self.PENTATONIC_PROGRESSIONS:
                candidates = self.PENTATONIC_PROGRESSIONS[closest_base]
                return random.choice(candidates)
            else:
                return random.choice(BASE_FREQS)
    
    def select_next_modal_root(self, current_root: float) -> float:
        """Select the next root note based on modal progressions"""
        if current_root is None:
            return random.choice(BASE_FREQS)
        
        # Find closest base frequency
        closest_base = min(BASE_FREQS, key=lambda x: abs(x - current_root))
        
        # 50% stay on same root - modal is somewhat dynamic
        if random.random() < 0.5:
            return current_root
        else:
            if closest_base in self.MODAL_PROGRESSIONS:
                candidates = self.MODAL_PROGRESSIONS[closest_base]
                return random.choice(candidates)
            else:
                return random.choice(BASE_FREQS)
    
    @classmethod
    def create_chord(cls, previous_root: float = None, mode: str = "balanced", chord_variety: str = "normal") -> List[float]:
        """Generate chords suitable for mindfulness with mode and variety consideration"""
        if previous_root is None:
            root = random.choice(BASE_FREQS)
        else:
            synth = cls()
            root = synth.select_next_root(previous_root, mode)
        
        # Chord types differ depending on variety level
        if chord_variety == "limited":
            # Simple chord types
            chord_types = [
                ([0], "Single Note", 0.25),                    # Drone
                ([0, 12], "Octave", 0.25),                     # Octave
                ([0, 7], "Perfect Fifth", 0.2),                # Perfect fifth
                ([0, 7, 12], "Open Fifth Octave", 0.15),       # Open fifth and octave
                ([0, 5], "Perfect Fourth", 0.15),              # Perfect fourth
            ]
        elif chord_variety == "normal":
            # Standard mindfulness chords
            if mode == "stable":
                # Prefer simpler, more stable chords
                chord_types = [
                    ([0], "Single Note", 0.25),                    # Drone
                    ([0, 12], "Octave", 0.2),                      # Octave
                    ([0, 7], "Perfect Fifth", 0.2),                # Perfect fifth
                    ([0, 7, 12], "Open Fifth Octave", 0.15),       # Open fifth and octave
                    ([0, 5], "Perfect Fourth", 0.1),               # Perfect fourth
                    ([0, 2, 7], "Sus2", 0.1),                      # Suspended 2nd
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
        else:  # extended
            # Extended chord variety (richer musical elements)
            if mode == "stable":
                # Even stable mode gets some variety
                chord_types = [
                    ([0], "Single Note", 0.1),                      # Drone
                    ([0, 12], "Octave", 0.1),                       # Octave
                    ([0, 7], "Perfect Fifth", 0.1),                 # Perfect fifth
                    ([0, 7, 12], "Open Fifth Octave", 0.1),         # Open fifth and octave
                    ([0, 5], "Perfect Fourth", 0.1),                # Perfect fourth
                    ([0, 5, 10], "Quartal Stack", 0.1),             # Quartal stack
                    ([0, 2, 7], "Sus2", 0.1),                       # Suspended 2nd
                    ([0, 2, 9], "Add9 Open", 0.08),                 # Open 9th
                    ([0, 4, 7], "Major", 0.08),                     # Major triad (slightly bright)
                    ([0, 3, 7], "Minor", 0.08),                     # Minor triad (slightly dark)
                    ([0, 2, 7, 9], "Sus2/9", 0.06),                 # Sus2 with 9th
                ]
            elif mode == "balanced":
                # Even more variety for balanced mode
                chord_types = [
                    ([0], "Single Note", 0.05),                     # Drone
                    ([0, 12], "Octave", 0.05),                      # Octave
                    ([0, 7], "Perfect Fifth", 0.05),                # Perfect fifth
                    ([0, 7, 12], "Open Fifth Octave", 0.05),        # Open fifth and octave
                    ([0, 5], "Perfect Fourth", 0.05),               # Perfect fourth
                    ([0, 5, 10], "Quartal Stack", 0.05),            # Quartal stack
                    ([0, 5, 10, 15], "Extended Quartal", 0.05),     # Extended quartal stack
                    ([0, 2, 7], "Sus2", 0.05),                      # Suspended 2nd
                    ([0, 7, 14], "Double Octave Fifth", 0.05),      # Two-octave fifth
                    ([0, 12, 19], "Octave Plus Fifth", 0.05),       # Octave plus fifth
                    ([0, 2, 9], "Add9 Open", 0.05),                 # Open 9th
                    ([0, 7, 17], "Tenth", 0.05),                    # 10th
                    ([0, 4, 7], "Major", 0.05),                     # Major triad
                    ([0, 3, 7], "Minor", 0.05),                     # Minor triad
                    ([0, 4, 7, 11], "Major7", 0.05),                # Major 7th (jazzy)
                    ([0, 3, 7, 10], "Minor7", 0.05),                # Minor 7th (jazzy)
                    ([0, 3, 7, 14], "Minor add9", 0.05),            # Minor add9
                    ([0, 2, 7, 11], "Sus2 Maj7", 0.05),             # Sus2 with Major 7
                    ([0, 5, 7, 12], "Sus4 Octave", 0.05),           # Sus4 with Octave
                    ([0, 4, 9, 14], "Major add11", 0.05),           # Major add11
                ]
            else:  # dynamic
                # Full spectrum of rich chords for dynamic mode
                chord_types = [
                    ([0], "Single Note", 0.02),                     # Drone
                    ([0, 12], "Octave", 0.02),                      # Octave
                    ([0, 7], "Perfect Fifth", 0.03),                # Perfect fifth
                    ([0, 7, 12], "Open Fifth Octave", 0.03),        # Open fifth and octave
                    ([0, 5], "Perfect Fourth", 0.03),               # Perfect fourth
                    ([0, 5, 10], "Quartal Stack", 0.03),            # Quartal stack
                    ([0, 5, 10, 15], "Extended Quartal", 0.03),     # Extended quartal stack
                    ([0, 2, 7], "Sus2", 0.03),                      # Suspended 2nd
                    ([0, 7, 14], "Double Octave Fifth", 0.03),      # Two-octave fifth
                    ([0, 12, 19], "Octave Plus Fifth", 0.03),       # Octave plus fifth
                    ([0, 2, 9], "Add9 Open", 0.03),                 # Open 9th
                    ([0, 7, 17], "Tenth", 0.03),                    # 10th
                    ([0, 4, 7], "Major", 0.05),                     # Major triad
                    ([0, 3, 7], "Minor", 0.05),                     # Minor triad
                    ([0, 4, 7, 11], "Major7", 0.05),                # Major 7th
                    ([0, 3, 7, 10], "Minor7", 0.05),                # Minor 7th
                    ([0, 3, 7, 14], "Minor add9", 0.05),            # Minor add9
                    ([0, 2, 7, 11], "Sus2 Maj7", 0.05),             # Sus2 with Major 7th
                    ([0, 5, 7, 12], "Sus4 Octave", 0.05),           # Sus4 with Octave
                    ([0, 4, 9, 14], "Major add11", 0.05),           # Major add11
                    ([0, 4, 7, 9], "Major add9", 0.05),             # Major add9
                    ([0, 3, 7, 9], "Minor add9", 0.05),             # Minor add9
                    ([0, 4, 7, 9, 14], "Major9/11", 0.05),          # Major 9/11
                    ([0, 3, 7, 10, 14], "Minor9", 0.05),            # Minor9
                    ([0, 1, 5, 8, 12], "Cluster1", 0.02),           # Cluster voicing 1
                    ([0, 2, 6, 9, 14], "Cluster2", 0.02),           # Cluster voicing 2
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
    
    @classmethod
    def create_pentatonic_chord(cls, previous_root: float = None, mode: str = "balanced") -> List[float]:
        """Generate chords based on the pentatonic scale"""
        if previous_root is None:
            root = random.choice(BASE_FREQS)
        else:
            synth = cls()
            root = synth.select_next_pentatonic_root(previous_root)
        
        # Chord types for pentatonic - emphasizes open voicings and perfect fifths/fourths
        chord_types = [
            ([0], "Root", 0.1),                            # Root only
            ([0, 12], "Octave", 0.1),                      # Octave
            ([0, 7], "Fifth", 0.1),                        # Perfect fifth
            ([0, 7, 12], "Fifth Octave", 0.1),             # Fifth + octave
            ([0, 7, 14], "Fifth + Major 9th", 0.1),        # Fifth + Major 9th
            ([0, 7, 16], "Fifth + Major 10th", 0.1),       # Fifth + Major 10th
            ([0, 4, 7], "Major", 0.08),                    # Major triad
            ([0, 2, 7], "Sus2", 0.08),                     # Suspended 2nd
            ([0, 7, 9], "Fifth + Major 2nd", 0.08),        # Fifth + Major 2nd
            ([0, 4, 9], "Major 3rd + 6th", 0.08),          # Major 3rd + 6th
            ([0, 2, 9], "Sus2 + 6th", 0.08),               # Sus2 + 6th
            ([0, 5, 9], "4th + 6th", 0.08),                # 4th + 6th
        ]
        
        # Adjust probabilities based on mode - pentatonic is naturally stable, so only slight adjustment
        if mode == "stable":
            # Prefer simpler chords in stable mode
            weights = [0.15, 0.15, 0.15, 0.15, 0.08, 0.08, 0.06, 0.06, 0.04, 0.04, 0.02, 0.02]
        elif mode == "balanced":
            # Default for balanced mode
            weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08]
        else:  # dynamic
            # Prefer more complex chords in dynamic mode
            weights = [0.05, 0.05, 0.06, 0.08, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.08, 0.08]
        
        # Select chord with weighted probability
        selected_index = random.choices(range(len(chord_types)), weights=weights)[0]
        intervals, chord_name = chord_types[selected_index]
        
        # Pentatonic scale adjustment - ensures all notes are in the pentatonic scale
        pentatonic_intervals = [0, 2, 4, 7, 9, 12, 14, 16, 19, 21]  # A pentatonic scale intervals
        adjusted_intervals = [i for i in intervals if i % 12 in [p % 12 for p in pentatonic_intervals]]
        
        # If all notes were filtered out, use the original intervals
        if not adjusted_intervals:
            adjusted_intervals = intervals
        
        frequencies = []
        for interval in adjusted_intervals:
            freq = root * (2 ** (interval / 12))
            frequencies.append(freq)
        
        return frequencies
    
    @classmethod
    def create_modal_chord(cls, previous_root: float = None, mode: str = "balanced") -> List[float]:
        """Generate chords based on modal progressions - Dorian, Mixolydian, Aeolian, etc."""
        if previous_root is None:
            root = random.choice(BASE_FREQS)
        else:
            synth = cls()
            root = synth.select_next_modal_root(previous_root)
        
        # Modal selection (Dorian, Phrygian, Lydian, Mixolydian, Aeolian)
        # Each mode has characteristic interval relationships for chord construction
        modal_types = [
            # Dorian mode - minor but with a bright 6th
            ([0, 3, 7, 9], "Dorian 1", 0.1),               # Minor add9
            ([0, 3, 7, 10], "Dorian 2", 0.1),              # Minor7
            ([0, 3, 9, 14], "Dorian 3", 0.1),              # Minor add6/9
            
            # Phrygian mode - minor with a dark 2nd
            ([0, 1, 7], "Phrygian 1", 0.08),               # Minor b9
            ([0, 3, 5, 10], "Phrygian 2", 0.08),           # Minor7 sus4
            ([0, 1, 5, 8], "Phrygian 3", 0.08),            # Phrygian chord
            
            # Lydian mode - major with a bright 4th
            ([0, 4, 6, 11], "Lydian 1", 0.08),             # Major #11
            ([0, 4, 11, 18], "Lydian 2", 0.08),            # Major7 #11
            ([0, 6, 11, 16], "Lydian 3", 0.08),            # Lydian voicing
            
            # Mixolydian mode - major with a minor 7th
            ([0, 4, 7, 10], "Mixolydian 1", 0.08),         # Dominant7
            ([0, 2, 7, 10], "Mixolydian 2", 0.08),         # Dominant7 sus2
            ([0, 5, 7, 10], "Mixolydian 3", 0.08),         # Dominant7 sus4
            
            # Aeolian mode - natural minor
            ([0, 3, 7], "Aeolian 1", 0.08),                # Minor triad
            ([0, 3, 10], "Aeolian 2", 0.08),               # Minor7 no5
            ([0, 3, 7, 10, 14], "Aeolian 3", 0.08),        # Minor9
        ]
        
        # Adjust probabilities based on mode
        if mode == "stable":
            # Prefer simpler chords in stable mode
            types_indices = [0, 1, 2, 6, 7, 8, 12, 13, 14]  # Dorian, Lydian, Aeolian
            weights = [0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        elif mode == "balanced":
            # Equal probability for all modes in balanced mode
            types_indices = range(len(modal_types))
            weights = None  # Uniform distribution
        else:  # dynamic
            # Prefer more complex voicings in dynamic mode
            types_indices = range(len(modal_types))
            weights = [0.05, 0.05, 0.08, 0.05, 0.05, 0.08, 0.05, 0.05, 0.08, 0.05, 0.05, 0.08, 0.05, 0.05, 0.08]
        
        # Select only from the allowed types
        filtered_types = [modal_types[i] for i in types_indices]
        if weights:
            filtered_weights = [weights[i] for i in range(len(types_indices))]
            selected_index = random.choices(range(len(filtered_types)), weights=filtered_weights)[0]
        else:
            selected_index = random.randrange(len(filtered_types))
        
        intervals, chord_name = filtered_types[selected_index]
        
        frequencies = []
        for interval in intervals:
            freq = root * (2 ** (interval / 12))
            frequencies.append(freq)
        
        return frequencies
    
    @staticmethod
    def generate_tone(freq: float, t: np.ndarray, sound_type: SoundType, harmonic_richness: str = "normal") -> np.ndarray:
        """Generate waveform according to sound type and harmonic richness"""
        # Adjust the strength and number of harmonics based on harmonic richness
        if harmonic_richness == "minimal":
            # Minimal harmonics - purer timbre
            harmonic_levels = {
                SoundType.HARMONIC: [1.0, 0.12, 0.05, 0.02],
                SoundType.PURE: [1.0, 0.03, 0.01],
                SoundType.SOFT_PAD: [1.0, 0.1, 0.05, 0.02],
                SoundType.WARM: [1.0, 0.2, 0.08, 0.03],
                SoundType.BELL_LIKE: [1.0, 0.2, 0.1, 0.05]
            }
        elif harmonic_richness == "normal":
            # Standard harmonic structure
            harmonic_levels = {
                SoundType.HARMONIC: [1.0, 0.25, 0.12, 0.06, 0.03, 0.015, 0.008],
                SoundType.PURE: [1.0, 0.05, 0.02],
                SoundType.SOFT_PAD: [1.0, 0.15, 0.08, 0.04],
                SoundType.WARM: [1.0, 0.35, 0.15, 0.08, 0.04, 0.02],
                SoundType.BELL_LIKE: [1.0, 0.3, 0.2, 0.1, 0.05]
            }
        else:  # rich
            # Rich harmonic structure
            harmonic_levels = {
                SoundType.HARMONIC: [1.0, 0.3, 0.15, 0.08, 0.04, 0.02, 0.01, 0.005, 0.0025],
                SoundType.PURE: [1.0, 0.08, 0.04, 0.02, 0.01],
                SoundType.SOFT_PAD: [1.0, 0.2, 0.1, 0.05, 0.025, 0.012],
                SoundType.WARM: [1.0, 0.4, 0.2, 0.1, 0.05, 0.025, 0.012, 0.006],
                SoundType.BELL_LIKE: [1.0, 0.35, 0.25, 0.15, 0.08, 0.04, 0.02]
            }
        
        # Add subharmonics (lower harmonics) - for WARM timbre
        subharmonic_levels = {
            "minimal": [0.25],
            "normal": [0.35, 0.02],
            "rich": [0.4, 0.08, 0.02]
        }
        
        # Harmonic levels to use
        levels = harmonic_levels[sound_type]
        
        if sound_type == SoundType.HARMONIC:
            wave = np.zeros_like(t)
            # Fundamental and harmonics
            for i, level in enumerate(levels):
                wave += level * np.sin(2 * np.pi * freq * (i + 1) * t)
            return wave * 0.35
            
        elif sound_type == SoundType.PURE:
            wave = np.zeros_like(t)
            for i, level in enumerate(levels):
                wave += level * np.sin(2 * np.pi * freq * (i + 1) * t)
            return wave * 0.5
            
        elif sound_type == SoundType.SOFT_PAD:
            wave = np.zeros_like(t)
            # Fundamental and harmonics
            for i, level in enumerate(levels):
                if i == 0:
                    wave += level * np.sin(2 * np.pi * freq * t)
                else:
                    wave += level * np.sin(2 * np.pi * freq * i * t)
            
            # Add 0.5th harmonic (perfect fifth below octave) for SOFT_PAD
            wave += 0.15 * np.sin(2 * np.pi * freq * 0.5 * t)
            
            # Soft attack
            attack = 1 - np.exp(-3 * t)
            return wave * attack * 0.4
            
        elif sound_type == SoundType.WARM:
            wave = np.zeros_like(t)
            # Fundamental and harmonics
            for i, level in enumerate(levels):
                if i < len(levels):
                    wave += level * np.sin(2 * np.pi * freq * (i + 1) * t)
            
            # Add subharmonics (lower harmonics)
            subs = subharmonic_levels[harmonic_richness]
            for i, level in enumerate(subs):
                subfreq = freq * (0.5 ** (i + 1))  # One octave down, two octaves down, etc.
                wave += level * np.sin(2 * np.pi * subfreq * t)
            
            return wave * 0.35
            
        elif sound_type == SoundType.BELL_LIKE:
            wave = np.zeros_like(t)
            # Fundamental and harmonics
            for i, level in enumerate(levels):
                if i < 3:  # First three are normal harmonics
                    wave += level * np.sin(2 * np.pi * freq * (i + 1) * t)
                elif i == 3:  # 4th is inharmonic overtone
                    wave += level * np.sin(2 * np.pi * freq * 5.7 * t)
                elif i == 4:  # 5th is also inharmonic overtone
                    wave += level * np.sin(2 * np.pi * freq * 7.3 * t)
                elif i >= 5:  # Higher inharmonic overtones
                    wave += level * np.sin(2 * np.pi * freq * (8.5 + i * 0.8) * t)
            
            # Bell-like decay
            decay = np.exp(-2 * t)
            return wave * decay * 0.4
            
        # Default fallback
        return np.sin(2 * np.pi * freq * t) * 0.5
