"""
Sound type definitions for Mindfulness BGM Generator
"""

from enum import Enum

class SoundType(Enum):
    """Different types of synthesized sounds"""
    
    HARMONIC = "harmonic"      # Harmonic-rich tone with many overtones
    PURE = "pure"              # Close to pure tone with minimal harmonics
    SOFT_PAD = "soft_pad"      # Soft pad sound with gentle attack
    WARM = "warm"              # Warm tone with sub-harmonics
    BELL_LIKE = "bell_like"    # Bell-like tone with inharmonic partials

    def __str__(self):
        return self.value
    
    @classmethod
    def get_all(cls):
        """Get all available sound types"""
        return list(cls)
    
    @classmethod
    def get_random_except(cls, exclude_type):
        """Get a random sound type excluding the specified one"""
        import random
        types = cls.get_all()
        types.remove(exclude_type)
        return random.choice(types)