"""
Configuration and constants for Mindfulness BGM Generator
"""

import numpy as np

# Audio settings - High Quality
SAMPLE_RATE = 48000        # Hz - CD quality or higher
BUFFER_FRAMES = 2048       # Buffer size - larger for noise prevention
VOLUME = 0.12              # Overall volume

# Event interval settings
MIN_EVENT_INTERVAL = 5.0   # Minimum event interval (seconds)
MAX_EVENT_INTERVAL = 15.0  # Maximum event interval (seconds)

# Volume change settings
MIN_VOLUME_CHANGE_DURATION = 3.0  # Minimum volume change duration (seconds)
MAX_VOLUME_CHANGE_DURATION = 8.0  # Maximum volume change duration (seconds)

# Fade settings
FADE_TIME = 3.0     # Fade in/out time - smoother transitions
LFO_FREQ = 0.03     # LFO frequency (Hz)

# Mindfulness settings
BREATH_CYCLE = 0.2  # Breathing rhythm frequency (Hz)

# Default ambient sound ratio
DEFAULT_AMBIENT_RATIO = 0.3  # Default ambient sound mix ratio

# Musical settings
# Pentatonic scale (A, C, E, A, C)
BASE_FREQS = np.array([220.00, 261.63, 329.63, 440.00, 523.25])

# Reverb settings
REVERB_TIME = 0.3  # Reverb buffer time in seconds
REVERB_MIX = 0.35  # Reverb mix ratio

# Stereo enhancement settings
STEREO_DELAY_1 = 3  # Primary stereo delay in samples
STEREO_DELAY_2 = 6  # Secondary stereo delay in samples
STEREO_MIX_1 = 0.08  # Primary stereo mix ratio
STEREO_MIX_2 = 0.04  # Secondary stereo mix ratio

# Soft limiting settings
LIMITER_THRESHOLD = 0.7  # Tanh threshold
LIMITER_CEILING = 0.95   # Output ceiling