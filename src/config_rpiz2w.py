"""
Configuration for Mindfulness BGM Generator - Raspberry Pi Zero 2 W Version
Optimized for lower-power devices
"""

import numpy as np

# Audio settings - Optimized for RPi Zero 2 W
SAMPLE_RATE = 16000        # Hz - Lower sample rate
BUFFER_FRAMES = 1024       # Buffer size - Smaller buffer
VOLUME = 0.12              # Overall volume

# Event interval settings
MIN_EVENT_INTERVAL = 5.0   # Minimum event interval (seconds)
MAX_EVENT_INTERVAL = 15.0  # Maximum event interval (seconds)

# Volume change settings
MIN_VOLUME_CHANGE_DURATION = 3.0  # Minimum volume change duration (seconds)
MAX_VOLUME_CHANGE_DURATION = 8.0  # Maximum volume change duration (seconds)

# Fade settings
FADE_TIME = 2.0     # Fade in/out time
LFO_FREQ = 0.03     # LFO frequency (Hz)

# Mindfulness settings
BREATH_CYCLE = 0.2  # Breathing rhythm frequency (Hz)

# Default ambient sound ratio
DEFAULT_AMBIENT_RATIO = 0.3  # Default ambient sound mix ratio

# Musical settings
# Pentatonic scale (A, C, E, A, C)
BASE_FREQS = np.array([220.00, 261.63, 329.63, 440.00, 523.25])

# Simplified effects for RPi Zero 2 W
REVERB_TIME = 0.1    # Shorter reverb buffer
REVERB_MIX = 0.2     # Less reverb mix
STEREO_DELAY = 2     # Simpler stereo delay
STEREO_MIX = 0.05    # Less stereo mix
