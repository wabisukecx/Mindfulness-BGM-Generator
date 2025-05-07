import numpy as np
import sounddevice as sd
import threading
import time
import random
import argparse
from typing import List, Tuple, Dict
from enum import Enum


# ── Settings ─────────────────────────────────────────────────────────
SAMPLE_RATE = 44100       # Hz
BUFFER_FRAMES = 1024      # Buffer size
VOLUME = 0.12             # Overall volume slightly increased

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
DEFAULT_AMBIENT_RATIO = 0.15  # Default ambient sound mix ratio

# ─────────────────────────────────────────────────────────────────
# Pentatonic scale
BASE_FREQS = np.array([220.00, 261.63, 329.63, 440.00, 523.25])  # A, C, E, A, C


class SoundType(Enum):
    """Sound types"""
    HARMONIC = "harmonic"      # Harmonic-rich tone
    PURE = "pure"              # Close to pure tone
    SOFT_PAD = "soft_pad"      # Soft pad sound
    WARM = "warm"              # Warm tone
    BELL_LIKE = "bell_like"    # Bell-like tone


class InstrumentConfig:
    """Percussion instrument configuration"""
    def __init__(self, min_interval: float, max_interval: float, enabled: bool = True):
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.enabled = enabled


class SlitDrum:
    """Slit drum class"""
    
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.is_playing = False
        self.drum_phase = 0
        self.drum_start_time = 0
        self.current_freq = 0
        
    def trigger(self, freq: float):
        """Trigger the drum"""
        self.is_playing = True
        self.drum_phase = 0
        self.drum_start_time = 0
        self.current_freq = freq
        
    def generate(self, frames: int) -> np.ndarray:
        """Generate slit drum sound"""
        if not self.is_playing:
            return np.zeros(frames)
            
        t = (np.arange(frames) + self.drum_phase) / self.sample_rate
        
        # Fundamental tone
        fundamental = np.sin(2 * np.pi * self.current_freq * t)
        # Add slight 2nd harmonic (for depth)
        second = np.sin(2 * np.pi * self.current_freq * 2 * t) * 0.2
        
        # Envelope (representing wooden resonance)
        envelope = np.exp(-3 * t) * (1 + 0.5 * np.exp(-10 * t))
        
        # Synthesis
        drum_sound = (fundamental + second) * envelope * 0.35
        
        self.drum_phase += frames
        self.drum_start_time += frames / self.sample_rate
        
        # Stop after 3 seconds
        if self.drum_start_time > 3.0:
            self.is_playing = False
            
        return drum_sound


class Handpan:
    """Handpan class"""
    
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.is_playing = False
        self.pan_phase = 0
        self.pan_start_time = 0
        self.current_freq = 0
        
    def trigger(self, freq: float):
        """Trigger the handpan"""
        self.is_playing = True
        self.pan_phase = 0
        self.pan_start_time = 0
        self.current_freq = freq
        
    def generate(self, frames: int) -> np.ndarray:
        """Generate handpan sound"""
        if not self.is_playing:
            return np.zeros(frames)
            
        t = (np.arange(frames) + self.pan_phase) / self.sample_rate
        
        # Fundamental and modest harmonics
        fundamental = np.sin(2 * np.pi * self.current_freq * t)
        second = np.sin(2 * np.pi * self.current_freq * 2 * t) * 0.25
        third = np.sin(2 * np.pi * self.current_freq * 3 * t) * 0.1
        
        # Add metallic resonance (with phase shift)
        metallic = np.sin(2 * np.pi * self.current_freq * 1.5 * t + np.pi/4) * 0.15
        
        # Envelope
        envelope = np.exp(-2 * t) * (1 + 0.3 * np.exp(-20 * t))
        
        # Synthesis
        pan_sound = (fundamental + second + third + metallic) * envelope * 0.3
        
        self.pan_phase += frames
        self.pan_start_time += frames / self.sample_rate
        
        # Stop after 4 seconds
        if self.pan_start_time > 4.0:
            self.is_playing = False
            
        return pan_sound


class TibetanBell:
    """Tibetan bell (Tingsha) class"""
    
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.is_playing = False
        self.bell_phase = 0
        self.bell_start_time = 0
        
    def trigger(self):
        """Ring the bell"""
        self.is_playing = True
        self.bell_phase = 0
        self.bell_start_time = 0
        
    def generate(self, frames: int) -> np.ndarray:
        """Generate bell sound"""
        if not self.is_playing:
            return np.zeros(frames)
            
        t = (np.arange(frames) + self.bell_phase) / self.sample_rate
        
        # Two frequencies (dissonant for bell-like quality)
        f1 = 2000
        f2 = 2100
        
        # Bell sound components
        wave1 = np.sin(2 * np.pi * f1 * t)
        wave2 = np.sin(2 * np.pi * f2 * t) * 0.8
        
        # Envelope (representing initial strike)
        attack = np.exp(-50 * t)
        decay = np.exp(-4 * t)
        envelope = attack * 0.3 + decay * 0.7
        
        # Synthesis
        bell_sound = (wave1 + wave2) * envelope * 0.25
        
        self.bell_phase += frames
        self.bell_start_time += frames / self.sample_rate
        
        # Stop after 3 seconds
        if self.bell_start_time > 3.0:
            self.is_playing = False
            
        return bell_sound


class MindfulnessBGM:
    """Dynamic mindfulness BGM generator class"""
    
    def __init__(self, bell_config: InstrumentConfig, drum_config: InstrumentConfig, 
                 handpan_config: InstrumentConfig, ambient_ratio: float):
        self.sample_rate = SAMPLE_RATE
        self.phase = 0
        self.lock = threading.Lock()
        
        # Sound type management
        self.current_sound_type = SoundType.PURE
        self.next_sound_type = None
        self.chord = self._create_chord()
        self.next_chord = None
        
        # Volume management
        self.current_volume = 0.7
        self.target_volume = 0.7
        self.volume_transition_speed = 0.0
        
        # Phase management
        self.noise_state = 0.0
        self.breath_phase = 0.0
        
        # Fade management
        self.fade_progress = 0.0
        self.is_transitioning = False
        
        # Ambient sound settings
        self.ambient_ratio = ambient_ratio
        
        # Percussion instruments
        self.tibetan_bell = TibetanBell(self.sample_rate)
        self.slit_drum = SlitDrum(self.sample_rate)
        self.handpan = Handpan(self.sample_rate)
        
        # Percussion configurations
        self.bell_config = bell_config
        self.drum_config = drum_config
        self.handpan_config = handpan_config
        
        # Effect buffer (reverb)
        self.reverb_buffer = np.zeros(int(self.sample_rate * 0.1))  # 0.1 second buffer
        self.reverb_index = 0
        
        # Start event scheduler
        threading.Thread(target=self._event_scheduler, daemon=True).start()
        
        # Conditionally start percussion schedulers
        if self.bell_config.enabled:
            threading.Thread(target=self._bell_scheduler, daemon=True).start()
        if self.drum_config.enabled:
            threading.Thread(target=self._drum_scheduler, daemon=True).start()
        if self.handpan_config.enabled:
            threading.Thread(target=self._handpan_scheduler, daemon=True).start()

    def _bell_scheduler(self):
        """Bell scheduler"""
        while True:
            wait_time = random.uniform(self.bell_config.min_interval, 
                                     self.bell_config.max_interval)
            time.sleep(wait_time)
            with self.lock:
                self.tibetan_bell.trigger()

    def _drum_scheduler(self):
        """Slit drum scheduler"""
        time.sleep(5)
        while True:
            wait_time = random.uniform(self.drum_config.min_interval, 
                                     self.drum_config.max_interval)
            time.sleep(wait_time)
            with self.lock:
                freq = random.choice(BASE_FREQS[:3]) / 2
                self.slit_drum.trigger(freq)

    def _handpan_scheduler(self):
        """Handpan scheduler"""
        time.sleep(10)
        while True:
            wait_time = random.uniform(self.handpan_config.min_interval, 
                                     self.handpan_config.max_interval)
            time.sleep(wait_time)
            with self.lock:
                freq = random.choice(BASE_FREQS)
                self.handpan.trigger(freq)

    def _event_scheduler(self):
        """Schedule random events"""
        while True:
            wait_time = random.uniform(MIN_EVENT_INTERVAL, MAX_EVENT_INTERVAL)
            time.sleep(wait_time)
            
            events = [
                self._change_sound_type,
                self._change_chord,
                self._change_volume,
                self._change_both
            ]
            
            event = random.choice(events)
            event()

    def _change_sound_type(self):
        """Change sound type"""
        with self.lock:
            sound_types = list(SoundType)
            sound_types.remove(self.current_sound_type)
            self.next_sound_type = random.choice(sound_types)
            self.is_transitioning = True
            self.fade_progress = 0.0

    def _change_chord(self):
        """Change chord"""
        with self.lock:
            self.next_chord = self._create_chord()
            self.is_transitioning = True
            self.fade_progress = 0.0

    def _change_volume(self):
        """Change volume"""
        with self.lock:
            self.target_volume = random.uniform(0.4, 0.8)
            duration = random.uniform(MIN_VOLUME_CHANGE_DURATION, MAX_VOLUME_CHANGE_DURATION)
            self.volume_transition_speed = 1.0 / (duration * self.sample_rate)

    def _change_both(self):
        """Change both sound type and chord"""
        self._change_sound_type()
        self._change_chord()

    def _create_chord(self) -> List[float]:
        """Generate well-balanced chords"""
        root = random.choice(BASE_FREQS)
        
        # Well-balanced chord types
        chord_types = [
            ([0, 7, 12], "Open Fifth"),          # Open fifth
            ([0, 4, 7], "Major"),                # Major
            ([0, 3, 7], "Minor"),                # Minor
            ([0, 5, 12], "Sus4"),                # Sus4
            ([0, 2, 7], "Sus2"),                 # Sus2
            ([0, 7, 14], "Add9"),                # Add9
            ([0, 4, 7, 12], "Major 7"),          # Major 7
            ([0, 3, 7, 12], "Minor 7"),          # Minor 7
            ([0, 5, 10], "Quartal"),             # Quartal harmony
            ([0, 7, 12, 19], "Open Octave"),     # Open octave
        ]
        
        intervals, chord_name = random.choice(chord_types)
        
        frequencies = []
        for interval in intervals:
            freq = root * (2 ** (interval / 12))
            frequencies.append(freq)
        
        return frequencies

    def generate_tone(self, freq: float, t: np.ndarray, sound_type: SoundType) -> np.ndarray:
        """Generate waveform according to sound type (moderate depth)"""
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
            # Soft attack
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
            # Bell-like decay
            decay = np.exp(-2 * t)
            return wave * decay * 0.4

    def apply_reverb(self, signal: np.ndarray) -> np.ndarray:
        """Simple reverb effect"""
        output = np.copy(signal)
        
        for i in range(len(signal)):
            # Read from reverb buffer
            reverb_sample = self.reverb_buffer[self.reverb_index] * 0.2
            
            # Add to output
            output[i] += reverb_sample
            
            # Add current sample to buffer
            self.reverb_buffer[self.reverb_index] = signal[i]
            
            # Advance index
            self.reverb_index = (self.reverb_index + 1) % len(self.reverb_buffer)
        
        return output

    def generate_nature_sounds(self, frames: int) -> np.ndarray:
        """Generate nature sounds (richer)"""
        if self.ambient_ratio == 0:
            return np.zeros(frames)
        
        t = (np.arange(frames) + self.noise_state) / self.sample_rate
        
        # Ocean waves (combining multiple frequencies)
        ocean1 = np.sin(2 * np.pi * 0.05 * t) * 0.03
        ocean2 = np.sin(2 * np.pi * 0.08 * t + np.pi/3) * 0.02
        ocean3 = np.sin(2 * np.pi * 0.03 * t + np.pi/6) * 0.01
        
        # Wind sound (filtered noise)
        wind = np.random.randn(frames) * 0.005
        # Simple low-pass filter
        for i in range(1, frames):
            wind[i] = 0.9 * wind[i-1] + 0.1 * wind[i]
        
        self.noise_state += frames
        return ocean1 + ocean2 + ocean3 + wind

    def callback(self, outdata, frames, time_info, status):
        """Audio callback"""
        t = (np.arange(frames) + self.phase) / self.sample_rate
        
        with self.lock:
            # Volume transition
            if self.current_volume != self.target_volume:
                if abs(self.current_volume - self.target_volume) < 0.01:
                    self.current_volume = self.target_volume
                else:
                    direction = 1 if self.target_volume > self.current_volume else -1
                    self.current_volume += direction * self.volume_transition_speed * frames
                    self.current_volume = np.clip(self.current_volume, 0.3, 0.8)
            
            # Fade processing
            if self.is_transitioning:
                self.fade_progress += frames / (FADE_TIME * self.sample_rate)
                if self.fade_progress >= 1.0:
                    self.fade_progress = 1.0
                    self.is_transitioning = False
                    if self.next_sound_type:
                        self.current_sound_type = self.next_sound_type
                        self.next_sound_type = None
                    if self.next_chord:
                        self.chord = self.next_chord
                        self.next_chord = None
        
        # Generate nature sounds
        nature_sounds = self.generate_nature_sounds(frames)
        
        # Generate percussion sounds
        bell_sound = self.tibetan_bell.generate(frames)
        drum_sound = self.slit_drum.generate(frames)
        handpan_sound = self.handpan.generate(frames)
        
        # Generate chord with current sound type
        current_wave = np.zeros(frames)
        for i, freq in enumerate(self.chord):
            # Add slightly different timing to each voice (natural spread)
            t_offset = t + i * 0.001
            tone = self.generate_tone(freq, t_offset, self.current_sound_type)
            current_wave += tone
        
        # Normalize by number of voices
        current_wave /= len(self.chord)
        
        # Generate next tone during transition
        if self.is_transitioning:
            next_wave = np.zeros(frames)
            next_chord = self.next_chord if self.next_chord else self.chord
            next_type = self.next_sound_type if self.next_sound_type else self.current_sound_type
            
            for i, freq in enumerate(next_chord):
                t_offset = t + i * 0.001
                tone = self.generate_tone(freq, t_offset, next_type)
                next_wave += tone
            
            next_wave /= len(next_chord)
            
            # Smooth crossfade
            alpha = self.fade_progress
            current_wave = (1 - alpha) * current_wave + alpha * next_wave
        
        # Breathing rhythm LFO
        self.breath_phase += frames / self.sample_rate
        breath_lfo = 0.97 + 0.03 * np.sin(2 * np.pi * BREATH_CYCLE * self.breath_phase)
        
        # Main LFO (slow modulation)
        lfo = 1.0 + 0.02 * np.sin(2 * np.pi * LFO_FREQ * t)
        
        # Apply LFO and volume
        current_wave *= breath_lfo * lfo * self.current_volume
        
        # Apply reverb
        current_wave = self.apply_reverb(current_wave)
        
        # Mix
        if self.ambient_ratio > 0:
            sig = current_wave * (1 - self.ambient_ratio) + nature_sounds * self.ambient_ratio
        else:
            sig = current_wave
        
        # Add percussion sounds
        sig += bell_sound + drum_sound + handpan_sound
        
        # Soft limiting
        sig *= VOLUME
        sig = np.tanh(sig * 0.8) * 0.9
        sig = sig.astype(np.float32)
        
        # Stereo output (with spread)
        sig_left = sig + np.roll(sig, 2) * 0.05
        sig_right = sig + np.roll(sig, -2) * 0.05
        outdata[:] = np.column_stack([sig_left, sig_right])
        
        self.phase += frames


def parse_instrument_interval(value: str) -> Tuple[float, float]:
    """Parse instrument interval settings"""
    if value == "0":
        return 0, 0
    
    if "-" in value:
        min_val, max_val = value.split("-")
        return float(min_val), float(max_val)
    else:
        val = float(value)
        return val * 0.8, val * 1.2


def parse_ambient_value(value: str) -> float:
    """Parse ambient sound settings"""
    val = float(value)
    return max(0.0, min(1.0, val))


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Mindfulness BGM Generator - Balanced Version")
    
    parser.add_argument("--bell", type=str, default="15-45",
                      help="Tibetan bell interval in seconds")
    parser.add_argument("--drum", type=str, default="8-25",
                      help="Slit drum interval in seconds")
    parser.add_argument("--handpan", type=str, default="12-30",
                      help="Handpan interval in seconds")
    parser.add_argument("--ambient", type=str, default=str(DEFAULT_AMBIENT_RATIO),
                      help=f"Ambient sound ratio (0-1, default: {DEFAULT_AMBIENT_RATIO})")
    
    args = parser.parse_args()
    
    # Parse instrument settings
    bell_min, bell_max = parse_instrument_interval(args.bell)
    drum_min, drum_max = parse_instrument_interval(args.drum)
    handpan_min, handpan_max = parse_instrument_interval(args.handpan)
    
    bell_config = InstrumentConfig(bell_min, bell_max, bell_min > 0)
    drum_config = InstrumentConfig(drum_min, drum_max, drum_min > 0)
    handpan_config = InstrumentConfig(handpan_min, handpan_max, handpan_min > 0)
    
    # Parse ambient sound settings
    ambient_ratio = parse_ambient_value(args.ambient)
    
    # Display settings
    print("Starting mindfulness BGM (Balanced Version)...")
    print("\nSettings:")
    print("- Enhanced sound depth without distortion")
    print("- Simple reverb effect")
    print("- Balanced harmonics")
    print("- Smooth transitions")
    
    print("\nInstrument settings:")
    
    if bell_config.enabled:
        print(f"- Tibetan bell: {bell_config.min_interval:.1f}-{bell_config.max_interval:.1f} seconds")
    else:
        print("- Tibetan bell: DISABLED")
    
    if drum_config.enabled:
        print(f"- Slit drum: {drum_config.min_interval:.1f}-{drum_config.max_interval:.1f} seconds")
    else:
        print("- Slit drum: DISABLED")
    
    if handpan_config.enabled:
        print(f"- Handpan: {handpan_config.min_interval:.1f}-{handpan_config.max_interval:.1f} seconds")
    else:
        print("- Handpan: DISABLED")
    
    if ambient_ratio > 0:
        print(f"- Ambient sounds: {ambient_ratio:.1%} mix ratio")
    else:
        print("- Ambient sounds: DISABLED")
    
    print("\nPress Ctrl+C to stop.")
    
    sd.default.samplerate = SAMPLE_RATE
    sd.default.blocksize = BUFFER_FRAMES
    sd.default.channels = 2
    
    generator = MindfulnessBGM(bell_config, drum_config, handpan_config, ambient_ratio)
    
    with sd.OutputStream(callback=generator.callback):
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nGracefully stopping...")


if __name__ == "__main__":
    main()