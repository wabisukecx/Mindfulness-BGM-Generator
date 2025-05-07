import streamlit as st
import numpy as np
import sounddevice as sd
import threading
import time
import random
from typing import List, Tuple
from enum import Enum
from streamlit.runtime.scriptrunner import add_script_run_ctx

# Initialize session state
if 'generator' not in st.session_state:
    st.session_state.generator = None
if 'stream' not in st.session_state:
    st.session_state.stream = None
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False
if 'selected_instrument' not in st.session_state:
    st.session_state.selected_instrument = None
if 'current_status' not in st.session_state:
    st.session_state.current_status = "Stopped"
if 'thread' not in st.session_state:
    st.session_state.thread = None

# Original constants from main.py
SAMPLE_RATE = 44100
BUFFER_FRAMES = 1024
VOLUME = 0.12
MIN_EVENT_INTERVAL = 5.0
MAX_EVENT_INTERVAL = 15.0
MIN_VOLUME_CHANGE_DURATION = 3.0
MAX_VOLUME_CHANGE_DURATION = 8.0
FADE_TIME = 2.0
LFO_FREQ = 0.03
BREATH_CYCLE = 0.2
DEFAULT_AMBIENT_RATIO = 0.3
BASE_FREQS = np.array([220.00, 261.63, 329.63, 440.00, 523.25])


# Copy all classes from main.py
class SoundType(Enum):
    """Sound types"""
    HARMONIC = "harmonic"
    PURE = "pure"
    SOFT_PAD = "soft_pad"
    WARM = "warm"
    BELL_LIKE = "bell_like"


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
        
        fundamental = np.sin(2 * np.pi * self.current_freq * t)
        second = np.sin(2 * np.pi * self.current_freq * 2 * t) * 0.2
        
        envelope = np.exp(-3 * t) * (1 + 0.5 * np.exp(-10 * t))
        drum_sound = (fundamental + second) * envelope * 0.35
        
        self.drum_phase += frames
        self.drum_start_time += frames / self.sample_rate
        
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
        
        fundamental = np.sin(2 * np.pi * self.current_freq * t)
        second = np.sin(2 * np.pi * self.current_freq * 2 * t) * 0.25
        third = np.sin(2 * np.pi * self.current_freq * 3 * t) * 0.1
        metallic = np.sin(2 * np.pi * self.current_freq * 1.5 * t + np.pi/4) * 0.15
        
        envelope = np.exp(-2 * t) * (1 + 0.3 * np.exp(-20 * t))
        pan_sound = (fundamental + second + third + metallic) * envelope * 0.3
        
        self.pan_phase += frames
        self.pan_start_time += frames / self.sample_rate
        
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
        self.current_freq = 2000
        
    def trigger(self, freq: float = None):
        """Ring the bell"""
        self.is_playing = True
        self.bell_phase = 0
        self.bell_start_time = 0
        if freq:
            self.current_freq = freq
        
    def generate(self, frames: int) -> np.ndarray:
        """Generate bell sound"""
        if not self.is_playing:
            return np.zeros(frames)
            
        t = (np.arange(frames) + self.bell_phase) / self.sample_rate
        
        f1 = self.current_freq
        f2 = self.current_freq * 1.05
        
        wave1 = np.sin(2 * np.pi * f1 * t)
        wave2 = np.sin(2 * np.pi * f2 * t) * 0.8
        
        attack = np.exp(-50 * t)
        decay = np.exp(-4 * t)
        envelope = attack * 0.3 + decay * 0.7
        
        bell_sound = (wave1 + wave2) * envelope * 0.25
        
        self.bell_phase += frames
        self.bell_start_time += frames / self.sample_rate
        
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
        self.reverb_buffer = np.zeros(int(self.sample_rate * 0.1))
        self.reverb_index = 0
        
        # Stop flag
        self.stop_flag = threading.Event()
        
        # Start event scheduler
        self.schedulers = []
        scheduler_thread = threading.Thread(target=self._event_scheduler, daemon=True)
        add_script_run_ctx(scheduler_thread)
        scheduler_thread.start()
        self.schedulers.append(scheduler_thread)
        
        # Conditionally start percussion schedulers
        if self.bell_config.enabled:
            bell_thread = threading.Thread(target=self._bell_scheduler, daemon=True)
            add_script_run_ctx(bell_thread)
            bell_thread.start()
            self.schedulers.append(bell_thread)
        if self.drum_config.enabled:
            drum_thread = threading.Thread(target=self._drum_scheduler, daemon=True)
            add_script_run_ctx(drum_thread)
            drum_thread.start()
            self.schedulers.append(drum_thread)
        if self.handpan_config.enabled:
            handpan_thread = threading.Thread(target=self._handpan_scheduler, daemon=True)
            add_script_run_ctx(handpan_thread)
            handpan_thread.start()
            self.schedulers.append(handpan_thread)

    def stop(self):
        """Stop all threads"""
        self.stop_flag.set()
        for thread in self.schedulers:
            thread.join(timeout=1.0)

    def _update_status(self, status: str):
        """Update status through callback"""
        # Avoid direct UI updates from threads
        pass

    def _bell_scheduler(self):
        """Bell scheduler"""
        while not self.stop_flag.is_set():
            wait_time = random.uniform(self.bell_config.min_interval, 
                                     self.bell_config.max_interval)
            if self.stop_flag.wait(wait_time):
                break
            
            num_strikes = random.randint(1, 3)
            for i in range(num_strikes):
                with self.lock:
                    if self.chord:
                        base_freq = random.choice(self.chord)
                        freq = base_freq * random.choice([4, 5, 6, 8])
                    else:
                        freq = random.choice(BASE_FREQS) * random.choice([4, 5, 6, 8])
                    self.tibetan_bell.trigger(freq)
                
                if i < num_strikes - 1:
                    if self.stop_flag.wait(random.uniform(1.0, 4.0)):
                        break

    def _drum_scheduler(self):
        """Slit drum scheduler"""
        if self.stop_flag.wait(5):
            return
        while not self.stop_flag.is_set():
            wait_time = random.uniform(self.drum_config.min_interval, 
                                     self.drum_config.max_interval)
            if self.stop_flag.wait(wait_time):
                break
            
            num_strikes = random.randint(1, 3)
            for i in range(num_strikes):
                with self.lock:
                    if self.chord:
                        base_freq = random.choice(self.chord)
                        freq = base_freq * random.choice([0.25, 0.5, 0.75])
                    else:
                        freq = random.choice(BASE_FREQS[:3]) * random.choice([0.5, 0.75, 1])
                    self.slit_drum.trigger(freq)
                
                if i < num_strikes - 1:
                    if self.stop_flag.wait(random.uniform(1.0, 3.0)):
                        break

    def _handpan_scheduler(self):
        """Handpan scheduler"""
        if self.stop_flag.wait(10):
            return
        while not self.stop_flag.is_set():
            wait_time = random.uniform(self.handpan_config.min_interval, 
                                     self.handpan_config.max_interval)
            if self.stop_flag.wait(wait_time):
                break
            
            num_strikes = random.randint(1, 3)
            for i in range(num_strikes):
                with self.lock:
                    if self.chord:
                        base_freq = random.choice(self.chord)
                        freq = base_freq * random.choice([1, 2, 3])
                    else:
                        freq = random.choice(BASE_FREQS) * random.choice([1, 1.5, 2])
                    self.handpan.trigger(freq)
                
                if i < num_strikes - 1:
                    if self.stop_flag.wait(random.uniform(1.0, 3.5)):
                        break

    def _event_scheduler(self):
        """Schedule random events"""
        while not self.stop_flag.is_set():
            wait_time = random.uniform(MIN_EVENT_INTERVAL, MAX_EVENT_INTERVAL)
            if self.stop_flag.wait(wait_time):
                break
            
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
        """Generate chords suitable for mindfulness"""
        root = random.choice(BASE_FREQS)
        
        chord_types = [
            ([0], "Single Note"),
            ([0, 12], "Octave"),
            ([0, 7], "Perfect Fifth"),
            ([0, 7, 12], "Open Fifth Octave"),
            ([0, 5], "Perfect Fourth"),
            ([0, 5, 10], "Quartal Stack"),
            ([0, 5, 10, 15], "Extended Quartal"),
            ([0, 2, 7], "Sus2"),
            ([0, 7, 14], "Double Octave Fifth"),
            ([0, 12, 19], "Octave Plus Fifth"),
            ([0, 2, 9], "Add9 Open"),
            ([0, 7, 17], "Tenth"),
        ]
        
        intervals, chord_name = random.choice(chord_types)
        
        frequencies = []
        for interval in intervals:
            freq = root * (2 ** (interval / 12))
            frequencies.append(freq)
        
        return frequencies

    def generate_tone(self, freq: float, t: np.ndarray, sound_type: SoundType) -> np.ndarray:
        """Generate waveform according to sound type"""
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

    def apply_reverb(self, signal: np.ndarray) -> np.ndarray:
        """Simple reverb effect"""
        output = np.copy(signal)
        
        for i in range(len(signal)):
            reverb_sample = self.reverb_buffer[self.reverb_index] * 0.2
            output[i] += reverb_sample
            self.reverb_buffer[self.reverb_index] = signal[i]
            self.reverb_index = (self.reverb_index + 1) % len(self.reverb_buffer)
        
        return output

    def generate_nature_sounds(self, frames: int) -> np.ndarray:
        """Generate nature sounds"""
        if self.ambient_ratio == 0:
            return np.zeros(frames)
        
        t = (np.arange(frames) + self.noise_state) / self.sample_rate
        
        ocean1 = np.sin(2 * np.pi * 0.05 * t) * 0.08
        ocean2 = np.sin(2 * np.pi * 0.08 * t + np.pi/3) * 0.06
        ocean3 = np.sin(2 * np.pi * 0.03 * t + np.pi/6) * 0.04
        ocean4 = np.sin(2 * np.pi * 0.02 * t + np.pi/2) * 0.03
        
        wind = np.random.randn(frames) * 0.02
        for i in range(1, frames):
            wind[i] = 0.85 * wind[i-1] + 0.15 * wind[i]
        
        rumble = np.sin(2 * np.pi * 0.01 * t) * 0.02
        
        self.noise_state += frames
        return ocean1 + ocean2 + ocean3 + ocean4 + wind + rumble

    def callback(self, outdata, frames, time_info, status):
        """Audio callback"""
        t = (np.arange(frames) + self.phase) / self.sample_rate
        
        with self.lock:
            if self.current_volume != self.target_volume:
                if abs(self.current_volume - self.target_volume) < 0.01:
                    self.current_volume = self.target_volume
                else:
                    direction = 1 if self.target_volume > self.current_volume else -1
                    self.current_volume += direction * self.volume_transition_speed * frames
                    self.current_volume = np.clip(self.current_volume, 0.3, 0.8)
            
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
        
        nature_sounds = self.generate_nature_sounds(frames)
        bell_sound = self.tibetan_bell.generate(frames)
        drum_sound = self.slit_drum.generate(frames)
        handpan_sound = self.handpan.generate(frames)
        
        current_wave = np.zeros(frames)
        for i, freq in enumerate(self.chord):
            t_offset = t + i * 0.001
            tone = self.generate_tone(freq, t_offset, self.current_sound_type)
            current_wave += tone
        
        current_wave /= len(self.chord)
        
        if self.is_transitioning:
            next_wave = np.zeros(frames)
            next_chord = self.next_chord if self.next_chord else self.chord
            next_type = self.next_sound_type if self.next_sound_type else self.current_sound_type
            
            for i, freq in enumerate(next_chord):
                t_offset = t + i * 0.001
                tone = self.generate_tone(freq, t_offset, next_type)
                next_wave += tone
            
            next_wave /= len(next_chord)
            
            alpha = self.fade_progress
            current_wave = (1 - alpha) * current_wave + alpha * next_wave
        
        self.breath_phase += frames / self.sample_rate
        breath_lfo = 0.97 + 0.03 * np.sin(2 * np.pi * BREATH_CYCLE * self.breath_phase)
        
        lfo = 1.0 + 0.02 * np.sin(2 * np.pi * LFO_FREQ * t)
        
        current_wave *= breath_lfo * lfo * self.current_volume
        current_wave = self.apply_reverb(current_wave)
        
        if self.ambient_ratio > 0:
            sig = current_wave * (1 - self.ambient_ratio) * 0.9 + nature_sounds * self.ambient_ratio * 1.2
        else:
            sig = current_wave
        
        sig += bell_sound + drum_sound + handpan_sound
        
        sig *= VOLUME
        sig = np.tanh(sig * 0.8) * 0.9
        sig = sig.astype(np.float32)
        
        sig_left = sig + np.roll(sig, 2) * 0.05
        sig_right = sig + np.roll(sig, -2) * 0.05
        outdata[:] = np.column_stack([sig_left, sig_right])
        
        self.phase += frames


# Streamlit UI - Removed status update callback


def start_bgm():
    """Start BGM generation"""
    if st.session_state.is_playing:
        return
    
    # Get settings from session state
    selected_instrument = st.session_state.get('instrument_selection', 'Random')
    bell_interval = st.session_state.get('bell_interval', (15, 45))
    drum_interval = st.session_state.get('drum_interval', (8, 25))
    handpan_interval = st.session_state.get('handpan_interval', (12, 30))
    ambient_ratio = st.session_state.get('ambient_ratio', 0.3)
    
    # Select instrument
    if selected_instrument == 'Random':
        selected = random.choice(['bell', 'drum', 'handpan'])
    else:
        selected = selected_instrument.lower()
    
    # Enable only the selected instrument
    bell_enabled = (selected == 'bell')
    drum_enabled = (selected == 'drum')
    handpan_enabled = (selected == 'handpan')
    
    bell_config = InstrumentConfig(bell_interval[0], bell_interval[1], bell_enabled)
    drum_config = InstrumentConfig(drum_interval[0], drum_interval[1], drum_enabled)
    handpan_config = InstrumentConfig(handpan_interval[0], handpan_interval[1], handpan_enabled)
    
    # Create generator
    generator = MindfulnessBGM(bell_config, drum_config, handpan_config, 
                               ambient_ratio)
    
    # Set up audio stream
    sd.default.samplerate = SAMPLE_RATE
    sd.default.blocksize = BUFFER_FRAMES
    sd.default.channels = 2
    
    try:
        stream = sd.OutputStream(callback=generator.callback)
        stream.start()
        
        # Update session state
        st.session_state.generator = generator
        st.session_state.stream = stream
        st.session_state.is_playing = True
        st.session_state.selected_instrument = selected
        st.session_state.current_status = f"Playing with {selected.title()}"
    except sd.PortAudioError as e:
        # Stop generator threads
        generator.stop()
        st.session_state.current_status = "Audio device not available"
        st.error("Audio device is not available in this environment. Please run the application locally to use audio features.")
        st.warning("Streamlit Cloud does not support audio device access. To use this application with full functionality, please download and run it on your local machine.")
        st.code("""
# Installation steps for local execution:
1. Clone the repository or download the files
2. Install requirements: pip install -r requirements.txt
3. Run locally: streamlit run app.py
        """, language="bash")


def stop_bgm():
    """Stop BGM generation"""
    if not st.session_state.is_playing:
        return
    
    if st.session_state.stream:
        st.session_state.stream.stop()
        st.session_state.stream.close()
        st.session_state.stream = None
    
    if st.session_state.generator:
        st.session_state.generator.stop()
        st.session_state.generator = None
    
    st.session_state.is_playing = False
    st.session_state.current_status = "Stopped"


def main():
    st.set_page_config(
        page_title="Mindfulness BGM Generator",
        page_icon="🎵",
        layout="wide"
    )
    
    st.title("🧘‍♀️ Mindfulness BGM Generator")
    st.markdown("Dynamic ambient music generator for meditation and mindfulness practices")
    
    # Main controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("▶️ Start", use_container_width=True, disabled=st.session_state.is_playing):
            start_bgm()
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop", use_container_width=True, disabled=not st.session_state.is_playing):
            stop_bgm()
            st.rerun()
    
    with col3:
        st.write(f"**Status:** {st.session_state.current_status}")
    
    # Settings
    st.markdown("---")
    st.subheader("🎛️ Settings")
    
    # Instrument selection
    col1, col2 = st.columns(2)
    
    with col1:
        instrument_options = ['Random', 'Bell', 'Drum', 'Handpan']
        selected_instrument = st.selectbox(
            "Select Instrument",
            instrument_options,
            key='instrument_selection',
            disabled=st.session_state.is_playing
        )
    
    with col2:
        ambient_ratio = st.slider(
            "Ambient Sound Mix",
            0.0, 1.0, DEFAULT_AMBIENT_RATIO,
            key='ambient_ratio',
            disabled=st.session_state.is_playing
        )
    
    # Instrument intervals
    st.markdown("### Instrument Intervals")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Tibetan Bell**")
        bell_interval = st.slider(
            "Interval (seconds)",
            0, 60, (15, 45),
            key='bell_interval',
            disabled=st.session_state.is_playing
        )
    
    with col2:
        st.markdown("**Slit Drum**")
        drum_interval = st.slider(
            "Interval (seconds)",
            0, 60, (8, 25),
            key='drum_interval',
            disabled=st.session_state.is_playing
        )
    
    with col3:
        st.markdown("**Handpan**")
        handpan_interval = st.slider(
            "Interval (seconds)",
            0, 60, (12, 30),
            key='handpan_interval',
            disabled=st.session_state.is_playing
        )
    
    # Information
    st.markdown("---")
    st.markdown("### ℹ️ About")
    
    with st.expander("Features"):
        st.markdown("""
        - **Dynamic Sound Generation**: Continuously evolving harmonies and textures
        - **Multiple Sound Types**: Harmonic-rich tones, pure tones, soft pad, warm tones, bell-like sounds
        - **Meditation Instruments**: Tibetan bells, slit drums, handpan sounds
        - **Natural Ambient Sounds**: Ocean waves and wind
        - **Breathing Rhythm Synchronization**: Volume modulation synced to breathing patterns
        - **Mindfulness-oriented Harmonies**: Perfect fifths, fourths, and open intervals
        """)
    
    with st.expander("Instructions"):
        st.markdown("""
        1. **Select an instrument** or leave it on Random for automatic selection
        2. **Adjust the ambient sound mix** to control the balance between instruments and nature sounds
        3. **Set interval ranges** for each instrument (only the selected instrument will play)
        4. **Click Start** to begin the meditation session
        5. **Click Stop** to end the session
        
        Note: Settings cannot be changed while playing. Stop the session to make adjustments.
        """)


if __name__ == "__main__":
    main()