"""
Main BGM generator class for Mindfulness BGM Generator
"""

import numpy as np
import threading
import time
import random

from src.config import *
from src.sound_types import SoundType
from src.instruments_base import InstrumentConfig
from src.instruments import TibetanBell, SlitDrum, Handpan, CrystalSingingBowl
from src.synthesizer import Synthesizer
from src.ambient_sounds import AmbientSoundGenerator
from src.effects import AudioEffects


class MindfulnessBGM:
    """Dynamic mindfulness BGM generator class"""
    
    def __init__(self, bell_config: InstrumentConfig, drum_config: InstrumentConfig, 
                 handpan_config: InstrumentConfig, crystal_bowl_config: InstrumentConfig,
                 ambient_ratio: float):
        self.sample_rate = SAMPLE_RATE
        self.phase = 0
        self.lock = threading.Lock()
        
        # Sound type management
        self.current_sound_type = SoundType.PURE
        self.next_sound_type = None
        self.chord = Synthesizer.create_chord()
        self.next_chord = None
        
        # Volume management
        self.current_volume = 0.7
        self.target_volume = 0.7
        self.volume_transition_speed = 0.0
        
        # Phase management
        self.breath_phase = 0.0
        
        # Fade management
        self.fade_progress = 0.0
        self.is_transitioning = False
        
        # Ambient sound settings
        self.ambient_ratio = ambient_ratio
        
        # Components
        self.synthesizer = Synthesizer()
        self.ambient_generator = AmbientSoundGenerator(self.sample_rate)
        self.effects = AudioEffects(self.sample_rate)
        
        # Percussion instruments
        self.tibetan_bell = TibetanBell(self.sample_rate)
        self.slit_drum = SlitDrum(self.sample_rate)
        self.handpan = Handpan(self.sample_rate)
        self.crystal_bowl = CrystalSingingBowl(self.sample_rate)
        
        # Percussion configurations
        self.bell_config = bell_config
        self.drum_config = drum_config
        self.handpan_config = handpan_config
        self.crystal_bowl_config = crystal_bowl_config
        
        # Start schedulers
        self._start_schedulers()
    
    def _start_schedulers(self):
        """Start event and instrument schedulers"""
        # Start event scheduler
        threading.Thread(target=self._event_scheduler, daemon=True).start()
        
        # Conditionally start percussion schedulers
        if self.bell_config.enabled:
            threading.Thread(target=self._bell_scheduler, daemon=True).start()
        if self.drum_config.enabled:
            threading.Thread(target=self._drum_scheduler, daemon=True).start()
        if self.handpan_config.enabled:
            threading.Thread(target=self._handpan_scheduler, daemon=True).start()
        if self.crystal_bowl_config.enabled:
            threading.Thread(target=self._crystal_bowl_scheduler, daemon=True).start()
    
    def _bell_scheduler(self):
        """Bell scheduler"""
        while True:
            wait_time = random.uniform(self.bell_config.min_interval, 
                                     self.bell_config.max_interval)
            time.sleep(wait_time)
            
            # Strike 1-3 times
            num_strikes = random.randint(1, 3)
            for i in range(num_strikes):
                with self.lock:
                    # Select note from current chord
                    if self.chord:
                        base_freq = random.choice(self.chord)
                        # Transpose to high range for bell
                        freq = base_freq * random.choice([4, 5, 6, 8])
                    else:
                        # If no chord, use traditional method
                        freq = random.choice(BASE_FREQS) * random.choice([4, 5, 6, 8])
                    self.tibetan_bell.trigger(freq)
                
                # Interval between consecutive strikes
                if i < num_strikes - 1:
                    time.sleep(random.uniform(1.0, 4.0))

    def _drum_scheduler(self):
        """Slit drum scheduler"""
        time.sleep(5)
        while True:
            wait_time = random.uniform(self.drum_config.min_interval, 
                                     self.drum_config.max_interval)
            time.sleep(wait_time)
            
            # Strike 1-3 times
            num_strikes = random.randint(1, 3)
            for i in range(num_strikes):
                with self.lock:
                    # Select note from current chord
                    if self.chord:
                        base_freq = random.choice(self.chord)
                        # Transpose to low range for drum
                        freq = base_freq * random.choice([0.25, 0.5, 0.75])
                    else:
                        # If no chord, use traditional method
                        freq = random.choice(BASE_FREQS[:3]) * random.choice([0.5, 0.75, 1])
                    self.slit_drum.trigger(freq)
                
                # Interval between consecutive strikes
                if i < num_strikes - 1:
                    time.sleep(random.uniform(1.0, 3.0))

    def _handpan_scheduler(self):
        """Handpan scheduler"""
        time.sleep(10)
        while True:
            wait_time = random.uniform(self.handpan_config.min_interval, 
                                     self.handpan_config.max_interval)
            time.sleep(wait_time)
            
            # Strike 1-3 times
            num_strikes = random.randint(1, 3)
            for i in range(num_strikes):
                with self.lock:
                    # Select note from current chord
                    if self.chord:
                        base_freq = random.choice(self.chord)
                        # Handpan range - adjusted to mid range
                        freq = base_freq * random.choice([0.5, 1, 1.5])
                    else:
                        # If no chord, use traditional method
                        freq = random.choice(BASE_FREQS) * random.choice([0.5, 1, 1.5])
                    self.handpan.trigger(freq)
                
                # Interval between consecutive strikes
                if i < num_strikes - 1:
                    time.sleep(random.uniform(1.0, 3.5))

    def _crystal_bowl_scheduler(self):
        """Crystal singing bowl scheduler"""
        time.sleep(8)
        while True:
            wait_time = random.uniform(self.crystal_bowl_config.min_interval, 
                                     self.crystal_bowl_config.max_interval)
            time.sleep(wait_time)
            
            with self.lock:
                # Select note from current chord
                if self.chord:
                    base_freq = random.choice(self.chord)
                    # Crystal bowls in middle-high range
                    freq = base_freq * random.choice([1, 1.5, 2, 3])
                else:
                    freq = random.choice(BASE_FREQS) * random.choice([1, 2, 3])
                self.crystal_bowl.trigger(freq)

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
            self.next_sound_type = SoundType.get_random_except(self.current_sound_type)
            self.is_transitioning = True
            self.fade_progress = 0.0

    def _change_chord(self):
        """Change chord"""
        with self.lock:
            self.next_chord = Synthesizer.create_chord()
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

    def _update_transitions(self, frames: int):
        """Update volume and fade transitions"""
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

    def _generate_chord_wave(self, t: np.ndarray) -> np.ndarray:
        """Generate current chord with sound type"""
        current_wave = np.zeros(len(t))
        
        for i, freq in enumerate(self.chord):
            # Add slightly different timing to each voice (natural spread)
            t_offset = t + i * 0.001
            tone = self.synthesizer.generate_tone(freq, t_offset, self.current_sound_type)
            current_wave += tone
        
        # Normalize by number of voices
        current_wave /= len(self.chord)
        
        # Handle transition if needed
        if self.is_transitioning:
            next_wave = np.zeros(len(t))
            next_chord = self.next_chord if self.next_chord else self.chord
            next_type = self.next_sound_type if self.next_sound_type else self.current_sound_type
            
            for i, freq in enumerate(next_chord):
                t_offset = t + i * 0.001
                tone = self.synthesizer.generate_tone(freq, t_offset, next_type)
                next_wave += tone
            
            next_wave /= len(next_chord)
            
            # Smooth crossfade
            alpha = self.fade_progress
            current_wave = (1 - alpha) * current_wave + alpha * next_wave
        
        return current_wave

    def callback(self, outdata, frames, time_info, status):
        """Audio callback"""
        t = (np.arange(frames) + self.phase) / self.sample_rate
        
        # Update transitions
        self._update_transitions(frames)
        
        # Generate nature sounds
        nature_sounds = self.ambient_generator.generate_nature_sounds(frames, self.ambient_ratio)
        
        # Generate percussion sounds
        bell_sound = self.tibetan_bell.generate(frames)
        drum_sound = self.slit_drum.generate(frames)
        handpan_sound = self.handpan.generate(frames)
        crystal_bowl_sound = self.crystal_bowl.generate(frames)
        
        # Generate chord
        current_wave = self._generate_chord_wave(t)
        
        # Apply breathing rhythm
        self.breath_phase += frames / self.sample_rate
        breath_lfo = 0.97 + 0.03 * np.sin(2 * np.pi * BREATH_CYCLE * self.breath_phase)
        
        # Main LFO (slow modulation)
        lfo = 1.0 + 0.02 * np.sin(2 * np.pi * LFO_FREQ * t)
        
        # Apply LFO and volume
        current_wave *= breath_lfo * lfo * self.current_volume
        
        # Apply reverb
        current_wave = self.effects.apply_reverb(current_wave)
        
        # Mix (dynamic ambient presence)
        if self.ambient_ratio > 0:
            # Dynamic scaling based on ambient_ratio
            instrument_scale = 1.0 - (self.ambient_ratio * 0.7)  # Gentler attenuation
            ambient_scale = 1.0 + (self.ambient_ratio * 1.0)     # Gentler amplification
            
            sig = current_wave * instrument_scale + nature_sounds * self.ambient_ratio * ambient_scale
        else:
            sig = current_wave
        
        # Add percussion sounds
        sig += bell_sound + drum_sound + handpan_sound + crystal_bowl_sound
        
        # Apply final processing
        sig *= VOLUME
        sig = self.effects.apply_soft_limiting(sig, LIMITER_THRESHOLD, LIMITER_CEILING)
        sig = sig.astype(np.float32)
        
        # Apply stereo enhancement
        outdata[:] = self.effects.apply_stereo_enhancement(
            sig, STEREO_DELAY_1, STEREO_DELAY_2, STEREO_MIX_1, STEREO_MIX_2
        )
        
        self.phase += frames