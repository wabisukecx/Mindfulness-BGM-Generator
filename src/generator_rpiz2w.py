"""
Main BGM generator class for Mindfulness BGM Generator - Raspberry Pi Zero 2 W Version
Optimized for lower computational requirements
"""

import numpy as np
import threading
import time
import random

from src.config_rpiz2w import *
from src.sound_types import SoundType
from src.instruments_base import InstrumentConfig, VoicePool
from src.instruments_rpiz2w import (
    TibetanBellRPiZ2W, SlitDrumRPiZ2W, HandpanRPiZ2W, CrystalSingingBowlRPiZ2W
)
from src.synthesizer_rpiz2w import SynthesizerRPiZ2W
from src.ambient_sounds_rpiz2w import AmbientSoundGeneratorRPiZ2W
from src.effects_rpiz2w import AudioEffectsRPiZ2W


from src.schedulers import SchedulerLifecycle
from src.transitions import TransitionMixin


class MindfulnessBGMRPiZ2W(SchedulerLifecycle, TransitionMixin):
    """Optimized mindfulness BGM generator for Raspberry Pi Zero 2 W"""
    
    def __init__(self, bell_config: InstrumentConfig, drum_config: InstrumentConfig, 
                 handpan_config: InstrumentConfig, crystal_bowl_config: InstrumentConfig,
                 ambient_ratio: float, auto_start: bool = True):
        self.sample_rate = SAMPLE_RATE
        self.phase = 0
        self.lock = threading.RLock()
        self._init_schedulers()
        self.sound_start_frame = 0
        self.next_sound_start_frame = 0
        
        # Sound type management
        self.current_sound_type = SoundType.PURE
        self.next_sound_type = None
        self.chord = SynthesizerRPiZ2W.create_chord()
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
        self.fade_time = FADE_TIME
        self._init_transitions()
        
        # Ambient sound settings
        self.ambient_ratio = ambient_ratio
        
        # Components
        self.synthesizer = SynthesizerRPiZ2W()
        self.ambient_generator = AmbientSoundGeneratorRPiZ2W(self.sample_rate)
        self.effects = AudioEffectsRPiZ2W(self.sample_rate)
        
        # Percussion instruments (optimized versions)
        self.tibetan_bell = VoicePool(TibetanBellRPiZ2W, self.sample_rate)
        self.slit_drum = VoicePool(SlitDrumRPiZ2W, self.sample_rate)
        self.handpan = VoicePool(HandpanRPiZ2W, self.sample_rate)
        self.crystal_bowl = VoicePool(CrystalSingingBowlRPiZ2W, self.sample_rate)
        
        # Percussion configurations
        self.bell_config = bell_config
        self.drum_config = drum_config
        self.handpan_config = handpan_config
        self.crystal_bowl_config = crystal_bowl_config
        
        # Start schedulers
        if auto_start:
            self.start()
    
    def _bell_scheduler(self):
        """Bell scheduler"""
        while not self.stop_flag.is_set():
            wait_time = random.uniform(self.bell_config.min_interval, 
                                     self.bell_config.max_interval)
            if self.stop_flag.wait(wait_time):
                return
            
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
                    if self.stop_flag.wait(random.uniform(1.0, 4.0)):
                        return

    def _drum_scheduler(self):
        """Slit drum scheduler"""
        if self.stop_flag.wait(5):
            return
        while not self.stop_flag.is_set():
            wait_time = random.uniform(self.drum_config.min_interval, 
                                     self.drum_config.max_interval)
            if self.stop_flag.wait(wait_time):
                return
            
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
                    if self.stop_flag.wait(random.uniform(1.0, 3.0)):
                        return

    def _handpan_scheduler(self):
        """Handpan scheduler"""
        if self.stop_flag.wait(10):
            return
        while not self.stop_flag.is_set():
            wait_time = random.uniform(self.handpan_config.min_interval, 
                                     self.handpan_config.max_interval)
            if self.stop_flag.wait(wait_time):
                return
            
            # Strike 1-3 times
            num_strikes = random.randint(1, 3)
            for i in range(num_strikes):
                with self.lock:
                    # Select note from current chord
                    if self.chord:
                        base_freq = random.choice(self.chord)
                        # Handpan range
                        freq = base_freq * random.choice([0.5, 1, 1.5])
                    else:
                        # If no chord, use traditional method
                        freq = random.choice(BASE_FREQS) * random.choice([0.5, 1, 1.5])
                    self.handpan.trigger(freq)
                
                # Interval between consecutive strikes
                if i < num_strikes - 1:
                    if self.stop_flag.wait(random.uniform(1.0, 3.5)):
                        return

    def _crystal_bowl_scheduler(self):
        """Crystal singing bowl scheduler"""
        if self.stop_flag.wait(8):
            return
        while not self.stop_flag.is_set():
            wait_time = random.uniform(self.crystal_bowl_config.min_interval, 
                                     self.crystal_bowl_config.max_interval)
            if self.stop_flag.wait(wait_time):
                return
            
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
        while not self.stop_flag.is_set():
            wait_time = random.uniform(MIN_EVENT_INTERVAL, MAX_EVENT_INTERVAL)
            if self.stop_flag.wait(wait_time):
                return
            
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
        self._change(sound_type=True)

    def _change_chord(self):
        """Change chord"""
        self._change(chord=True)

    def _change(self, sound_type=False, chord=False):
        """Start one crossfade to a new timbre and/or chord"""
        with self.lock:
            if not self._ready_for_change():
                return
            if sound_type:
                self.next_sound_type = SoundType.get_random_except(self.current_sound_type)
                self.next_sound_start_frame = self.phase
            if chord:
                self.next_chord = SynthesizerRPiZ2W.create_chord()
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
        self._change(sound_type=True, chord=True)

    def _generate_chord_wave(self, t: np.ndarray) -> np.ndarray:
        """Generate current chord with sound type"""
        envelope_t = (np.arange(len(t)) + self.phase - self.sound_start_frame) / self.sample_rate
        current_wave = np.zeros(len(t))
        
        for i, freq in enumerate(self.chord):
            # Add slightly different timing to each voice
            t_offset = t + i * 0.001
            tone = self.synthesizer.generate_tone(freq, t_offset, self.current_sound_type, envelope_t=envelope_t)
            current_wave += tone
        
        # Normalize by number of voices
        current_wave /= len(self.chord)
        
        # Handle transition if needed
        if self.is_transitioning:
            next_start = self.next_sound_start_frame if self.next_sound_type else self.sound_start_frame
            next_envelope_t = (np.arange(len(t)) + self.phase - next_start) / self.sample_rate
            next_wave = np.zeros(len(t))
            next_chord = self.next_chord if self.next_chord else self.chord
            next_type = self.next_sound_type if self.next_sound_type else self.current_sound_type
            
            for i, freq in enumerate(next_chord):
                t_offset = t + i * 0.001
                tone = self.synthesizer.generate_tone(freq, t_offset, next_type, envelope_t=next_envelope_t)
                next_wave += tone
            
            next_wave /= len(next_chord)
            
            # Smooth crossfade, interpolated sample by sample
            alpha = self._crossfade_amount(len(t))
            current_wave = (1 - alpha) * current_wave + alpha * next_wave
        
        return current_wave

    def callback(self, outdata, frames, time_info, status):
        """Audio callback"""
        with self.lock:
            self._render_audio(outdata, frames)

    def _render_audio(self, outdata, frames):
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
        
        # Breathing rhythm, main LFO (slow modulation) and volume, per sample
        current_wave *= self._level_envelope(frames, BREATH_CYCLE) * (1.0 + 0.02 * np.sin(2 * np.pi * LFO_FREQ * t))
        
        # Apply reverb
        current_wave = self.effects.apply_reverb(current_wave)
        
        # Mix (dynamic ambient presence)
        if self.ambient_ratio > 0:
            # Dynamic scaling based on ambient_ratio
            instrument_scale = 1.0 - (self.ambient_ratio * 0.7)
            ambient_scale = 1.0
            
            sig = current_wave * instrument_scale + nature_sounds * self.ambient_ratio * ambient_scale
        else:
            sig = current_wave
        
        # Add percussion sounds
        sig += bell_sound + drum_sound + handpan_sound + crystal_bowl_sound
        
        # Apply final processing
        sig *= VOLUME
        sig = self.effects.apply_soft_limiting(sig)
        sig = sig.astype(np.float32)
        
        # Apply stereo effect
        outdata[:] = self.effects.create_stereo(sig, STEREO_DELAY, STEREO_MIX)
        
        self.phase += frames
