"""
Main BGM generator class for Mindfulness BGM Generator
Enhanced with extended harmonic progression and chord variations
"""

import numpy as np
import threading
import time
import random

from src.config import *
from src.sound_types import SoundType
from src.instruments_base import InstrumentConfig, VoicePool
from src.instruments import TibetanBell, SlitDrum, Handpan, CrystalSingingBowl
from src.synthesizer import Synthesizer
from src.ambient_sounds import AmbientSoundGenerator
from src.effects import AudioEffects
from src.meditation import SmokeDrone, ShoVoice, BreathingSpace


from src.schedulers import SchedulerLifecycle
from src.transitions import TransitionMixin


class MindfulnessBGM(SchedulerLifecycle, TransitionMixin):
    """Dynamic mindfulness BGM generator class with enhanced harmonic features"""
    
    def __init__(self, bell_config: InstrumentConfig, drum_config: InstrumentConfig, 
                 handpan_config: InstrumentConfig, crystal_bowl_config: InstrumentConfig,
                 ambient_ratio: float, modulation_mode: str = "balanced",
                 harmonic_richness: str = "normal", chord_variety: str = "normal",
                 progression_style: str = "circle-of-fifths", session_evolution: bool = False,
                 binaural_enhancement: bool = False, auto_start: bool = True,
                 meditation_mode: bool = False, meditation_voice: str = "drone"):
        self.sample_rate = SAMPLE_RATE
        self.meditation_mode = meditation_mode
        self.fade_time = 12.0 if meditation_mode else FADE_TIME
        self.phase = 0
        self.lock = threading.RLock()
        self._init_schedulers()
        self.sound_start_frame = 0
        self.next_sound_start_frame = 0
        
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
        self._init_transitions()
        
        # Ambient sound settings
        self.ambient_ratio = ambient_ratio
        
        # Extended chord and chord progression settings
        self.harmonic_richness = harmonic_richness
        self.chord_variety = chord_variety
        self.progression_style = progression_style
        self.session_evolution = session_evolution
        self.binaural_enhancement = binaural_enhancement
        
        # Components
        self.synthesizer = Synthesizer()
        self.ambient_generator = AmbientSoundGenerator(self.sample_rate)
        self.effects = AudioEffects(self.sample_rate)
        
        # Percussion instruments
        self.tibetan_bell = VoicePool(TibetanBell, self.sample_rate)
        self.slit_drum = VoicePool(SlitDrum, self.sample_rate)
        self.handpan = VoicePool(Handpan, self.sample_rate)
        self.crystal_bowl = VoicePool(CrystalSingingBowl, self.sample_rate)
        
        # Percussion configurations
        self.bell_config = bell_config
        self.drum_config = drum_config
        self.handpan_config = handpan_config
        self.crystal_bowl_config = crystal_bowl_config
        self.percussion_gain = 1.0
        self.percussion_silent = True

        # Enhanced modulation system
        self.modulation_mode = modulation_mode
        self.modulation_chances = {
            "stable": 0.1,    # 10% - minimal modulation
            "balanced": 0.3,  # 30% - balanced
            "dynamic": 0.5    # 50% - active changes
        }
        
        # Modulation state tracking
        self.current_root = self._extract_root(self.chord)
        self.preparation_chord = None
        self.is_preparing_modulation = False
        self.modulation_target = None  # Root announced by the bridge chord
        self.session_start = time.monotonic()
        self.modulation_phase = 0
        
        # Parameters for session evolution
        self.evolution_progress = 0.0  # 0.0 (start) to 1.0 (fully evolved)
        self.evolution_target = random.uniform(0.4, 0.6)  # Evolution target value
        
        # Chord progression tracking in healing mode
        self.progression_history = []
        self.progression_position = 0
        
        # Set chord generation method before start
        if self.progression_style == "modal":
            self.chord = self.synthesizer.create_modal_chord(mode=self.modulation_mode)
        elif self.progression_style == "pentatonic":
            self.chord = self.synthesizer.create_pentatonic_chord(mode=self.modulation_mode)
        else:
            self.chord = self.synthesizer.create_chord(mode=self.modulation_mode, chord_variety=self.chord_variety)
        
        self.current_root = self._extract_root(self.chord)
        if self.meditation_mode:
            self.chord = [110.0, 165.0]
            self.current_root = 110.0
            self.current_volume = self.target_volume = 0.5
            if meditation_voice == "sho":
                self.drone = ShoVoice(self.sample_rate)
            elif meditation_voice == "drone":
                self.drone = SmokeDrone(self.chord, self.sample_rate)
            else:
                raise ValueError(f"Unknown meditation voice: {meditation_voice}")
            self.space = BreathingSpace()

        # Start schedulers
        if auto_start:
            self.start()
    
    def _extract_root(self, chord):
        """Extract root note from chord"""
        if chord:
            return min(chord)  # Lowest frequency is typically the root
        return None
    
    def _get_binaural_factor(self, channel):
        """Get frequency shift factor for binaural enhancement"""
        if not self.binaural_enhancement:
            return 1.0
        
        return 1.0 + (-0.0008 if channel == 0 else 0.0008)

    def _create_bridge_chord(self, current_root, target_root):
        """Create a transitional chord between two roots"""
        if current_root is None or target_root is None:
            return self.synthesizer.create_chord(chord_variety=self.chord_variety)
        
        # Create a chord that contains notes from both keys
        # This creates a smoother transition
        bridge_intervals = [0, 7]  # Perfect fifth
        
        # Add a note that connects both keys
        if abs(current_root - target_root) > 100:  # Large interval
            bridge_intervals.append(5)  # Add fourth
        
        # Add extra notes depending on chord variety
        if self.chord_variety == "extended":
            bridge_intervals.extend([2, 9])  # Add 2nd and 9th for extended variety
        
        frequencies = []
        for interval in bridge_intervals:
            freq = current_root * (2 ** (interval / 12))
            frequencies.append(freq)
        
        return frequencies
    
    def _evolution_scheduler(self):
        """Session evolution scheduler - music slowly changes over time"""
        # Gradually changes over 30 minutes
        evolution_duration = 1800  # 30 minutes
        
        start_time = time.monotonic()
        while not self.stop_flag.is_set():
            elapsed = time.monotonic() - start_time
            
            # Calculate evolution progress (0.0 to 1.0)
            self.evolution_progress = min(1.0, elapsed / evolution_duration)
            
            # Wait at each evolution step (every 1%)
            if self.stop_flag.wait(evolution_duration / 100):
                return
            
            # Stop when evolution is complete
            if self.evolution_progress >= 1.0:
                break
    
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
                        # Handpan range - adjusted to mid range
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
        """Enhanced event scheduler with natural modulation patterns"""
        while not self.stop_flag.is_set():
            wait_time = (random.uniform(45.0, 90.0) if self.meditation_mode else
                         random.uniform(MIN_EVENT_INTERVAL, MAX_EVENT_INTERVAL))
            if self.stop_flag.wait(wait_time):
                return
            if self.meditation_mode:
                # Keep the low open fifth steady; only a slow, small level drift.
                with self.lock:
                    self.target_volume = random.uniform(0.45, 0.55)
                    self.volume_transition_speed = 1.0 / (random.uniform(20, 40) * self.sample_rate)
                continue
            
            # Time-based modulation probability
            elapsed = time.monotonic() - self.session_start
            
            # Wave-like pattern for modulation probability
            self.modulation_phase += 0.1
            wave = 0.5 + 0.3 * np.sin(self.modulation_phase)
            
            # Base probability from mode
            base_probability = self.modulation_chances[self.modulation_mode]
            
            # Probability adjustment by session evolution
            if self.session_evolution:
                if self.evolution_progress < self.evolution_target:
                    # First half: less change
                    modulation_probability = base_probability * (0.5 + 0.5 * self.evolution_progress / self.evolution_target)
                else:
                    # Latter half: more change, then gradually calms down
                    progress_after_target = (self.evolution_progress - self.evolution_target) / (1.0 - self.evolution_target)
                    modulation_probability = base_probability * (1.2 - 0.4 * progress_after_target)
            else:
                # Time decay for "stable" mode
                if self.modulation_mode == "stable":
                    time_decay = np.exp(-elapsed / 1200)  # 20 minutes half-life
                    modulation_probability = base_probability * time_decay
                else:
                    # Wave pattern for balanced and dynamic modes
                    modulation_probability = base_probability * wave
                    
                    # Gradual calming effect for all modes
                    calming_factor = max(0.3, np.exp(-elapsed / 1800))  # 30 minutes
                    modulation_probability *= calming_factor
            
            # Choose event type
            if random.random() < modulation_probability:
                self._change_chord()
            else:
                # Other changes remain frequent
                events = [
                    self._change_sound_type,
                    self._change_volume,
                    self._change_sound_type  # Weighted towards sound type changes
                ]
                event = random.choice(events)
                event()

    def _change_sound_type(self):
        """Change sound type"""
        with self.lock:
            if not self._ready_for_change():
                return
            self.next_sound_type = SoundType.get_random_except(self.current_sound_type)
            self.next_sound_start_frame = self.phase
            self.is_transitioning = True
            self.fade_progress = 0.0

    def _change_chord(self):
        """Enhanced chord change with smoother transitions and progression styles"""
        with self.lock:
            if not self._ready_for_change():
                return
            if not self.is_preparing_modulation and self.modulation_mode != "dynamic":
                # Preparation stage: create bridge chord
                if self.progression_style == "circle-of-fifths":
                    new_root = self.synthesizer.select_next_root(self.current_root, self.modulation_mode)
                elif self.progression_style == "pentatonic":
                    new_root = self.synthesizer.select_next_pentatonic_root(self.current_root)
                elif self.progression_style == "modal":
                    new_root = self.synthesizer.select_next_modal_root(self.current_root)
                else:  # random
                    new_root = random.choice(BASE_FREQS)
                
                self.preparation_chord = self._create_bridge_chord(self.current_root, new_root)
                self.is_preparing_modulation = True
                self.modulation_target = new_root
                self.next_chord = self.preparation_chord
            else:
                # Actual modulation (to the root the bridge prepared, if any)
                # or direct change for dynamic mode
                target = self.modulation_target
                if self.progression_style == "circle-of-fifths":
                    self.next_chord = self.synthesizer.create_chord(
                        previous_root=self.current_root,
                        mode=self.modulation_mode,
                        chord_variety=self.chord_variety,
                        root=target
                    )
                elif self.progression_style == "pentatonic":
                    self.next_chord = self.synthesizer.create_pentatonic_chord(
                        previous_root=self.current_root,
                        mode=self.modulation_mode,
                        root=target
                    )
                elif self.progression_style == "modal":
                    self.next_chord = self.synthesizer.create_modal_chord(
                        previous_root=self.current_root,
                        mode=self.modulation_mode,
                        root=target
                    )
                else:  # random
                    self.next_chord = self.synthesizer.create_chord(
                        chord_variety=self.chord_variety,
                        root=target
                    )
                
                self.is_preparing_modulation = False
                self.modulation_target = None
                self.current_root = self._extract_root(self.next_chord)
                
                # Add to progression history
                if len(self.progression_history) > 10:
                    self.progression_history.pop(0)  # Remove old chord
                self.progression_history.append(self.next_chord)
            
            self.is_transitioning = True
            self.fade_progress = 0.0

    def _change_volume(self):
        """Change volume"""
        with self.lock:
            self.target_volume = random.uniform(0.4, 0.8)
            duration = random.uniform(MIN_VOLUME_CHANGE_DURATION, MAX_VOLUME_CHANGE_DURATION)
            self.volume_transition_speed = 1.0 / (duration * self.sample_rate)

    def _generate_chord_wave(self, t: np.ndarray) -> np.ndarray:
        """Generate current chord with sound type"""
        if self.meditation_mode:
            # The quiet presets keep one chord; its timbre drifts like smoke
            # and it recedes now and then into near-silence
            return self.drone.render(t) * self.space.gain(t)
        envelope_t = (np.arange(len(t)) + self.phase - self.sound_start_frame) / self.sample_rate
        current_wave = np.zeros(len(t))
        
        for i, freq in enumerate(self.chord):
            # Add slightly different timing to each voice (natural spread)
            t_offset = t + i * 0.001
            tone = self.synthesizer.generate_tone(freq, t_offset, self.current_sound_type, self.harmonic_richness,
                                                   envelope_t=envelope_t, sample_rate=self.sample_rate)
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
                tone = self.synthesizer.generate_tone(freq, t_offset, next_type, self.harmonic_richness,
                                                       envelope_t=next_envelope_t, sample_rate=self.sample_rate)
                next_wave += tone
            
            next_wave /= len(next_chord)
            
            # Smooth crossfade, interpolated sample by sample
            alpha = self._crossfade_amount(len(t))
            current_wave = (1 - alpha) * current_wave + alpha * next_wave
        
        return current_wave

    def _percussion_gain(self, main_rms: float, perc_rms: float, frames: int) -> np.ndarray:
        """Per-sample gain keeping percussion at about half the drone level.

        The gain ramps within each block instead of stepping between blocks
        (which clicks). While a sound rings it may only fall, so decays keep
        their natural shape rather than being pulled back up; it recovers
        only during silence, where the change is inaudible.
        """
        if perc_rms <= 0.001:
            start, end = self.percussion_gain, 1.0
            self.percussion_silent = True
        else:
            target = min(1.0, main_rms / (perc_rms * 2))
            # An attack out of silence may take its gain at once: the
            # preceding samples are effectively zero, so nothing clicks.
            start = target if self.percussion_silent else self.percussion_gain
            end = min(start, target)
            self.percussion_silent = False
        self.percussion_gain = end
        return np.linspace(start, end, frames + 1)[1:]

    def callback(self, outdata, frames, time_info, status):
        """Audio callback with improved audio processing"""
        with self.lock:
            self._render_audio(outdata, frames)

    def _render_audio(self, outdata, frames):
        t = (np.arange(frames) + self.phase) / self.sample_rate
        
        # Update transitions
        self._update_transitions(frames)
        
        # Generate nature sounds
        nature_sounds = self.ambient_generator.generate_nature_sounds(frames, self.ambient_ratio)
        
        # Generate percussion sounds with individual levels
        bell_sound = self.tibetan_bell.generate(frames) * 0.9
        drum_sound = self.slit_drum.generate(frames) * 0.8
        handpan_sound = self.handpan.generate(frames) * 0.85
        crystal_bowl_sound = self.crystal_bowl.generate(frames) * 0.9
        
        # Generate chord
        if self.binaural_enhancement:
            current_wave = np.column_stack([
                self._generate_chord_wave(t * self._get_binaural_factor(channel))
                for channel in (0, 1)
            ])
            nature_sounds = nature_sounds[:, None]
        else:
            current_wave = self._generate_chord_wave(t)
        
        # Breathing rhythm, main LFO (slow modulation) and volume, per sample
        envelope = self._level_envelope(frames, BREATH_CYCLE) * (1.0 + 0.02 * np.sin(2 * np.pi * LFO_FREQ * t))
        if self.binaural_enhancement:
            envelope = envelope[:, None]
        current_wave *= envelope
        
        # Apply reverb
        current_wave = self.effects.apply_reverb(current_wave)
        
        # Mix with headroom management
        if self.ambient_ratio > 0:
            # Dynamic scaling based on ambient_ratio
            instrument_scale = 1.0 - (self.ambient_ratio * 0.7)
            ambient_scale = 1.0
            
            sig = current_wave * instrument_scale + nature_sounds * self.ambient_ratio * ambient_scale
        else:
            sig = current_wave
        
        # Add percussion sounds with automatic gain compensation
        percussion_mix = bell_sound + drum_sound + handpan_sound + crystal_bowl_sound

        # RMS-based automatic gain control, applied sample by sample
        main_rms = np.sqrt(np.mean(sig ** 2)) if len(sig) > 0 else 0.001
        perc_rms = np.sqrt(np.mean(percussion_mix ** 2)) if len(percussion_mix) > 0 else 0.001
        percussion_mix = percussion_mix * self._percussion_gain(main_rms, perc_rms, frames)
        if self.binaural_enhancement:
            percussion_mix = percussion_mix[:, None]

        sig += percussion_mix
        
        # Apply final processing with improved limiting
        sig *= VOLUME * HEADROOM_LINEAR  # Ensure headroom
        sig = self.effects.apply_soft_limiting(sig, LIMITER_THRESHOLD, LIMITER_CEILING, LIMITER_KNEE)
        sig = sig.astype(np.float32)
        
        # Apply stereo enhancement with optional binaural enhancement
        if self.binaural_enhancement:
            out_stereo = sig
        else:
            out_stereo = self.effects.apply_stereo_enhancement(
                sig, STEREO_DELAY_1, STEREO_DELAY_2, STEREO_MIX_1, STEREO_MIX_2
            )
        
        outdata[:] = out_stereo
        
        self.phase += frames
