import numpy as np
import sounddevice as sd
import threading
import time
import random
import argparse
from typing import List, Tuple
from enum import Enum

# 設定
SAMPLE_RATE = 16000
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

class SoundType(Enum):
    HARMONIC = "harmonic"
    PURE = "pure"
    SOFT_PAD = "soft_pad"
    WARM = "warm"
    BELL_LIKE = "bell_like"

class InstrumentConfig:
    def __init__(self, min_interval: float, max_interval: float, enabled: bool = True):
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.enabled = enabled

class SlitDrum:
    MIN_FREQ = 100
    MAX_FREQ = 500
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.is_playing = False
        self.drum_phase = 0
        self.drum_start_time = 0
        self.current_freq = 0

    def trigger(self, freq: float):
        if freq < self.MIN_FREQ or freq > self.MAX_FREQ:
            return
        self.is_playing = True
        self.drum_phase = 0
        self.drum_start_time = 0
        self.current_freq = freq

    def generate(self, frames: int) -> np.ndarray:
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
    MIN_FREQ = 147
    MAX_FREQ = 587
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.is_playing = False
        self.pan_phase = 0
        self.pan_start_time = 0
        self.current_freq = 0

    def trigger(self, freq: float):
        if freq < self.MIN_FREQ or freq > self.MAX_FREQ:
            return
        self.is_playing = True
        self.pan_phase = 0
        self.pan_start_time = 0
        self.current_freq = freq

    def generate(self, frames: int) -> np.ndarray:
        if not self.is_playing:
            return np.zeros(frames)
        t = (np.arange(frames) + self.pan_phase) / self.sample_rate
        pitch_bend = np.exp(-15 * t) * 0.005 + 1
        fundamental = np.sin(2 * np.pi * self.current_freq * t * pitch_bend)
        second = np.sin(2 * np.pi * self.current_freq * 2.02 * t) * 0.3
        third = np.sin(2 * np.pi * self.current_freq * 2.99 * t) * 0.15
        fourth = np.sin(2 * np.pi * self.current_freq * 3.98 * t) * 0.08
        resonance_freq = self.current_freq * 1.51
        resonance_mod = 1 + 0.1 * np.exp(-5 * t)
        metallic = np.sin(2 * np.pi * resonance_freq * t * resonance_mod + np.pi/4) * 0.2
        sympathetic = np.sin(2 * np.pi * self.current_freq * 5.1 * t) * 0.05 * np.exp(-2 * t)
        strike = np.exp(-30 * t)
        bloom = 1 - np.exp(-5 * t)
        decay = np.exp(-2.5 * t)
        sustain = np.exp(-0.8 * t)
        envelope = strike * 0.3 + bloom * decay * 0.5 + sustain * 0.2
        beating = 1 + 0.02 * np.sin(2 * np.pi * 2 * t) * np.exp(-3 * t)
        pan_sound = (fundamental + second + third + fourth + metallic + sympathetic) * envelope * beating * 0.28
        self.pan_phase += frames
        self.pan_start_time += frames / self.sample_rate
        if self.pan_start_time > 3.5 or np.max(np.abs(pan_sound)) < 0.0005:
            self.is_playing = False
        return pan_sound

class CrystalSingingBowl:
    MIN_FREQ = 200
    MAX_FREQ = 2000
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.is_playing = False
        self.crystal_phase = 0
        self.crystal_start_time = 0
        self.current_freq = 0
        self.rubbing_phase = 0

    def trigger(self, freq: float):
        if freq < self.MIN_FREQ or freq > self.MAX_FREQ:
            return
        self.is_playing = True
        self.crystal_phase = 0
        self.crystal_start_time = 0
        self.current_freq = freq
        self.rubbing_phase = np.random.uniform(0, 2*np.pi)

    def generate(self, frames: int) -> np.ndarray:
        if not self.is_playing:
            return np.zeros(frames)
        t = (np.arange(frames) + self.crystal_phase) / self.sample_rate
        fm_depth1 = 0.0001
        fm_depth2 = 0.00005
        fm_freq1 = 0.2
        fm_freq2 = 0.5
        freq_mod1 = np.sin(2 * np.pi * fm_freq1 * t) * fm_depth1
        freq_mod2 = np.sin(2 * np.pi * fm_freq2 * t + self.rubbing_phase) * fm_depth2
        freq_final = self.current_freq * (1 + freq_mod1 + freq_mod2)
        fundamental = np.sin(2 * np.pi * freq_final * t)
        second = np.sin(2 * np.pi * self.current_freq * 2 * t + np.pi/4) * 0.02
        third = np.sin(2 * np.pi * self.current_freq * 3 * t + np.pi/3) * 0.01
        ring_freq = self.current_freq * 7.1
        ring = np.sin(2 * np.pi * ring_freq * t) * 0.005
        am_depth1 = 0.005
        am_depth2 = 0.002
        am_freq1 = 3.7
        am_freq2 = 5.3
        amplitude_mod = 1 + am_depth1 * np.sin(2 * np.pi * am_freq1 * t)
        amplitude_mod += am_depth2 * np.sin(2 * np.pi * am_freq2 * t + np.pi/2)
        t_fade = t + self.crystal_phase / self.sample_rate
        rub_attack = 1 - np.exp(-2 * t_fade)
        strike_attack = np.exp(-20 * t_fade)
        decay = np.exp(-0.4 * t_fade)
        envelope = rub_attack * 0.7 + strike_attack * 0.3
        envelope *= decay
        final_fade = np.exp(-0.2 * t_fade)
        envelope *= final_fade
        noise = np.random.randn(frames) * 0.0005 * np.exp(-5 * t_fade)
        crystal_sound = (fundamental + second + third + ring) * envelope * amplitude_mod * 0.35 + noise
        self.crystal_phase += frames
        self.crystal_start_time += frames / self.sample_rate
        if self.crystal_start_time > 12.0 or np.max(np.abs(crystal_sound)) < 0.0005:
            self.is_playing = False
        return crystal_sound

class TibetanBell:
    MIN_FREQ = 1000
    MAX_FREQ = 8000
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.is_playing = False
        self.bell_phase = 0
        self.bell_start_time = 0
        self.current_freq = 2000

    def trigger(self, freq: float = None):
        if freq and (freq < self.MIN_FREQ or freq > self.MAX_FREQ):
            return
        self.current_freq = freq if freq else self.current_freq
        self.is_playing = True
        self.bell_phase = 0
        self.bell_start_time = 0

    def generate(self, frames: int) -> np.ndarray:
        if not self.is_playing:
            return np.zeros(frames)
        t = (np.arange(frames) + self.bell_phase) / self.sample_rate
        f1 = self.current_freq
        f2 = self.current_freq * 1.02
        f3 = self.current_freq * 2.76
        f4 = self.current_freq * 5.43
        wave1 = np.sin(2 * np.pi * f1 * t)
        wave2 = np.sin(2 * np.pi * f2 * t) * 0.8
        wave3 = np.sin(2 * np.pi * f3 * t) * 0.3
        wave4 = np.sin(2 * np.pi * f4 * t) * 0.1
        t_fade = t + self.bell_phase / self.sample_rate
        attack = np.exp(-100 * t_fade)
        decay1 = np.exp(-5 * t_fade)
        decay2 = np.exp(-1.5 * t_fade)
        envelope = attack * 0.5 + decay1 * 0.3 + decay2 * 0.2
        final_fade = np.exp(-0.8 * t_fade)
        envelope *= final_fade
        am_depth = 0.015 * np.exp(-2 * t_fade)
        am_freq = 8
        amplitude_mod = 1 + am_depth * np.sin(2 * np.pi * am_freq * t)
        bell_sound = (wave1 + wave2 + wave3 + wave4) * envelope * amplitude_mod * 0.2
        self.bell_phase += frames
        self.bell_start_time += frames / self.sample_rate
        if self.bell_start_time > 4.0 or np.max(np.abs(bell_sound)) < 0.001:
            self.is_playing = False
        return bell_sound

class MindfulnessBGM:
    def __init__(self, bell_config: InstrumentConfig, drum_config: InstrumentConfig,
                 handpan_config: InstrumentConfig, crystal_bowl_config: InstrumentConfig,
                 ambient_ratio: float):
        self.sample_rate = SAMPLE_RATE
        self.phase = 0
        self.lock = threading.Lock()
        self.current_sound_type = SoundType.PURE
        self.next_sound_type = None
        self.chord = self._create_chord()
        self.next_chord = None
        self.current_volume = 0.7
        self.target_volume = 0.7
        self.volume_transition_speed = 0.0
        self.noise_state = 0.0
        self.breath_phase = 0.0
        self.fade_progress = 0.0
        self.is_transitioning = False
        self.ambient_ratio = ambient_ratio
        self.ocean_phase = np.random.uniform(0, 2*np.pi)
        self.ocean_variation = np.random.uniform(0.8, 1.2)
        self.tibetan_bell = TibetanBell(self.sample_rate)
        self.slit_drum = SlitDrum(self.sample_rate)
        self.handpan = Handpan(self.sample_rate)
        self.crystal_bowl = CrystalSingingBowl(self.sample_rate)
        self.bell_config = bell_config
        self.drum_config = drum_config
        self.handpan_config = handpan_config
        self.crystal_bowl_config = crystal_bowl_config
        self.reverb_buffer = np.zeros(int(self.sample_rate * 0.1))
        self.reverb_index = 0
        threading.Thread(target=self._event_scheduler, daemon=True).start()
        if self.bell_config.enabled:
            threading.Thread(target=self._bell_scheduler, daemon=True).start()
        if self.drum_config.enabled:
            threading.Thread(target=self._drum_scheduler, daemon=True).start()
        if self.handpan_config.enabled:
            threading.Thread(target=self._handpan_scheduler, daemon=True).start()
        if self.crystal_bowl_config.enabled:
            threading.Thread(target=self._crystal_bowl_scheduler, daemon=True).start()

    def _bell_scheduler(self):
        while True:
            wait_time = random.uniform(self.bell_config.min_interval, self.bell_config.max_interval)
            time.sleep(wait_time)
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
                    time.sleep(random.uniform(1.0, 4.0))

    def _drum_scheduler(self):
        time.sleep(5)
        while True:
            wait_time = random.uniform(self.drum_config.min_interval, self.drum_config.max_interval)
            time.sleep(wait_time)
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
                    time.sleep(random.uniform(1.0, 3.0))

    def _handpan_scheduler(self):
        time.sleep(10)
        while True:
            wait_time = random.uniform(self.handpan_config.min_interval, self.handpan_config.max_interval)
            time.sleep(wait_time)
            num_strikes = random.randint(1, 3)
            for i in range(num_strikes):
                with self.lock:
                    if self.chord:
                        base_freq = random.choice(self.chord)
                        freq = base_freq * random.choice([0.5, 1, 1.5])
                    else:
                        freq = random.choice(BASE_FREQS) * random.choice([0.5, 1, 1.5])
                    self.handpan.trigger(freq)
                if i < num_strikes - 1:
                    time.sleep(random.uniform(1.0, 3.5))

    def _crystal_bowl_scheduler(self):
        time.sleep(8)
        while True:
            wait_time = random.uniform(self.crystal_bowl_config.min_interval, self.crystal_bowl_config.max_interval)
            time.sleep(wait_time)
            with self.lock:
                if self.chord:
                    base_freq = random.choice(self.chord)
                    freq = base_freq * random.choice([1, 1.5, 2, 3])
                else:
                    freq = random.choice(BASE_FREQS) * random.choice([1, 2, 3])
                self.crystal_bowl.trigger(freq)

    def _event_scheduler(self):
        while True:
            wait_time = random.uniform(MIN_EVENT_INTERVAL, MAX_EVENT_INTERVAL)
            time.sleep(wait_time)
            events = [self._change_sound_type, self._change_chord, self._change_volume, self._change_both]
            event = random.choice(events)
            event()

    def _change_sound_type(self):
        with self.lock:
            sound_types = list(SoundType)
            sound_types.remove(self.current_sound_type)
            self.next_sound_type = random.choice(sound_types)
            self.is_transitioning = True
            self.fade_progress = 0.0

    def _change_chord(self):
        with self.lock:
            self.next_chord = self._create_chord()
            self.is_transitioning = True
            self.fade_progress = 0.0

    def _change_volume(self):
        with self.lock:
            self.target_volume = random.uniform(0.4, 0.8)
            duration = random.uniform(MIN_VOLUME_CHANGE_DURATION, MAX_VOLUME_CHANGE_DURATION)
            self.volume_transition_speed = 1.0 / (duration * self.sample_rate)

    def _change_both(self):
        self._change_sound_type()
        self._change_chord()

    def _create_chord(self) -> List[float]:
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
        output = np.copy(signal)
        for i in range(len(signal)):
            reverb_sample = self.reverb_buffer[self.reverb_index] * 0.2
            output[i] += reverb_sample
            self.reverb_buffer[self.reverb_index] = signal[i]
            self.reverb_index = (self.reverb_index + 1) % len(self.reverb_buffer)
        return output

    def _generate_ocean_waves(self, t: np.ndarray, frames: int) -> np.ndarray:
        t_ocean = t + self.ocean_phase
        var = self.ocean_variation
        wave1 = np.sin(2 * np.pi * 0.08 * var * t_ocean) * 0.8
        wave2 = np.sin(2 * np.pi * 0.06 * var * t_ocean + np.pi/3) * 0.7
        wave3 = np.sin(2 * np.pi * 0.04 * var * t_ocean + np.pi/6) * 0.6
        wave4 = np.sin(2 * np.pi * 0.03 * var * t_ocean + np.pi/4) * 0.5
        swell = 0.6 + 0.4 * np.sin(2 * np.pi * 0.02 * var * t_ocean)
        main_waves = (wave1 + wave2 + wave3 + wave4) * swell
        breaking_trigger = np.sin(2 * np.pi * 0.05 * var * t_ocean)
        breaking_envelope = np.maximum(0, breaking_trigger) ** 2
        breaking_noise = np.random.randn(frames) * 0.05
        for i in range(1, frames):
            breaking_noise[i] = 0.7 * breaking_noise[i-1] + 0.3 * breaking_noise[i]
        breaking_waves = breaking_noise * breaking_envelope * 0.3
        deep_waves = np.sin(2 * np.pi * 0.015 * t_ocean) * 0.5
        deep_waves += np.sin(2 * np.pi * 0.01 * t_ocean + np.pi/2) * 0.4
        deep_waves += np.sin(2 * np.pi * 0.007 * t_ocean + np.pi/3) * 0.3
        wash_envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.03 * t_ocean + np.pi)
        wash_sound = np.sin(2 * np.pi * 0.025 * t_ocean) * wash_envelope * 0.3
        ocean_sound = main_waves + breaking_waves + deep_waves + wash_sound
        for i in range(1, frames):
            ocean_sound[i] = 0.5 * ocean_sound[i-1] + 0.5 * ocean_sound[i]
        return ocean_sound

    def generate_nature_sounds(self, frames: int) -> np.ndarray:
        if self.ambient_ratio == 0:
            return np.zeros(frames)
        t = (np.arange(frames) + self.noise_state) / self.sample_rate
        ocean = self._generate_ocean_waves(t, frames)
        nature_mix = ocean
        for i in range(1, frames):
            nature_mix[i] = 0.7 * nature_mix[i-1] + 0.3 * nature_mix[i]
        self.noise_state += frames
        return nature_mix * 1.5

    def callback(self, outdata, frames, time_info, status):
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
            crystal_bowl_sound = self.crystal_bowl.generate(frames)
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
                instrument_scale = 1.0 - (self.ambient_ratio * 0.7)
                ambient_scale = 1.0 + (self.ambient_ratio * 1.0)
                sig = current_wave * instrument_scale + nature_sounds * self.ambient_ratio * ambient_scale
            else:
                sig = current_wave
            sig += bell_sound + drum_sound + handpan_sound + crystal_bowl_sound
            sig *= VOLUME
            sig = np.tanh(sig * 0.8) * 0.9
            sig = sig.astype(np.float32)
            sig_left = sig + np.roll(sig, 2) * 0.05
            sig_right = sig + np.roll(sig, -2) * 0.05
            outdata[:] = np.column_stack([sig_left, sig_right])
            self.phase += frames

def parse_instrument_interval(value: str) -> Tuple[float, float]:
    if value == "0":
        return 0, 0
    if "-" in value:
        min_val, max_val = value.split("-")
        return float(min_val), float(max_val)
    else:
        val = float(value)
        return val * 0.8, val * 1.2

def parse_ambient_value(value: str) -> float:
    val = float(value)
    return max(0.0, min(1.0, val))

def main():
    parser = argparse.ArgumentParser(description="Mindfulness BGM Generator - Modified Version")
    parser.add_argument("--bell", type=str, default="15-45", help="Tibetan bell interval in seconds")
    parser.add_argument("--drum", type=str, default="8-25", help="Slit drum interval in seconds")
    parser.add_argument("--handpan", type=str, default="12-30", help="Handpan interval in seconds")
    parser.add_argument("--crystal-bowl", type=str, default="25-60", help="Crystal singing bowl interval in seconds")
    parser.add_argument("--ambient", type=str, default=str(DEFAULT_AMBIENT_RATIO), help="Ambient sound ratio (0-1)")
    parser.add_argument("--instrument", type=str, default=None, help="Specify instrument")
    args = parser.parse_args()
    bell_min, bell_max = parse_instrument_interval(args.bell)
    drum_min, drum_max = parse_instrument_interval(args.drum)
    handpan_min, handpan_max = parse_instrument_interval(args.handpan)
    crystal_bowl_min, crystal_bowl_max = parse_instrument_interval(args.crystal_bowl)
    instruments = ['bell', 'drum', 'handpan', 'crystal-bowl']
    if args.instrument:
        selected_instrument = args.instrument.lower()
        if selected_instrument not in instruments:
            print(f"Invalid instrument: {selected_instrument}")
            print(f"Available instruments: {', '.join(instruments)}")
            return
    else:
        selected_instrument = random.choice(instruments)
    bell_enabled = (selected_instrument == 'bell' and bell_min > 0)
    drum_enabled = (selected_instrument == 'drum' and drum_min > 0)
    handpan_enabled = (selected_instrument == 'handpan' and handpan_min > 0)
    crystal_bowl_enabled = (selected_instrument == 'crystal-bowl' and crystal_bowl_min > 0)
    bell_config = InstrumentConfig(bell_min, bell_max, bell_enabled)
    drum_config = InstrumentConfig(drum_min, drum_max, drum_enabled)
    handpan_config = InstrumentConfig(handpan_min, handpan_max, handpan_enabled)
    crystal_bowl_config = InstrumentConfig(crystal_bowl_min, crystal_bowl_max, crystal_bowl_enabled)
    ambient_ratio = parse_ambient_value(args.ambient)
    print("Starting mindfulness BGM (Modified Version)...")
    print("\nSettings:")
    print("- Four meditation instruments")
    print("- Enhanced sound depth without distortion")
    print("- Simple reverb effect")
    print("- Balanced harmonics")
    print("- Smooth transitions")
    print("\nInstrument settings:")
    instrument_configs = [
        ("Tibetan bell", bell_config),
        ("Slit drum", drum_config),
        ("Handpan", handpan_config),
        ("Crystal singing bowl", crystal_bowl_config)
    ]
    for name, config in instrument_configs:
        if config.enabled:
            print(f"- {name}: {config.min_interval:.1f}-{config.max_interval:.1f} seconds (SELECTED)")
        else:
            print(f"- {name}: DISABLED")
    if ambient_ratio > 0:
        print(f"- Ambient sounds: {ambient_ratio:.1%} mix ratio")
    else:
        print("- Ambient sounds: DISABLED")
    print("\nPress Ctrl+C to stop.")
    sd.default.samplerate = SAMPLE_RATE
    sd.default.blocksize = BUFFER_FRAMES
    sd.default.channels = 2
    generator = MindfulnessBGM(bell_config, drum_config, handpan_config, crystal_bowl_config, ambient_ratio)
    with sd.OutputStream(callback=generator.callback):
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nGracefully stopping...")

if __name__ == "__main__":
    main()
