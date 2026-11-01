"""Regression checks run without opening an audio device."""
import random
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

from src.generator import MindfulnessBGM
from src.generator_rpiz2w import MindfulnessBGMRPiZ2W
from src.instruments_base import InstrumentConfig, VoicePool
from src.sound_types import SoundType
from src.ambient_sounds import AmbientSoundGenerator
from src.effects import AudioEffects
from src.effects_rpiz2w import AudioEffectsRPiZ2W
from src.playback import Playback
from src.utils import parse_instrument_interval, parse_ambient_value
from src.instruments import TibetanBell, SlitDrum, Handpan, CrystalSingingBowl
from src.instruments_rpiz2w import (
    TibetanBellRPiZ2W, SlitDrumRPiZ2W, HandpanRPiZ2W, CrystalSingingBowlRPiZ2W)
from src.synthesizer import Synthesizer


def generator(cls=MindfulnessBGM, **options):
    off = InstrumentConfig(0, 0)
    return cls(off, off, off, off, options.pop('ambient_ratio', 0),
               auto_start=False, **options)


class AudioRegressionTests(unittest.TestCase):
    def test_all_progression_settings_can_start_and_change_chord(self):
        for style in ('circle-of-fifths', 'pentatonic', 'modal', 'random'):
            for mode in ('stable', 'balanced', 'dynamic'):
                with self.subTest(style=style, mode=mode):
                    bgm = generator(progression_style=style, modulation_mode=mode)
                    for _ in range(4):
                        bgm._change_chord()
                        out = np.empty((2048, 2), np.float32)
                        bgm.callback(out, len(out), None, None)
                        self.assertTrue(np.isfinite(out).all())
                        bgm._update_transitions(bgm.sample_rate * 4)

    def test_repeated_start_stop_has_no_duplicate_or_remaining_workers(self):
        for cls in (MindfulnessBGM, MindfulnessBGMRPiZ2W):
            with self.subTest(cls=cls.__name__):
                config = InstrumentConfig(60, 120)
                options = {'session_evolution': True} if cls is not MindfulnessBGMRPiZ2W else {}
                bgm = cls(config, config, config, config, 0, auto_start=False, **options)
                for _ in range(3):
                    bgm.start()
                    threads = list(bgm.schedulers)
                    self.assertEqual(len(threads), 6 if options else 5)
                    bgm.start()
                    self.assertEqual(bgm.schedulers, threads)
                    bgm.stop()
                    self.assertTrue(all(not t.is_alive() for t in threads))
                    self.assertEqual(bgm.schedulers, [])
                    bgm.stop()

    def test_zero_interval_disables_instrument_for_all_callers(self):
        self.assertFalse(InstrumentConfig(0, 0, True).enabled)
        self.assertFalse(InstrumentConfig(0, 30, True).enabled)

    def test_invalid_intervals_are_rejected(self):
        for value in ('-1', '20-10', 'nan', 'inf', '0-inf', '1-2-3'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_instrument_interval(value)
        for low, high in ((-1, 1), (2, 1), (0, float('inf'))):
            with self.assertRaises(ValueError):
                InstrumentConfig(low, high)
        self.assertEqual(parse_instrument_interval('30'), (24, 36))
        self.assertEqual(parse_instrument_interval(' 0 '), (0, 0))
        self.assertEqual(parse_instrument_interval('10-20'), (10, 20))
        for value in ('nan', 'inf'):
            with self.assertRaises(ValueError):
                parse_ambient_value(value)

    def test_late_bell_sound_starts_with_a_fresh_envelope(self):
        for cls in (MindfulnessBGM, MindfulnessBGMRPiZ2W):
            with self.subTest(cls=cls.__name__):
                bgm = generator(cls)
                bgm.phase = 60 * bgm.sample_rate
                bgm.chord = [220.0]
                with patch.object(SoundType, 'get_random_except', return_value=SoundType.BELL_LIKE):
                    bgm._change_sound_type()
                bgm.fade_progress = 1.0
                t = (np.arange(2048) + bgm.phase) / bgm.sample_rate
                wave = bgm._generate_chord_wave(t)
                self.assertGreater(np.sqrt(np.mean(wave**2)), 0.1)
                bgm._update_transitions(2048)
                self.assertEqual(bgm.sound_start_frame, 60 * bgm.sample_rate)

    def test_audio_is_finite_in_all_timbres_and_output_modes(self):
        for cls in (MindfulnessBGM, MindfulnessBGMRPiZ2W):
            for binaural in (False, True) if cls is MindfulnessBGM else (False,):
                options = {'binaural_enhancement': binaural} if cls is MindfulnessBGM else {}
                bgm = generator(cls, ambient_ratio=0.3, **options)
                for sound in SoundType:
                    bgm.current_sound_type = sound
                    bgm.sound_start_frame = bgm.phase
                    out = np.empty((2048, 2), np.float32)
                    for _ in range(3):
                        bgm.callback(out, len(out), None, None)
                        self.assertTrue(np.isfinite(out).all())
                        self.assertLess(np.max(np.abs(out)), 1)

    def test_binaural_oscillators_have_distinct_frequencies(self):
        bgm = generator(binaural_enhancement=True)
        bgm.chord = [440.0]
        out = np.empty((bgm.sample_rate * 5, 2), np.float32)
        bgm.callback(out, len(out), None, None)
        bins = np.fft.rfftfreq(len(out), 1/bgm.sample_rate)
        spectrum = np.abs(np.fft.rfft(out * np.hanning(len(out))[:, None], axis=0))
        peaks = bins[np.argmax(spectrum, axis=0)]
        np.testing.assert_allclose(peaks, [440*0.9992, 440*1.0008], atol=0.11)

    def test_ocean_has_audible_energy_and_small_dc_component(self):
        np.random.seed(12)
        ambient = AmbientSoundGenerator(16000)
        audio = np.concatenate([ambient.generate_nature_sounds(1024, 0.3) for _ in range(80)])
        power = np.abs(np.fft.rfft(audio)) ** 2
        frequencies = np.fft.rfftfreq(len(audio), 1/16000)
        self.assertLess(power[frequencies < 20].sum() / power.sum(), 0.05)
        rms = np.sqrt(np.mean(audio**2))
        self.assertGreater(rms, 0.005)
        self.assertLess(rms, 0.08)

    def test_ocean_filter_is_independent_of_block_size(self):
        np.random.seed(10)
        whole = AmbientSoundGenerator(16000)
        expected = whole.generate_nature_sounds(4096, 0.3)
        np.random.seed(10)
        split = AmbientSoundGenerator(16000)
        actual = np.concatenate([split.generate_nature_sounds(1024, 0.3) for _ in range(4)])
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_stereo_delay_retains_history_across_blocks(self):
        signal = np.zeros(32)
        signal[15] = 1
        expected = AudioEffects(48000).apply_stereo_enhancement(signal)
        effect = AudioEffects(48000)
        actual = np.vstack([effect.apply_stereo_enhancement(signal[:16]),
                            effect.apply_stereo_enhancement(signal[16:])])
        np.testing.assert_allclose(actual, expected)

    def test_cli_flags_can_disable_optional_features(self):
        from main import parse_arguments
        with patch('sys.argv', ['main.py']):
            args = parse_arguments()
            self.assertTrue(args.session_evolution)
            self.assertTrue(args.binaural_enhancement)
        with patch('sys.argv', ['main.py', '--no-session-evolution', '--no-binaural-enhancement']):
            args = parse_arguments()
            self.assertFalse(args.session_evolution)
            self.assertFalse(args.binaural_enhancement)


def spectrum_peaks(audio, sample_rate, threshold=0.01):
    """Frequencies of local spectral maxima above threshold * the largest peak."""
    magnitude = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    bins = np.fft.rfftfreq(len(audio), 1 / sample_rate)
    local_max = (magnitude[1:-1] > magnitude[:-2]) & (magnitude[1:-1] >= magnitude[2:])
    strong = magnitude[1:-1] > threshold * magnitude.max()
    return bins[1:-1][local_max & strong]


class SoundQualityRegressionTests(unittest.TestCase):
    INSTRUMENTS = (
        (TibetanBell, 48000, 2000.0), (SlitDrum, 48000, 220.0),
        (Handpan, 48000, 330.0), (CrystalSingingBowl, 48000, 440.0),
        (TibetanBellRPiZ2W, 16000, 2000.0), (SlitDrumRPiZ2W, 16000, 220.0),
        (HandpanRPiZ2W, 16000, 330.0), (CrystalSingingBowlRPiZ2W, 16000, 440.0),
    )

    def test_instrument_envelopes_are_independent_of_block_size(self):
        for cls, rate, freq in self.INSTRUMENTS:
            with self.subTest(cls=cls.__name__):
                np.random.seed(3)
                split = cls(rate)
                split.trigger(freq)
                blocks = np.concatenate([split.generate(1024) for _ in range(8)])
                np.random.seed(3)
                whole = cls(rate)
                whole.trigger(freq)
                np.testing.assert_allclose(blocks, whole.generate(8192), atol=1e-9)

    def test_bell_partials_never_alias_below_nyquist(self):
        ratios = (1, 1.02, 2.76, 5.43, 3.98, 6.51)
        for cls, rate, freq in ((TibetanBell, 48000, 4186.0), (TibetanBellRPiZ2W, 16000, 2000.0)):
            with self.subTest(cls=cls.__name__):
                bell = cls(rate)
                bell.trigger(freq)
                audio = bell.generate(rate)
                expected = [freq * r for r in ratios if freq * r < rate / 2]
                for peak in spectrum_peaks(audio, rate):
                    self.assertLess(min(abs(peak - e) for e in expected), 200, peak)

    def test_drone_timbres_never_alias_below_nyquist(self):
        # Every harmonic, subharmonic and inharmonic ratio generate_tone may use
        ratios = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0.5, 0.25, 0.125, 5.7, 7.3] + [8.5 + i * 0.8 for i in range(5, 7)]
        freq, rate = 2400.0, 48000
        t = np.arange(rate) / rate
        for sound in SoundType:
            for richness in ('minimal', 'normal', 'rich'):
                with self.subTest(sound=sound, richness=richness):
                    audio = Synthesizer.generate_tone(freq, t, sound, richness, sample_rate=rate)
                    for peak in spectrum_peaks(audio, rate):
                        self.assertLess(min(abs(peak - freq * r) for r in ratios), 20, peak)

    def test_soft_pad_uses_consecutive_harmonics(self):
        rate, freq = 48000, 440.0
        t = np.arange(rate) / rate
        audio = Synthesizer.generate_tone(freq, t, SoundType.SOFT_PAD, 'normal',
                                          envelope_t=t + 10, sample_rate=rate)
        magnitude = np.abs(np.fft.rfft(audio))  # 1 Hz bins
        levels = magnitude[[440, 880, 1320, 1760]] / magnitude[440]
        np.testing.assert_allclose(levels, [1, 0.15, 0.08, 0.04], rtol=1e-3)

    def test_bell_like_drone_does_not_fall_silent(self):
        for cls in (MindfulnessBGM, MindfulnessBGMRPiZ2W):
            with self.subTest(cls=cls.__name__):
                bgm = generator(cls)
                bgm.chord = [220.0]
                bgm.current_sound_type = SoundType.BELL_LIKE
                bgm.sound_start_frame = 0
                rms = []
                for seconds in (0, 30):
                    bgm.phase = seconds * bgm.sample_rate
                    t = (np.arange(2048) + bgm.phase) / bgm.sample_rate
                    rms.append(np.sqrt(np.mean(bgm._generate_chord_wave(t) ** 2)))
                self.assertGreater(rms[1], 0.25 * rms[0])

    def test_bowl_vibrato_depth_does_not_grow_over_time(self):
        rate, freq = 48000, 1000.0
        t = np.arange(20 * rate) / rate
        phase = CrystalSingingBowl._bowl_fm_phase(freq, t, rubbing_phase=1.0)
        instantaneous = np.diff(phase) * rate / (2 * np.pi)
        self.assertLess(np.max(np.abs(instantaneous - freq)), 0.2)

    def test_modulation_lands_on_the_root_the_bridge_prepared(self):
        selectors = {'circle-of-fifths': 'select_next_root', 'pentatonic': 'select_next_pentatonic_root',
                     'modal': 'select_next_modal_root'}
        for style, selector in selectors.items():
            with self.subTest(style=style):
                bgm = generator(progression_style=style, modulation_mode='balanced')
                bgm.current_root = 220.0
                with patch.object(Synthesizer, selector, side_effect=[329.63, 261.63]):
                    bgm._change_chord()  # bridge, announcing 329.63
                    self.assertTrue(bgm.is_preparing_modulation)
                    bgm._update_transitions(bgm.sample_rate * 4)  # Bridge crossfade completes
                    bgm._change_chord()  # actual modulation
                self.assertAlmostEqual(bgm.current_root, 329.63)
                self.assertIsNone(bgm.modulation_target)


class SmoothnessRegressionTests(unittest.TestCase):
    def test_restrike_lets_the_previous_sound_ring_on(self):
        rate = 48000
        pool = VoicePool(TibetanBell, rate)
        first, second = TibetanBell(rate), TibetanBell(rate)
        pool.trigger(2000.0)
        first.trigger(2000.0)
        for _ in range(10):
            np.testing.assert_allclose(pool.generate(2048), first.generate(2048))
        pool.trigger(3000.0)
        second.trigger(3000.0)
        np.testing.assert_allclose(pool.generate(2048), first.generate(2048) + second.generate(2048))
        self.assertEqual(len(pool.voices), 2)

    def test_voice_stealing_fades_the_oldest_voice(self):
        rate = 48000
        pool = VoicePool(TibetanBell, rate, max_voices=1)
        old, new = TibetanBell(rate), TibetanBell(rate)
        pool.trigger(2000.0)
        old.trigger(2000.0)
        pool.generate(12000)
        old.generate(12000)
        pool.trigger(3000.0)
        new.trigger(3000.0)
        release = int(VoicePool.RELEASE_SECONDS * rate)
        fade = np.clip(1 - np.arange(1, release + 1) / release, 0, 1)
        expected = old.generate(release) * fade + new.generate(release)
        np.testing.assert_allclose(pool.generate(release), expected)
        self.assertEqual(len(pool.voices), 1)  # The old voice is gone once faded

    def test_out_of_range_strike_is_ignored(self):
        pool = VoicePool(TibetanBell, 48000)
        pool.trigger(50.0)
        self.assertFalse(pool.is_playing)
        self.assertTrue(np.all(pool.generate(256) == 0))

    def test_percussion_gain_is_continuous_and_keeps_decays_natural(self):
        bgm = generator()
        frames = 1024
        # silence, attack, three decaying blocks, silence
        blocks = [(0.1, 0.0), (0.1, 0.2), (0.1, 0.1), (0.1, 0.05), (0.1, 0.02), (0.1, 0.0), (0.1, 0.0)]
        gains = [bgm._percussion_gain(main, perc, frames) for main, perc in blocks]
        np.testing.assert_allclose(gains[1], 0.25)  # Attack out of silence takes its gain at once
        ringing = np.concatenate(gains[1:5])
        self.assertTrue(np.all(np.diff(ringing) <= 1e-12))  # Never pulled back up while ringing
        whole = np.concatenate(gains[1:])
        self.assertLess(np.max(np.abs(np.diff(whole))), 1.0 / frames)  # No steps between blocks
        self.assertEqual(gains[-1][-1], 1.0)  # Recovers during silence

    def test_limiter_gain_ramps_instead_of_stepping(self):
        rate, frames = 48000, 2048
        effects = AudioEffects(rate)
        loud, quiet = np.full(frames, 1.0), np.full(frames, 0.1)
        out_loud = effects.apply_soft_limiting(loud)
        self.assertAlmostEqual(effects.limiter_gain, 0.5)
        out_quiet = effects.apply_soft_limiting(quiet)
        self.assertAlmostEqual(effects.limiter_gain, 0.5 + frames / (0.5 * rate))
        self.assertTrue(np.all(np.abs(np.concatenate([out_loud, out_quiet])) <= 0.85))
        # Undo the tanh stage to recover the applied gain: no step at the boundary
        applied = np.concatenate([np.arctanh(out_loud / 0.85) / 0.9 * 0.85 / loud,
                                  np.arctanh(out_quiet / 0.85) / 0.9 * 0.85 / quiet])
        self.assertLess(np.max(np.abs(np.diff(applied))), 1e-3)

    def test_stereo_signals_are_limited_per_sample(self):
        effects = AudioEffects(48000)
        out = effects.apply_soft_limiting(np.ones((512, 2)))
        self.assertEqual(out.shape, (512, 2))
        np.testing.assert_allclose(out[:, 0], out[:, 1])


class OutputLevelTests(unittest.TestCase):
    def test_default_output_level_is_audible_and_leaves_headroom(self):
        for cls, frames in ((MindfulnessBGM, 2048), (MindfulnessBGMRPiZ2W, 1024)):
            with self.subTest(cls=cls.__name__):
                random.seed(0)
                np.random.seed(0)
                bgm = generator(cls)
                out = np.empty((frames, 2), np.float32)
                squares, peak = [], 0.0
                for _ in range(int(20 * bgm.sample_rate / frames)):
                    bgm.callback(out, frames, None, None)
                    squares.append(np.mean(out.astype(float) ** 2))
                    peak = max(peak, np.abs(out).max())
                rms_db = 10 * np.log10(np.mean(squares))
                self.assertGreater(rms_db, -28)  # Audible without maxing system volume
                self.assertLess(rms_db, -14)
                self.assertLess(20 * np.log10(peak), -6)  # Peaks stay below the limiter


def a_weighted_db(audio, sample_rate):
    """A-weighted level (dBFS), approximating how loud the ear hears a signal."""
    spectrum = np.fft.rfft(audio)
    f2 = np.maximum(np.fft.rfftfreq(len(audio), 1 / sample_rate), 1e-3) ** 2
    weight = (12194**2 * f2**2 / ((f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2))
                                  * (f2 + 12194**2))) * 10 ** 0.1
    return 10 * np.log10(np.mean(np.fft.irfft(spectrum * weight, len(audio)) ** 2))


class QuietPresetLoudnessTests(unittest.TestCase):
    def render(self, seconds=10, **options):
        random.seed(0)
        np.random.seed(0)
        bgm = generator(**options)
        out = np.empty((2048, 2), np.float32)
        blocks = []
        for _ in range(int(seconds * bgm.sample_rate / 2048)):
            bgm.callback(out, 2048, None, None)
            blocks.append(out[:, 0].copy())
        return np.concatenate(blocks), bgm.sample_rate

    def test_quiet_preset_drone_is_carried_by_audible_harmonics(self):
        audio, rate = self.render(meditation_mode=True, harmonic_richness='minimal')
        power = np.abs(np.fft.rfft(audio)) ** 2
        frequencies = np.fft.rfftfreq(len(audio), 1 / rate)
        # Small speakers barely reproduce 110 Hz; harmonics must carry the drone
        self.assertGreater(power[frequencies > 200].sum() / power.sum(), 0.25)
        self.assertLess(20 * np.log10(np.abs(audio).max()), -6)

    def test_quiet_preset_is_only_a_little_quieter_than_custom(self):
        quiet = a_weighted_db(*self.render(meditation_mode=True, harmonic_richness='minimal'))
        custom = a_weighted_db(*self.render())
        self.assertGreater(quiet, custom - 6)
        self.assertLess(quiet, custom)


class PerSampleTransitionTests(unittest.TestCase):
    def test_crossfade_ramps_within_blocks_and_ends_on_the_new_sound(self):
        for cls in (MindfulnessBGM, MindfulnessBGMRPiZ2W):
            with self.subTest(cls=cls.__name__):
                bgm = generator(cls)
                bgm._change_sound_type()
                ramps = []
                while bgm.is_transitioning:
                    bgm._update_transitions(2048)
                    if bgm.fade_ramp is not None:
                        ramps.append(bgm.fade_ramp)
                alpha = np.concatenate(ramps)
                step = 2048 / (bgm.fade_time * bgm.sample_rate) / 2048
                self.assertAlmostEqual(alpha[0], step)
                self.assertEqual(alpha[-1], 1.0)  # The last faded block ends fully on the new sound
                self.assertLess(np.max(np.abs(np.diff(alpha))), 2 * step)  # No steps between blocks
                self.assertIsNone(bgm.next_sound_type)  # Committed on the following block

    def test_rendered_crossfade_uses_the_per_sample_ramp(self):
        bgm = generator()
        bgm.chord, bgm.next_chord = [220.0], [330.0]
        bgm.current_sound_type, bgm.next_sound_type = SoundType.PURE, None
        bgm.is_transitioning, bgm.fade_progress = True, 0.2
        bgm._update_transitions(2048)
        t = (np.arange(2048) + bgm.phase) / bgm.sample_rate
        tone = lambda f: Synthesizer.generate_tone(f, t, SoundType.PURE, 'normal', sample_rate=bgm.sample_rate)
        expected = (1 - bgm.fade_ramp) * tone(220.0) + bgm.fade_ramp * tone(330.0)
        np.testing.assert_allclose(bgm._generate_chord_wave(t), expected, atol=1e-12)

    def test_change_during_a_crossfade_waits_instead_of_jumping_back(self):
        for cls in (MindfulnessBGM, MindfulnessBGMRPiZ2W):
            with self.subTest(cls=cls.__name__):
                bgm = generator(cls)
                bgm._change_sound_type()
                target = bgm.next_sound_type
                bgm._update_transitions(int(bgm.fade_time * bgm.sample_rate / 2))
                progress = bgm.fade_progress
                bgm._change_sound_type()
                bgm._change_chord()
                self.assertEqual(bgm.fade_progress, progress)
                self.assertEqual(bgm.next_sound_type, target)

    def test_volume_and_breath_move_sample_by_sample(self):
        for cls in (MindfulnessBGM, MindfulnessBGMRPiZ2W):
            with self.subTest(cls=cls.__name__):
                bgm = generator(cls)
                bgm.target_volume = 0.4
                bgm.volume_transition_speed = 1.0 / (3 * bgm.sample_rate)
                envelope = []
                for _ in range(40):
                    bgm._update_transitions(1024)
                    envelope.append(bgm._level_envelope(1024, 0.2))
                envelope = np.concatenate(envelope)
                self.assertLess(envelope[-1], envelope[0])
                self.assertLess(np.max(np.abs(np.diff(envelope))), 1e-4)


class PiStereoTests(unittest.TestCase):
    def test_stereo_delay_is_causal_and_continuous_across_blocks(self):
        signal = np.random.RandomState(1).randn(64)
        expected = AudioEffectsRPiZ2W(16000).create_stereo(signal)
        effect = AudioEffectsRPiZ2W(16000)
        actual = np.vstack([effect.create_stereo(signal[i:i + 16]) for i in range(0, 64, 16)])
        np.testing.assert_allclose(actual, expected)
        delayed = np.concatenate([np.zeros(2), signal[:-2]])
        np.testing.assert_allclose(expected[:, 0], signal + 0.05 * delayed)
        np.testing.assert_allclose(expected[:, 1], delayed + 0.05 * signal)


class CliInterruptionTests(unittest.TestCase):
    def stream(self, active):
        stream = MagicMock()
        stream.active = active
        stream.__enter__.return_value = stream
        return stream

    def test_cli_exits_when_the_device_stops(self):
        import main
        stream = self.stream(active=False)
        with (patch('sys.argv', ['main.py', '--instrument', 'bell']),
              patch('sounddevice.OutputStream', return_value=stream),
              patch('time.sleep'), patch('builtins.print')):
            self.assertEqual(main.main(), 1)
        stream.close.assert_called_once()

    def test_cli_ctrl_c_fades_out_and_releases_the_device(self):
        import main
        stream = self.stream(active=True)
        with (patch('sys.argv', ['main.py']),
              patch('sounddevice.OutputStream', return_value=stream),
              patch('main.wait_until_stopped', side_effect=KeyboardInterrupt),
              patch('builtins.print'),
              patch('src.playback.Playback.stop_gracefully', autospec=True,
                    side_effect=lambda player, *args: player.stop()) as graceful):
            self.assertEqual(main.main(), 0)
        graceful.assert_called_once()
        stream.close.assert_called_once()

    def test_pi_cli_exits_when_the_device_stops(self):
        import main_rpiz2w
        stream = self.stream(active=False)
        with (patch('sys.argv', ['main_rpiz2w.py']),
              patch('sounddevice.OutputStream', return_value=stream),
              patch('time.sleep'), patch('builtins.print')):
            self.assertEqual(main_rpiz2w.main(), 1)

    def test_stop_gracefully_gives_up_when_the_device_does_not_call_back(self):
        player = Playback()
        with patch('sounddevice.OutputStream', return_value=self.stream(active=True)):
            player.start('none', {k: (0, 0) for k in ('bell', 'drum', 'handpan', 'crystal_bowl')}, 0)
        started = time.monotonic()
        player.stop_gracefully()
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertIsNone(player.stream)


if __name__ == '__main__':
    unittest.main()
