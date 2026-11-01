"""The quiet presets: smoke-like drone, spaces of near-silence, and the rin."""
import random
import unittest
from unittest.mock import patch

import numpy as np

from src.meditation import SmokeDrone, ShoVoice, BreathingSpace, Rin, RIN_FREQ
from src.generator import MindfulnessBGM
from src.instruments_base import InstrumentConfig
from src.playback import Playback


RATE = 48000
INTERVALS = {key: (0, 0) for key in ('bell', 'drum', 'handpan', 'crystal_bowl')}


def harmonic_level_db(drone, freq, start):
    t = (np.arange(2 * RATE) + start * RATE) / RATE
    spectrum = np.abs(np.fft.rfft(drone.render(t) * np.hanning(len(t))))
    index = int(round(freq * len(t) / RATE))
    return 20 * np.log10(spectrum[index - 2:index + 3].max())


class SmokeDroneTests(unittest.TestCase):
    def setUp(self):
        random.seed(1)
        self.drone = SmokeDrone([110.0, 165.0], RATE)

    def test_blocks_join_seamlessly(self):
        t = np.arange(8192) / RATE + 100
        split = np.concatenate([self.drone.render(t[i:i + 1024]) for i in range(0, 8192, 1024)])
        np.testing.assert_allclose(split, self.drone.render(t))

    def test_timbre_drifts_while_the_fundamental_holds(self):
        starts = range(0, 300, 10)
        fundamental = [harmonic_level_db(self.drone, 110, s) for s in starts]
        upper = [harmonic_level_db(self.drone, 440, s) for s in starts]
        self.assertLess(max(fundamental) - min(fundamental), 3)
        self.assertGreater(max(upper) - min(upper), 6)

    def test_coinciding_harmonics_beat_slowly(self):
        # 110 Hz x 3 and the detuned fifth x 2 land a fraction of a hertz apart
        t = np.arange(100 * RATE) / RATE
        spectrum = np.abs(np.fft.rfft(self.drone.render(t)))
        frequencies = np.fft.rfftfreq(len(t), 1 / RATE)
        band = (frequencies > 329.9) & (frequencies < 330.4)
        peaks = frequencies[band][np.argsort(spectrum[band])[-2:]]
        # Two distinct components beating every 3-10 seconds
        self.assertTrue(0.1 < abs(peaks[0] - peaks[1]) < 0.34, peaks)


class BreathingSpaceTests(unittest.TestCase):
    def setUp(self):
        random.seed(2)
        self.space = BreathingSpace()

    def test_recedes_almost_to_silence_then_returns(self):
        t = np.arange(0, 1500, 0.1)
        gain = self.space.gain(t)
        np.testing.assert_allclose(gain[t < BreathingSpace.FIRST[0]], 1)
        self.assertAlmostEqual(gain.min(), BreathingSpace.FLOOR)
        start, end = self.space.recesses[0]
        self.assertEqual(gain[np.argmin(abs(t - (end + 1)))], 1)
        # Raised-cosine fades: never faster than a gentle slope
        max_slope = np.pi / 2 * (1 - BreathingSpace.FLOOR) / BreathingSpace.FADE
        self.assertLessEqual(np.max(np.abs(np.diff(gain))) / 0.1, max_slope + 1e-6)

    def test_result_does_not_depend_on_block_size(self):
        t = np.arange(0, 1500, 0.1)
        whole = self.space.gain(t)
        random.seed(2)
        split_space = BreathingSpace()
        split = np.concatenate([split_space.gain(t[i:i + 500]) for i in range(0, len(t), 500)])
        np.testing.assert_allclose(split, whole)


class RinTests(unittest.TestCase):
    def test_ring_is_long_but_ends(self):
        rin = Rin(RATE)
        rin.trigger(RIN_FREQ)
        blocks = []
        while rin.is_playing:
            blocks.append(rin.generate(2048))
        audio = np.concatenate(blocks)
        self.assertGreater(len(audio) / RATE, 15)
        self.assertLess(len(audio) / RATE, 30)
        self.assertLess(20 * np.log10(np.abs(audio).max()), -6)

    def start(self, meditation_mode, duration_seconds=60):
        player = Playback()
        stream = patch('sounddevice.OutputStream')
        stream.start()
        self.addCleanup(stream.stop)
        player.start('none', INTERVALS, 0, duration_seconds=duration_seconds,
                     meditation_mode=meditation_mode)
        self.addCleanup(player.stop)
        return player

    def callback(self, player, blocks=1):
        out = np.zeros((2048, 2), np.float32)
        with patch.object(player.generator, 'callback'):
            for _ in range(blocks):
                player._callback(out, len(out), None, None)
        return out

    def test_opening_rin_is_heard_through_the_fade_in(self):
        player = self.start(meditation_mode=True)
        self.assertEqual(player.session.gains(1)[0], 0)  # The drone itself is still silent
        self.assertGreater(np.abs(self.callback(player, 2)).max(), 0.05)

    def test_closing_rin_sounds_as_the_fade_out_begins_and_rings_out(self):
        player = self.start(meditation_mode=True)
        self.callback(player, int(30 * RATE / 2048))  # Let the opening rin die away
        session = player.session
        session.position = session.duration - session.fade_out - 2 * 2048
        self.assertLess(np.abs(self.callback(player)).max(), 0.01)
        self.assertGreater(np.abs(self.callback(player, 2)).max(), 0.05)
        session.position = session.duration
        self.assertFalse(player.finished)  # Still ringing
        self.callback(player, int(30 * RATE / 2048))
        self.assertTrue(player.finished)

    def test_custom_settings_have_no_rin(self):
        player = self.start(meditation_mode=False)
        self.assertIsNone(player.rin)
        self.assertTrue(np.all(self.callback(player) == 0))


class ShoVoiceTests(unittest.TestCase):
    def setUp(self):
        random.seed(3)
        self.sho = ShoVoice(RATE)

    def sounding(self, time):
        """Pipes at full level at a moment in time."""
        self.sho._schedule_until(time)
        return {p for p, spans in self.sho.spans.items()
                if any(start + ShoVoice.FADE <= time <= end for start, end in spans)}

    def test_blocks_join_seamlessly(self):
        t = np.arange(0, 120, 1 / RATE)[::7]  # Sparse samples across several cluster changes
        whole = self.sho.render(t)
        random.seed(3)
        split_voice = ShoVoice(RATE)
        split = np.concatenate([split_voice.render(t[i:i + 1000]) for i in range(0, len(t), 1000)])
        np.testing.assert_allclose(split, whole)

    def test_holds_clusters_and_moves_one_pipe_at_a_time(self):
        self.sho._schedule_until(900)
        changes = sorted([start for spans in self.sho.spans.values() for start, _ in spans if start > 0]
                         + [end for spans in self.sho.spans.values() for _, end in spans if end < np.inf])
        self.assertGreater(len(changes), 10)
        gaps = np.diff(changes)
        self.assertGreater(gaps.min(), 0.3)  # Fingers move one at a time, never together
        clusters = {frozenset(c) for c in ShoVoice.CLUSTERS}
        for time in range(0, 900, 5):
            if all(abs(time - c) > 10 for c in changes):
                self.assertIn(frozenset(self.sounding(time)), clusters)

    def test_sounds_in_the_shos_register_with_a_slight_shimmer(self):
        for pitch, freq in self.sho.freqs.items():
            exact = ShoVoice.A4 * 2 ** (pitch / 12)
            cents = 1200 * np.log2(freq / exact)
            self.assertLessEqual(abs(cents), ShoVoice.DETUNE_CENTS)
            self.assertTrue(400 < freq < 1400)
        self.assertGreater(np.std([1200 * np.log2(f / (ShoVoice.A4 * 2 ** (p / 12)))
                                   for p, f in self.sho.freqs.items()]), 0.1)

    def test_swells_with_each_breath(self):
        hold = ShoVoice.HOLD[0]
        t = np.arange(0, hold, 1 / RATE)  # Within the first held cluster
        audio = self.sho.render(t)
        window = RATE // 2
        rms = np.sqrt(np.mean(audio[:len(audio) // window * window].reshape(-1, window) ** 2, axis=1))
        swing = 20 * np.log10(rms.max() / rms.min())
        self.assertGreater(swing, 1.5)
        self.assertLess(swing, 4)

    def test_preset_is_as_loud_as_the_drone_preset_and_leaves_headroom(self):
        from tests.test_regressions import a_weighted_db
        levels = {}
        for voice in ('drone', 'sho'):
            random.seed(0)
            np.random.seed(0)
            off = InstrumentConfig(0, 0)
            bgm = MindfulnessBGM(off, off, off, off, 0, auto_start=False,
                                 meditation_mode=True, meditation_voice=voice)
            out = np.empty((2048, 2), np.float32)
            blocks = []
            for _ in range(int(20 * RATE / 2048)):
                bgm.callback(out, 2048, None, None)
                blocks.append(out[:, 0].copy())
            audio = np.concatenate(blocks)
            levels[voice] = a_weighted_db(audio, RATE)
            self.assertLess(20 * np.log10(np.abs(audio).max()), -6)
        self.assertLess(abs(levels['sho'] - levels['drone']), 3)

    def test_unknown_voice_is_rejected(self):
        off = InstrumentConfig(0, 0)
        with self.assertRaises(ValueError):
            MindfulnessBGM(off, off, off, off, 0, auto_start=False,
                           meditation_mode=True, meditation_voice='organ')


if __name__ == '__main__':
    unittest.main()
