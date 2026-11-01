"""Session envelopes and the quiet sound profile, without waiting in real time."""
import unittest
from unittest.mock import patch

import numpy as np

from src.session import SessionEnvelope
from src.generator import MindfulnessBGM
from src.instruments_base import InstrumentConfig


class SessionTests(unittest.TestCase):
    def test_timed_session_fades_in_and_out_then_silences(self):
        envelope = SessionEnvelope(100, duration_seconds=10, fade_in=2, fade_out=3)
        gains = envelope.gains(1100)
        self.assertEqual(gains[0], 0)
        self.assertEqual(gains[500], 1)
        self.assertLess(gains[950], 0.1)
        self.assertTrue(np.all(gains[1000:] == 0))
        self.assertTrue(envelope.finished)
        self.assertEqual(envelope.remaining_seconds, 0)

    def test_envelope_is_continuous_across_audio_blocks(self):
        whole = SessionEnvelope(100, 10, 2, 3).gains(1200)
        envelope = SessionEnvelope(100, 10, 2, 3)
        split = np.concatenate([envelope.gains(100) for _ in range(12)])
        np.testing.assert_array_equal(whole, split)

    def test_untimed_session_stays_open(self):
        envelope = SessionEnvelope(100, 0, 2)
        gains = envelope.gains(6000)
        self.assertFalse(envelope.finished)
        self.assertIsNone(envelope.remaining_seconds)
        self.assertEqual(gains[-1], 1)

    def test_quiet_mode_keeps_chord_and_timbre_stable(self):
        off = InstrumentConfig(0, 0)
        bgm = MindfulnessBGM(off, off, off, off, 0, auto_start=False,
                             harmonic_richness='minimal', meditation_mode=True)
        chord = bgm.chord.copy()
        sound = bgm.current_sound_type
        self.assertEqual(chord, [110.0, 165.0])
        with patch.object(bgm.stop_flag, 'wait', side_effect=[False, True]) as wait:
            bgm._event_scheduler()
        self.assertTrue(45 <= wait.call_args_list[0].args[0] <= 90)
        self.assertEqual(bgm.chord, chord)
        self.assertEqual(bgm.current_sound_type, sound)
        self.assertTrue(0.45 <= bgm.target_volume <= 0.55)
