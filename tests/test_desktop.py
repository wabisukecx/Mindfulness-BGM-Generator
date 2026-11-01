"""Qt interaction tests with audio devices mocked out."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import sounddevice as sd
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest

from app import MainWindow
from src.playback import Playback


class DesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def setUp(self):
        self.preferences_patch = patch('app.QSettings')
        self.preferences = self.preferences_patch.start().return_value
        self.preferences.value.return_value = 'en'
        self.addCleanup(self.preferences_patch.stop)
        self.devices = patch('sounddevice.query_devices', return_value=[
            {'name': 'Test speakers', 'max_output_channels': 2},
            {'name': 'Input only', 'max_output_channels': 0}])
        self.devices.start()
        self.stream = MagicMock()
        self.stream.active = True
        self.stream_patch = patch('sounddevice.OutputStream', return_value=self.stream)
        self.stream_patch.start()
        self.message = patch('app.QMessageBox.warning')
        self.warning = self.message.start()
        self.window = MainWindow()
        self.window.show()
        self.qt.processEvents()

    def tearDown(self):
        self.stream.stop.side_effect = None
        self.window.close()
        self.qt.processEvents()
        self.devices.stop()
        self.stream_patch.stop()
        self.message.stop()

    def click(self, button):
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        self.qt.processEvents()

    def test_repeated_start_stop_and_settings(self):
        self.window.instrument.setCurrentIndex(1)
        self.window.intervals['bell'][0].setValue(0)
        self.window.mode.setCurrentIndex(0)
        self.window.progression.setCurrentIndex(2)
        self.window.binaural.setChecked(False)
        self.window.device.setCurrentIndex(1)
        self.assertEqual(self.window.device.count(), 2)
        for _ in range(3):
            self.click(self.window.start_button)
            bgm = self.window.player.generator
            self.assertFalse(bgm.bell_config.enabled)
            self.assertEqual(bgm.modulation_mode, 'stable')
            self.assertEqual(bgm.progression_style, 'modal')
            self.assertFalse(bgm.binaural_enhancement)
            self.assertFalse(self.window.settings.isEnabled())
            threads = list(bgm.schedulers)
            self.press_stop()
            self.assertIsNone(self.window.player.stream)
            self.assertTrue(self.window.settings.isEnabled())
            self.assertTrue(all(not t.is_alive() for t in threads))
        self.warning.assert_not_called()

    def test_close_stops_workers_and_audio(self):
        self.click(self.window.start_button)
        threads = list(self.window.player.generator.schedulers)
        self.window.close()
        self.assertTrue(all(not t.is_alive() for t in threads))
        self.assertIsNone(self.window.player.generator)
        self.stream.close.assert_called_once()

    def test_device_start_failure_restores_stopped_state(self):
        self.stream.start.side_effect = sd.PortAudioError('No output')
        self.click(self.window.start_button)
        self.assertIsNone(self.window.player.generator)
        self.assertIsNone(self.window.player.stream)
        self.assertTrue(self.window.start_button.isEnabled())
        self.assertFalse(self.window.stop_button.isEnabled())
        self.stream.close.assert_called_once()
        self.warning.assert_called_once()

    def test_device_constructor_failure_leaves_no_workers(self):
        with patch('sounddevice.OutputStream', side_effect=sd.PortAudioError('No device')):
            self.click(self.window.start_button)
        self.assertIsNone(self.window.player.generator)
        self.warning.assert_called_once()

    def test_interruption_returns_to_stopped_state(self):
        self.click(self.window.start_button)
        threads = list(self.window.player.generator.schedulers)
        self.stream.active = False
        self.window.check_playback()
        self.assertIsNone(self.window.player.stream)
        self.assertTrue(all(not t.is_alive() for t in threads))
        self.assertTrue(self.window.settings.isEnabled())
        self.warning.assert_called_once()

    def test_interval_bounds_stay_ordered(self):
        low, high = self.window.intervals['bell']
        low.setValue(90)
        self.assertGreaterEqual(high.value(), low.value())
        high.setValue(10)
        self.assertLessEqual(low.value(), high.value())

    def test_stop_failure_still_closes_device_and_workers(self):
        self.click(self.window.start_button)
        threads = list(self.window.player.generator.schedulers)
        self.stream.stop.side_effect = sd.PortAudioError('Device disconnected')
        self.press_stop()
        self.assertIsNone(self.window.player.stream)
        self.assertTrue(all(not t.is_alive() for t in threads))
        self.stream.close.assert_called_once()
        self.warning.assert_called_once()

    def test_callback_failure_silences_output_and_aborts(self):
        self.click(self.window.start_button)
        out = np.ones((32, 2), np.float32)
        with patch.object(self.window.player.generator, 'callback', side_effect=RuntimeError('Synthesis error')):
            with self.assertRaises(sd.CallbackAbort):
                self.window.player._callback(out, 32, None, None)
        self.assertTrue(np.all(out == 0))
        self.window.check_playback()
        self.assertIsNone(self.window.player.stream)
        self.warning.assert_called_once()

    def test_incense_defaults_are_quiet_and_simple(self):
        self.assertEqual(self.window.profile.currentData(), 'incense')
        self.assertEqual(self.window.ambient.value(), 0)
        self.assertEqual(self.window.instrument.currentData(), 'none')
        self.assertFalse(self.window.binaural.isChecked())
        self.click(self.window.start_button)
        self.assertTrue(self.window.player.generator.meditation_mode)
        self.assertEqual(self.window.player.session.duration, 20 * 60 * 48000)
        self.assertFalse(self.window.settings.isVisible())
        self.assertTrue(self.window.sitting.isVisible())

    def test_profiles_and_custom_changes(self):
        self.window.profile.setCurrentIndex(self.window.profile.findData('bowl'))
        self.assertEqual(self.window.instrument.currentData(), 'crystal_bowl')
        self.assertEqual(self.window.intervals['crystal_bowl'][0].value(), 90)
        self.assertTrue(self.window.meditation_mode)
        self.window.mode.setCurrentIndex(2)
        self.assertEqual(self.window.profile.currentData(), 'custom')
        self.assertFalse(self.window.meditation_mode)

    def press_stop(self):
        """Press 終える and let its fade-out run, as the monitor timer would."""
        self.click(self.window.stop_button)
        if self.window.player.stream is not None and self.window.player.generator is not None:
            self.pump(2)
        self.window.check_playback()

    def test_stop_fades_out_before_closing_the_device(self):
        self.window.profile.setCurrentIndex(self.window.profile.findData('custom'))  # No rin
        self.click(self.window.start_button)
        self.pump(10)  # Past the session fade-in
        self.click(self.window.stop_button)
        self.assertIsNotNone(self.window.player.stream)  # Still sounding while it fades
        self.assertFalse(self.window.stop_button.isEnabled())
        self.assertEqual(self.window.status.text(), 'Ending gently…')
        levels = []
        out = np.ones((2048, 2), np.float32)
        with patch.object(self.window.player.generator, 'callback'):
            for _ in range(40):
                out.fill(1)
                self.window.player._callback(out, len(out), None, None)
                levels.append(out[:, 0].copy())
        gain = np.concatenate(levels) / self.window.player.volume
        self.assertGreater(gain[0], 0.99)
        self.assertTrue(np.all(np.diff(gain) <= 1e-9))       # Falls steadily
        self.assertLess(np.max(np.abs(np.diff(gain))), 1e-4)  # With no step
        self.assertEqual(gain[-1], 0)
        self.window.check_playback()
        self.assertIsNone(self.window.player.stream)
        self.assertEqual(self.window.status.text(), 'Stopped')
        self.warning.assert_not_called()

    def pump(self, seconds):
        """Run the audio callback for a stretch of time without a real device."""
        out = np.zeros((2048, 2), np.float32)
        with patch.object(self.window.player.generator, 'callback'):
            for _ in range(int(seconds * 48000 / 2048)):
                self.window.player._callback(out, len(out), None, None)
        return out

    def test_session_completion_stops_without_an_alert(self):
        self.click(self.window.start_button)
        threads = list(self.window.player.generator.schedulers)
        session = self.window.player.session
        session.position = session.duration - session.fade_out
        self.pump(1)  # The closing rin sounds as the fade-out begins
        session.position = session.duration
        self.window.check_playback()
        self.assertIsNotNone(self.window.player.stream)  # Its ring is allowed to die away
        self.pump(40)
        self.window.check_playback()
        self.assertIsNone(self.window.player.stream)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(self.window.status.text(), 'Session complete')
        self.warning.assert_not_called()

    def test_volume_and_session_envelope_reach_the_audio_output(self):
        self.window.profile.setCurrentIndex(self.window.profile.findData('custom'))  # No rin
        self.window.volume.setValue(50)
        self.click(self.window.start_button)
        out = np.ones((1024, 2), np.float32)
        with patch.object(self.window.player.generator, 'callback'):
            self.window.player._callback(out, len(out), None, None)
        self.assertEqual(out[0, 0], 0)
        self.assertLess(out[-1, 0], 0.5)
        self.window.player.session.position = self.window.player.session.duration
        out.fill(1)
        with patch.object(self.window.player.generator, 'callback'):
            self.window.player._callback(out, len(out), None, None)
        self.assertTrue(np.all(out == 0))


    def test_sho_preset_plays_the_sho_voice(self):
        from src.meditation import ShoVoice, SmokeDrone
        self.window.profile.setCurrentIndex(self.window.profile.findData('sho'))
        self.click(self.window.start_button)
        self.assertIsInstance(self.window.player.generator.drone, ShoVoice)
        self.assertIsNotNone(self.window.player.rin)
        self.press_stop()
        self.window.profile.setCurrentIndex(self.window.profile.findData('incense'))
        self.click(self.window.start_button)
        self.assertIsInstance(self.window.player.generator.drone, SmokeDrone)

    def set_language(self, code):
        self.window.language.setCurrentIndex(self.window.language.findData(code))
        self.qt.processEvents()

    def test_language_switch_preserves_custom_settings(self):
        self.assertEqual(self.window.start_button.text(), 'Begin')
        self.window.profile.setCurrentIndex(self.window.profile.findData('bowl'))
        self.window.volume.setValue(37)
        self.window.ambient.setValue(17)
        self.window.mode.setCurrentIndex(self.window.mode.findData('dynamic'))
        self.window.duration.setCurrentIndex(self.window.duration.findData(0))
        self.window.device.setCurrentIndex(1)
        self.window.intervals['bell'][0].setValue(42)
        self.window.advanced.setChecked(True)
        self.window.tabs.setCurrentIndex(1)
        self.set_language('ja')
        self.assertEqual(self.window.start_button.text(), 'はじめる')
        self.assertEqual(self.window.profile.currentText(), '自分で調整')
        self.assertEqual(self.window.mode.currentText(), '変化を楽しむ')
        self.assertEqual(self.window.duration.currentText(), '時間を決めない')
        self.assertEqual(self.window.tabs.tabText(1), '楽器の間隔')
        self.assertEqual(self.window.intervals['bell'][0].suffix(), ' 秒')
        self.assertEqual(self.window.status.text(), '停止中')
        self.assertEqual(self.window.volume.value(), 37)
        self.assertEqual(self.window.volume_label.text(), '37%')
        self.assertEqual(self.window.ambient.value(), 17)
        self.assertEqual(self.window.intervals['bell'][0].value(), 42)
        self.assertEqual(self.window.device.currentData(), 0)
        self.assertFalse(self.window.meditation_mode)
        self.assertEqual(self.window.tabs.currentIndex(), 1)
        self.set_language('en')
        self.assertEqual(self.window.start_button.text(), 'Begin')
        self.assertEqual(self.window.profile.currentData(), 'custom')
        self.assertEqual(self.window.volume.value(), 37)

    def test_language_switch_during_playback_and_fade(self):
        self.click(self.window.start_button)
        generator = self.window.player.generator
        session = self.window.player.session
        self.set_language('ja')
        self.assertEqual(self.window.status.text(), '静坐中')
        self.assertEqual(self.window.session_note.text(), '20分の静坐 · 終わりは静かに')
        self.assertEqual(self.window.profile.currentText(), '線香と静坐')
        self.assertFalse(self.window.settings.isVisible())
        self.assertIs(self.window.player.generator, generator)
        self.assertIs(self.window.player.session, session)
        self.click(self.window.stop_button)
        self.assertEqual(self.window.status.text(), '静かに終えています…')
        self.set_language('en')
        self.assertEqual(self.window.status.text(), 'Ending gently…')
        self.assertFalse(self.window.stop_button.isEnabled())
        self.pump(2)
        self.window.check_playback()
        self.assertEqual(self.window.status.text(), 'Stopped')

    def test_language_preference_is_saved_and_restored(self):
        self.set_language('ja')
        self.preferences.setValue.assert_called_with('language', 'ja')
        self.preferences.value.return_value = 'ja'
        window = MainWindow()
        try:
            self.assertEqual(window.language.currentData(), 'ja')
            self.assertEqual(window.start_button.text(), 'はじめる')
            self.assertEqual(window.device.itemText(0), 'システムの既定の出力先')
        finally:
            window.close()

    def test_unknown_language_preference_falls_back_to_english(self):
        self.preferences.value.return_value = 'unsupported'
        window = MainWindow()
        try:
            self.assertEqual(window.language.currentData(), 'en')
            self.assertEqual(window.start_button.text(), 'Begin')
        finally:
            window.close()

    def test_japanese_playback_error(self):
        self.set_language('ja')
        self.stream.start.side_effect = sd.PortAudioError('No output')
        self.click(self.window.start_button)
        self.assertEqual(self.window.status.text(), '再生を開始できませんでした')
        args = self.warning.call_args.args
        self.assertEqual(args[1], '再生できません')
        self.assertIn('出力先と音声機器の接続を確認してください。', args[2])
        self.assertIn('No output', args[2])

if __name__ == '__main__':
    unittest.main()
