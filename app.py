#!/usr/bin/env python3
"""PyQt6 desktop interface for the Mindfulness BGM Generator."""
import sys

import sounddevice as sd
from PyQt6.QtCore import Qt, QTimer, QSettings
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFormLayout, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
    QSlider, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from src.config import DEFAULT_AMBIENT_RATIO
from src.playback import Playback
from src.i18n import translate


INSTRUMENTS = {
    'random': 'Random', 'bell': 'Tibetan bell', 'drum': 'Slit drum',
    'handpan': 'Handpan', 'crystal_bowl': 'Crystal bowl', 'none': 'No percussion',
}
DEFAULT_INTERVALS = {'bell': (15, 45), 'drum': (8, 25),
                     'handpan': (12, 30), 'crystal_bowl': (25, 60)}


PROFILES = {
    'incense': dict(name='Incense & Stillness', instrument='none', ambient=0, volume=55,
                    note='A quiet, low sustained tone. Return gently to scent and breath.'),
    'bowl': dict(name='Sitting with Resonance', instrument='crystal_bowl', ambient=0, volume=50,
                 note='Sparse bowl resonance and long stretches of stillness.'),
    'shore': dict(name='Distant Shore', instrument='none', ambient=10, volume=55,
                  note='Sit peacefully with a faint trace of distant waves.'),
    'sho': dict(name='Shō Resonance', instrument='none', ambient=0, volume=55, voice='sho',
                note='Shō-inspired chords swell with the breath, shifting one note at a time.'),
}


def combo(items):
    widget = QComboBox()
    for label, value in items:
        widget.addItem(label, value)
    return widget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.preferences = QSettings('MindfulnessBGM', 'MindfulnessBGMGenerator')
        language = self.preferences.value('language', 'en')
        self.language_code = language if language in ('en', 'ja') else 'en'
        self._text_bindings = []
        self._combo_bindings = []
        self._status_message = 'Stopped'
        self.player = Playback()
        self.setWindowTitle(self.tr_text('Incense & Stillness — Mindfulness BGM'))
        self._applying_profile = False
        self.meditation_mode = True
        self.meditation_voice = 'drone'
        self.resize(760, 730)
        self.setMinimumSize(660, 640)
        self._build_ui()
        self.apply_profile()
        self.refresh_devices()
        self.monitor = QTimer(self)
        self.monitor.setInterval(500)
        self.monitor.timeout.connect(self.check_playback)
        self.monitor.start()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(30, 26, 30, 24)
        layout.setSpacing(18)
        title = self._text_widget(QLabel, 'Incense & Stillness')
        title.setObjectName('title')
        header = QHBoxLayout()
        header.addWidget(title, 1)
        header.addWidget(self._text_widget(QLabel, 'Language'))
        self.language = combo([('English', 'en'), ('日本語', 'ja')])
        self.language.setCurrentIndex(self.language.findData(self.language_code))
        self.language.currentIndexChanged.connect(self.change_language)
        header.addWidget(self.language)
        layout.addLayout(header)
        subtitle = self._text_widget(QLabel, 'Sit quietly as the fragrance unfolds.')
        subtitle.setObjectName('subtitle')
        layout.addWidget(subtitle)

        transport = QFrame()
        transport.setObjectName('transport')
        controls = QHBoxLayout(transport)
        controls.setContentsMargins(18, 16, 18, 16)
        self.status = QLabel(self.tr_text('Stopped'))
        self.status.setObjectName('status')
        controls.addWidget(self.status, 1)
        self.start_button = self._text_widget(QPushButton, 'Begin')
        self.start_button.setObjectName('start')
        self.start_button.setMinimumWidth(100)
        self.start_button.clicked.connect(self.start_playback)
        self.stop_button = self._text_widget(QPushButton, 'End')
        self.stop_button.setMinimumWidth(90)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.request_stop)
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        layout.addWidget(transport)

        self.settings = QWidget()
        settings_layout = QHBoxLayout(self.settings)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(18)
        self.basic_settings = QWidget()
        self.basic_settings.setMinimumWidth(560)
        basic_layout = QVBoxLayout(self.basic_settings)
        basic_layout.setContentsMargins(0, 0, 0, 0)
        basic_layout.setSpacing(18)
        basic = self._text_widget(QGroupBox, 'Sound')
        form = QFormLayout(basic)
        form.setVerticalSpacing(16)
        self.profile = self._translated_combo([(profile['name'], key) for key, profile in PROFILES.items()]
                             + [('Custom', 'custom')])
        self.profile.currentIndexChanged.connect(self.apply_profile)
        form.addRow(self._text_widget(QLabel, 'Preset'), self.profile)
        self.profile_note = QLabel()
        self.profile_note.setWordWrap(True)
        self.profile_note.setObjectName('subtitle')
        form.addRow(self.profile_note)
        self.duration = self._translated_combo([('15 min', 15), ('20 min', 20), ('30 min', 30), ('60 min', 60), ('Untimed', 0)])
        self.duration.setCurrentIndex(1)
        form.addRow(self._text_widget(QLabel, 'Session length'), self.duration)
        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(55)
        self.volume_label = QLabel('55%')
        self.volume.valueChanged.connect(lambda value: self.volume_label.setText(f'{value}%'))
        volume_row = QWidget()
        volume_layout = QHBoxLayout(volume_row)
        volume_layout.setContentsMargins(0, 0, 0, 0)
        volume_layout.addWidget(self.volume, 1)
        volume_layout.addWidget(self.volume_label)
        form.addRow(self._text_widget(QLabel, 'Volume'), volume_row)
        self.instrument = self._translated_combo([(label, key) for key, label in INSTRUMENTS.items()])
        form.addRow(self._text_widget(QLabel, 'Instrument'), self.instrument)
        device_row = QWidget()
        device_layout = QHBoxLayout(device_row)
        device_layout.setContentsMargins(0, 0, 0, 0)
        self.device = QComboBox()
        device_layout.addWidget(self.device, 1)
        self.refresh_button = self._text_widget(QPushButton, 'Refresh')
        self.refresh_button.clicked.connect(self.refresh_devices)
        device_layout.addWidget(self.refresh_button)
        form.addRow(self._text_widget(QLabel, 'Output device'), device_row)
        ambient_row = QWidget()
        ambient_layout = QHBoxLayout(ambient_row)
        ambient_layout.setContentsMargins(0, 0, 0, 0)
        self.ambient = QSlider(Qt.Orientation.Horizontal)
        self.ambient.setRange(0, 100)
        self.ambient.setValue(round(DEFAULT_AMBIENT_RATIO * 100))
        self.ambient_label = QLabel(f'{self.ambient.value()}%')
        self.ambient_label.setMinimumWidth(42)
        self.ambient.valueChanged.connect(lambda value: self.ambient_label.setText(f'{value}%'))
        ambient_layout.addWidget(self.ambient, 1)
        ambient_layout.addWidget(self.ambient_label)
        form.addRow(self._text_widget(QLabel, 'Ocean ambience'), ambient_row)
        basic_layout.addWidget(basic)

        self.advanced = self._text_widget(QCheckBox, 'Advanced sound settings')
        basic_layout.addWidget(self.advanced)
        basic_layout.addStretch()
        settings_layout.addWidget(self.basic_settings, 1)
        tabs = QTabWidget()
        self.tabs = tabs
        tabs.setMinimumWidth(480)
        tabs.setVisible(False)
        self.advanced.toggled.connect(self.toggle_advanced)
        harmony = QWidget()
        harmonic_form = QFormLayout(harmony)
        harmonic_form.setContentsMargins(18, 18, 18, 18)
        harmonic_form.setVerticalSpacing(12)
        self.mode = self._translated_combo([('Stable', 'stable'), ('Balanced', 'balanced'), ('Dynamic', 'dynamic')])
        self.mode.setCurrentIndex(1)
        self.richness = self._translated_combo([('Minimal', 'minimal'), ('Normal', 'normal'), ('Rich', 'rich')])
        self.richness.setCurrentIndex(1)
        self.variety = self._translated_combo([('Limited', 'limited'), ('Normal', 'normal'), ('Extended', 'extended')])
        self.variety.setCurrentIndex(1)
        self.progression = self._translated_combo([('Circle of fifths', 'circle-of-fifths'), ('Pentatonic', 'pentatonic'),
                                  ('Modal', 'modal'), ('Random', 'random')])
        harmonic_form.addRow(self._text_widget(QLabel, 'Harmonic movement'), self.mode)
        harmonic_form.addRow(self._text_widget(QLabel, 'Harmonic richness'), self.richness)
        harmonic_form.addRow(self._text_widget(QLabel, 'Chord variety'), self.variety)
        harmonic_form.addRow(self._text_widget(QLabel, 'Progression style'), self.progression)
        self.evolution = self._text_widget(QCheckBox, 'Evolve harmonic movement over 30 minutes')
        self.evolution.setChecked(True)
        self.binaural = self._text_widget(QCheckBox, 'Subtle left/right frequency differences')
        self.binaural.setChecked(True)
        harmonic_form.addRow(self.evolution)
        harmonic_form.addRow(self.binaural)
        self._add_tab(harmony, 'Harmony & Tone')

        intervals = QWidget()
        grid = QGridLayout(intervals)
        grid.setContentsMargins(18, 18, 18, 18)
        grid.setVerticalSpacing(16)
        grid.addWidget(self._text_widget(QLabel, 'Instrument'), 0, 0)
        grid.addWidget(self._text_widget(QLabel, 'Minimum'), 0, 1)
        grid.addWidget(self._text_widget(QLabel, 'Maximum'), 0, 2)
        self.intervals = {}
        for row, (key, values) in enumerate(DEFAULT_INTERVALS.items(), 1):
            low, high = QSpinBox(), QSpinBox()
            for spin, value in zip((low, high), values):
                spin.setRange(0, 600)
                spin.setSuffix(self.tr_text(' s'))
                spin.setValue(value)
            low.valueChanged.connect(lambda value, other=high: other.setValue(max(value, other.value())))
            high.valueChanged.connect(lambda value, other=low: other.setValue(min(value, other.value())))
            self.intervals[key] = (low, high)
            grid.addWidget(self._text_widget(QLabel, INSTRUMENTS[key]), row, 0)
            grid.addWidget(low, row, 1)
            grid.addWidget(high, row, 2)
        note = self._text_widget(QLabel, 'Set the minimum to 0 seconds to disable an instrument.')
        note.setWordWrap(True)
        grid.addWidget(note, 5, 0, 1, 3)
        grid.setRowStretch(6, 1)
        self._add_tab(intervals, 'Instrument Intervals')
        settings_layout.addWidget(tabs, 1)
        layout.addWidget(self.settings, 1)
        self.sitting = QWidget()
        sitting_layout = QVBoxLayout(self.sitting)
        sitting_layout.addStretch()
        stillness = self._text_widget(QLabel, 'Just be here.')
        stillness.setObjectName('stillness')
        stillness.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sitting_layout.addWidget(stillness)
        self.session_note = QLabel()
        self.session_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.session_note.setObjectName('subtitle')
        sitting_layout.addWidget(self.session_note)
        sitting_layout.addStretch()
        self.sitting.setVisible(False)
        layout.addWidget(self.sitting, 1)
        footer = self._text_widget(QLabel, 'Sound fades in over 8 seconds and out over the final 20 seconds. Meditation presets open and close with a rin bell.')
        footer.setWordWrap(True)
        footer.setObjectName('subtitle')
        layout.addWidget(footer)
        self.setStyleSheet('''
            QMainWindow { background: #f4f0e8; }
            QWidget { font-family: "Segoe UI", "Yu Gothic UI", "Meiryo", sans-serif; font-size: 13px; color: #45463e; }
            QLabel#title { font-size: 29px; font-weight: 700; }
            QLabel#subtitle { color: #77766b; }
            QLabel#status { font-size: 15px; font-weight: 600; }
            QFrame#transport { background: #e9e4d9; border-radius: 12px; }
            QGroupBox { background: #fcfaf5; border: 1px solid #dcd5c7; border-radius: 8px; margin-top: 12px; padding: 16px 10px 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 5px; }
            QPushButton { background: #fcfaf5; border: 1px solid #c7beab; border-radius: 6px; padding: 9px 16px; }
            QPushButton:hover { background: #ede7db; }
            QPushButton#start { background: #696f57; color: #fcfaf5; border: 1px solid #696f57; font-weight: 600; }
            QPushButton#start:hover { background: #535b43; }
            QPushButton:disabled { background: #e7e2d8; color: #999487; border-color: #ded7ca; }
            QComboBox, QSpinBox { background: #fcfaf5; border: 1px solid #c7beab; border-radius: 5px; padding: 6px; min-height: 22px; }
            QTabWidget::pane { background: #fcfaf5; border: 1px solid #dcd5c7; border-radius: 6px; }
            QTabBar::tab { padding: 10px 18px; background: #e8e2d5; }
            QTabBar::tab:selected { background: #fcfaf5; color: #696f57; }
            QSlider::groove:horizontal { height: 5px; background: #dcd5c7; border-radius: 2px; }
            QSlider::handle:horizontal { background: #696f57; width: 16px; margin: -5px 0; border-radius: 8px; }
            QLabel#stillness { font-size: 32px; color: #696f57; padding: 20px; }
        ''')
        for control in (self.instrument, self.mode, self.richness, self.variety, self.progression):
            control.currentIndexChanged.connect(self.mark_custom)
        self.ambient.valueChanged.connect(self.mark_custom)
        self.evolution.toggled.connect(self.mark_custom)
        self.binaural.toggled.connect(self.mark_custom)
        for pair in self.intervals.values():
            for spin in pair:
                spin.valueChanged.connect(self.mark_custom)

    def toggle_advanced(self, expanded):
        if expanded:
            self._collapsed_width = self.width()
            self._collapsed_minimum_width = self.minimumWidth()
        self.tabs.setVisible(expanded)
        # Keep the existing height; give the advanced panel space on the right.
        self.settings.layout().activate()
        self.centralWidget().layout().activate()
        self.setMinimumWidth(self.centralWidget().minimumSizeHint().width()
                             if expanded else self._collapsed_minimum_width)
        if not self.isMaximized():
            if expanded:
                width = self._collapsed_width + self.tabs.sizeHint().width() + 18
                width = min(width, max(self.minimumWidth(),
                                       self.screen().availableGeometry().width()))
            else:
                width = self._collapsed_width
            self.resize(width, self.height())

    def tr_text(self, message, **values):
        return translate(message, self.language_code, **values)

    def _text_widget(self, widget_type, message):
        widget = widget_type(self.tr_text(message))
        setter = widget.setTitle if isinstance(widget, QGroupBox) else widget.setText
        self._text_bindings.append((setter, message))
        return widget

    def _translated_combo(self, items):
        widget = combo([(self.tr_text(label), value) for label, value in items])
        self._combo_bindings.append((widget, [label for label, _ in items]))
        return widget

    def _add_tab(self, widget, message):
        index = self.tabs.addTab(widget, self.tr_text(message))
        self._text_bindings.append((lambda text: self.tabs.setTabText(index, text), message))

    def set_status(self, message):
        self._status_message = message
        self.status.setText(self.tr_text(message))

    def _update_profile_note(self):
        profile = PROFILES.get(self.profile.currentData())
        message = profile['note'] if profile else 'Make quiet time with your own sounds and intervals.'
        self.profile_note.setText(self.tr_text(message))

    def change_language(self):
        self.language_code = self.language.currentData()
        self.preferences.setValue('language', self.language_code)
        self.setWindowTitle(self.tr_text('Incense & Stillness — Mindfulness BGM'))
        for setter, message in self._text_bindings:
            setter(self.tr_text(message))
        for widget, messages in self._combo_bindings:
            # Updating labels must never apply a preset or change audio settings.
            blocked = widget.blockSignals(True)
            try:
                for index, message in enumerate(messages):
                    widget.setItemText(index, self.tr_text(message))
            finally:
                widget.blockSignals(blocked)
        self.device.setItemText(0, self.tr_text('System default output'))
        for pair in self.intervals.values():
            for spin in pair:
                spin.setSuffix(self.tr_text(' s'))
        self._update_profile_note()
        self._update_session_note()
        self.set_status(self._status_message)

    def mark_custom(self):
        if not self._applying_profile:
            self.profile.setCurrentIndex(self.profile.findData('custom'))

    def apply_profile(self):
        key = self.profile.currentData()
        if key == 'custom':
            self.meditation_mode = False
            self._update_profile_note()
            return
        self._applying_profile = True
        try:
            profile = PROFILES[key]
            self.meditation_mode = True
            self.meditation_voice = profile.get('voice', 'drone')
            self.instrument.setCurrentIndex(self.instrument.findData(profile['instrument']))
            self.ambient.setValue(profile['ambient'])
            self.volume.setValue(profile['volume'])
            self.mode.setCurrentIndex(self.mode.findData('stable'))
            self.richness.setCurrentIndex(self.richness.findData('minimal'))
            self.variety.setCurrentIndex(self.variety.findData('limited'))
            self.progression.setCurrentIndex(0)
            self.evolution.setChecked(False)
            self.binaural.setChecked(False)
            for low, high in self.intervals.values():
                high.setValue(150)
                low.setValue(90)
            self._update_profile_note()
        finally:
            self._applying_profile = False

    def refresh_devices(self):
        previous = self.device.currentData()
        self.device.clear()
        self.device.addItem(self.tr_text('System default output'), None)
        try:
            for index, device in enumerate(sd.query_devices()):
                if device['max_output_channels'] >= 2:
                    self.device.addItem(device['name'], index)
            selected = self.device.findData(previous)
            if selected >= 0:
                self.device.setCurrentIndex(selected)
            self.set_status('Stopped')
        except Exception:
            self.set_status('Unable to detect output devices. Please refresh.')

    def _set_playing(self, playing):
        self.settings.setEnabled(not playing)
        self.settings.setVisible(not playing)
        self.sitting.setVisible(playing)
        self.start_button.setEnabled(not playing)
        self.stop_button.setEnabled(playing)

    def start_playback(self):
        if self.player.stream is not None:
            return
        try:
            selected = self.player.start(
                self.instrument.currentData(),
                {key: (low.value(), high.value()) for key, (low, high) in self.intervals.items()},
                self.ambient.value() / 100,
                duration_seconds=self.duration.currentData() * 60, volume=self.volume.value()/100,
                meditation_mode=self.meditation_mode, meditation_voice=self.meditation_voice,
                device=self.device.currentData(), modulation_mode=self.mode.currentData(),
                harmonic_richness=self.richness.currentData(), chord_variety=self.variety.currentData(),
                progression_style=self.progression.currentData(),
                session_evolution=self.evolution.isChecked(), binaural_enhancement=self.binaural.isChecked())
        except Exception as exc:
            self.set_status('Unable to start playback')
            QMessageBox.warning(self, self.tr_text('Playback unavailable'),
                                self.tr_text('Check the output device and audio connections.\n\n{error}', error=exc))
            return
        self._set_playing(True)
        self.set_status('Sitting')
        self._update_session_note()

    def _update_session_note(self):
        minutes = self.duration.currentData()
        message = (self.tr_text('{minutes}-minute sitting · a gentle ending', minutes=minutes)
                   if minutes else self.tr_text('At your own pace, without watching the time'))
        self.session_note.setText(message)

    def request_stop(self):
        """Fade out gently; check_playback stops once the sound has gone."""
        if self.player.stream is None:
            return
        self.player.begin_fade_out()
        self.stop_button.setEnabled(False)
        self.set_status('Ending gently…')

    def stop_playback(self):
        try:
            self.player.stop()
        except Exception as exc:
            QMessageBox.warning(self, self.tr_text('Error stopping playback'), str(exc))
        finally:
            self._set_playing(False)
            self.set_status('Stopped')

    def check_playback(self):
        if self.player.stream is None:
            return
        if self.player.finished:
            self.stop_playback()
            self.set_status('Session complete')
            return
        if self.player.faded_out:
            self.stop_playback()
            return
        try:
            active = self.player.stream.active
        except Exception:
            active = False
        if not active or self.player.error:
            error = self.player.error or self.tr_text('Output was interrupted. Check your audio connections.')
            self.stop_playback()
            self.set_status('Playback interrupted')
            QMessageBox.warning(self, self.tr_text('Playback interruption'), error)

    def closeEvent(self, event):
        self.monitor.stop()
        try:
            self.player.stop_gracefully(0.5)
        except Exception:
            pass  # The window is closing; the device is released either way.
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
