"""Audio-device ownership, independent of the GUI framework."""
import random
import time

import numpy as np
import sounddevice as sd

from src.config import SAMPLE_RATE, BUFFER_FRAMES
from src.generator import MindfulnessBGM
from src.instruments_base import InstrumentConfig, VoicePool
from src.meditation import Rin, RIN_FREQ
from src.session import SessionEnvelope


STOP_FADE_SECONDS = 1.5  # A stop fades out rather than cutting the waveform (a click)


class Playback:
    def __init__(self):
        self.generator = None
        self.stream = None
        self.error = None
        self.session = None
        self.volume = 1.0
        self.rin = None
        self._closing_rin_struck = False
        self._stop_fade = 0  # Samples in the stop fade-out, once one is requested
        self._stop_fade_position = 0

    def start(self, instrument, intervals, ambient_ratio, device=None,
              duration_seconds=0, volume=1.0, **harmonics):
        if self.stream is not None:
            return
        self.error = None
        self._stop_fade = self._stop_fade_position = 0
        self.volume = max(0.0, min(1.0, volume))
        self.session = SessionEnvelope(SAMPLE_RATE, duration_seconds)
        if instrument == 'random':
            candidates = [key for key, (low, _) in intervals.items() if low > 0]
            instrument = random.choice(candidates) if candidates else 'none'
        configs = [InstrumentConfig(*intervals[key], enabled=(instrument == key))
                   for key in ('bell', 'drum', 'handpan', 'crystal_bowl')]
        self.generator = MindfulnessBGM(*configs, ambient_ratio,
                                       auto_start=False, **harmonics)
        # The quiet presets open with a rin, and close with one as the sound fades
        self.rin = None
        self._closing_rin_struck = False
        if harmonics.get('meditation_mode'):
            self.rin = VoicePool(Rin, SAMPLE_RATE, max_voices=2)
            self.rin.trigger(RIN_FREQ)
        try:
            self.stream = sd.OutputStream(
                callback=self._callback, samplerate=SAMPLE_RATE,
                blocksize=BUFFER_FRAMES, channels=2, dtype='float32', device=device)
            self.stream.start()
            self.generator.start()
        except Exception:
            try:
                self.stop()
            except Exception:
                pass  # Preserve the original device error.
            raise
        return instrument

    def _callback(self, outdata, frames, time_info, status):
        try:
            self.generator.callback(outdata, frames, time_info, status)
            outdata *= (self.session.gains(frames) * self.volume)[:, None]
            if self.rin is not None:
                self._add_rin(outdata, frames)
            if self._stop_fade:
                positions = self._stop_fade_position + np.arange(frames)
                gain = np.cos(np.clip(positions / self._stop_fade, 0, 1) * np.pi / 2) ** 2
                gain[positions >= self._stop_fade] = 0
                outdata *= gain[:, None]
                self._stop_fade_position += frames
        except Exception as exc:
            outdata.fill(0)
            self.error = str(exc)
            raise sd.CallbackAbort from exc

    def _add_rin(self, outdata, frames):
        """Mix the rin outside the session envelope, so the fade-in cannot mute
        the opening strike; the closing strike sounds as the fade-out begins."""
        session = self.session
        if (session.duration and not self._closing_rin_struck
                and session.position >= session.duration - session.fade_out):
            self.rin.trigger(RIN_FREQ)
            self._closing_rin_struck = True
        outdata += (self.rin.generate(frames) * self.volume)[:, None]

    @property
    def finished(self):
        """A timed session ends once its sound has faded and the rin has died away."""
        return (self.session is not None and self.session.finished
                and not (self.rin is not None and self.rin.is_playing))

    def begin_fade_out(self, seconds=STOP_FADE_SECONDS):
        """Start fading everything out; call stop() once faded_out is true."""
        if self.stream is not None and not self._stop_fade:
            self._stop_fade_position = 0
            self._stop_fade = max(1, round(seconds * SAMPLE_RATE))

    @property
    def faded_out(self):
        return bool(self._stop_fade) and self._stop_fade_position >= self._stop_fade

    def stop_gracefully(self, seconds=STOP_FADE_SECONDS):
        """Fade out, wait for it, then stop. Gives up waiting if the audio
        device stops calling back (it has failed, or there is none)."""
        self.begin_fade_out(seconds)
        last, last_change = self._stop_fade_position, time.monotonic()
        while self.stream is not None and not self.faded_out:
            time.sleep(0.01)
            if self._stop_fade_position != last:
                last, last_change = self._stop_fade_position, time.monotonic()
            elif time.monotonic() - last_change > 0.1:
                break
        self.stop()

    def stop(self):
        stream, generator = self.stream, self.generator
        try:
            if stream is not None:
                try:
                    stream.stop()
                finally:
                    stream.close()
        finally:
            if generator is not None:
                generator.stop()
            self.stream = None
            self.generator = None
