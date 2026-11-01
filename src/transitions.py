"""Chord/timbre crossfades and level changes, interpolated sample by sample.

Shared by the desktop and Raspberry Pi generators, which provide: lock,
sample_rate, fade_time, phase, current/target volume and transition speed,
and current/next sound type and chord.
"""
import numpy as np


class TransitionMixin:
    def _init_transitions(self):
        self.fade_ramp = None    # Crossfade amount for each sample of this block
        self.volume_ramp = None  # Level for each sample of this block

    def _ready_for_change(self) -> bool:
        """Commit a finished crossfade; refuse a new one while one is under way.

        Restarting a crossfade midway would jump straight back to the old sound.
        """
        if self.is_transitioning and self.fade_progress >= 1.0:
            self._commit_transition()
        return not self.is_transitioning

    def _commit_transition(self):
        self.is_transitioning = False
        self.fade_ramp = None
        if self.next_sound_type:
            self.current_sound_type = self.next_sound_type
            self.sound_start_frame = self.next_sound_start_frame
            self.next_sound_type = None
        if self.next_chord:
            self.chord = self.next_chord
            self.next_chord = None

    def _update_transitions(self, frames: int):
        """Advance volume and crossfade, keeping per-sample ramps for this block.

        A crossfade that reaches 1.0 is committed at the start of the next
        block, so its last block still ends exactly on the new sound.
        """
        with self.lock:
            start = self.current_volume
            if self.current_volume != self.target_volume:
                if abs(self.current_volume - self.target_volume) < 0.01:
                    self.current_volume = self.target_volume
                else:
                    direction = 1 if self.target_volume > self.current_volume else -1
                    self.current_volume += direction * self.volume_transition_speed * frames
                    self.current_volume = float(np.clip(self.current_volume, 0.3, 0.8))
            self.volume_ramp = np.linspace(start, self.current_volume, frames + 1)[1:]

            if self.is_transitioning and self.fade_progress >= 1.0:
                self._commit_transition()
            if self.is_transitioning:
                start = self.fade_progress
                self.fade_progress = min(1.0, start + frames / (self.fade_time * self.sample_rate))
                self.fade_ramp = np.linspace(start, self.fade_progress, frames + 1)[1:]
            else:
                self.fade_ramp = None

    def _crossfade_amount(self, frames: int):
        """Per-sample crossfade for this block, or the scalar progress outside rendering."""
        if self.fade_ramp is not None and len(self.fade_ramp) == frames:
            return self.fade_ramp
        return self.fade_progress

    def _level_envelope(self, frames: int, breath_cycle: float) -> np.ndarray:
        """Breathing rhythm times the volume ramp, both sample by sample."""
        breath_t = self.breath_phase + np.arange(frames) / self.sample_rate
        self.breath_phase += frames / self.sample_rate
        breath = 0.97 + 0.03 * np.sin(2 * np.pi * breath_cycle * breath_t)
        volume = self.volume_ramp if self.volume_ramp is not None else self.current_volume
        return breath * volume
