"""Shared, cancellable scheduler lifecycle for local audio generators."""

import threading


class SchedulerLifecycle:
    def _init_schedulers(self):
        self.stop_flag = threading.Event()
        self.schedulers = []
        self._lifecycle_lock = threading.Lock()

    def start(self):
        """Start once; call only after the output device has opened."""
        with self._lifecycle_lock:
            if self.schedulers:
                return
            self.stop_flag.clear()
            targets = [self._event_scheduler]
            if getattr(self, 'session_evolution', False):
                targets.append(self._evolution_scheduler)
            for name in ('bell', 'drum', 'handpan', 'crystal_bowl'):
                if getattr(self, name + '_config').enabled:
                    targets.append(getattr(self, '_' + name + '_scheduler'))
            for target in targets:
                thread = threading.Thread(target=target, daemon=True)
                self.schedulers.append(thread)
                thread.start()

    def stop(self):
        """Wake sleeping workers and wait for them before discarding references."""
        with self._lifecycle_lock:
            self.stop_flag.set()
            for thread in self.schedulers:
                thread.join()
            self.schedulers.clear()
