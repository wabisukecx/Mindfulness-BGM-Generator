"""
Streamlit-specific BGM generator class
Extends the base generator to support Streamlit's threading requirements
"""

import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx

from src.generator import MindfulnessBGM
from src.instruments_base import InstrumentConfig


class MindfulnessBGMStreamlit(MindfulnessBGM):
    """Streamlit-compatible mindfulness BGM generator"""
    
    def __init__(self, bell_config: InstrumentConfig, drum_config: InstrumentConfig, 
                 handpan_config: InstrumentConfig, crystal_bowl_config: InstrumentConfig,
                 ambient_ratio: float):
        # Initialize parent class but don't start schedulers yet
        super().__init__(bell_config, drum_config, handpan_config, crystal_bowl_config, ambient_ratio)
        
        # Override scheduler start to add Streamlit context
        self.stop_flag = threading.Event()
        self.schedulers = []
        self._start_streamlit_threads()
    
    def _start_streamlit_threads(self):
        """Start threads with Streamlit script context"""
        # Event scheduler
        scheduler_thread = threading.Thread(target=self._event_scheduler_with_stop, daemon=True)
        add_script_run_ctx(scheduler_thread)
        scheduler_thread.start()
        self.schedulers.append(scheduler_thread)
        
        # Instrument schedulers
        if self.bell_config.enabled:
            bell_thread = threading.Thread(target=self._bell_scheduler_with_stop, daemon=True)
            add_script_run_ctx(bell_thread)
            bell_thread.start()
            self.schedulers.append(bell_thread)
            
        if self.drum_config.enabled:
            drum_thread = threading.Thread(target=self._drum_scheduler_with_stop, daemon=True)
            add_script_run_ctx(drum_thread)
            drum_thread.start()
            self.schedulers.append(drum_thread)
            
        if self.handpan_config.enabled:
            handpan_thread = threading.Thread(target=self._handpan_scheduler_with_stop, daemon=True)
            add_script_run_ctx(handpan_thread)
            handpan_thread.start()
            self.schedulers.append(handpan_thread)
            
        if self.crystal_bowl_config.enabled:
            crystal_bowl_thread = threading.Thread(target=self._crystal_bowl_scheduler_with_stop, daemon=True)
            add_script_run_ctx(crystal_bowl_thread)
            crystal_bowl_thread.start()
            self.schedulers.append(crystal_bowl_thread)
    
    def stop(self):
        """Stop all threads gracefully"""
        self.stop_flag.set()
        for thread in self.schedulers:
            if thread.is_alive():
                thread.join(timeout=1.0)
        self.schedulers.clear()
    
    def _event_scheduler_with_stop(self):
        """Event scheduler with stop flag support"""
        import random
        from src.config import MIN_EVENT_INTERVAL, MAX_EVENT_INTERVAL
        
        while not self.stop_flag.is_set():
            wait_time = random.uniform(MIN_EVENT_INTERVAL, MAX_EVENT_INTERVAL)
            if self.stop_flag.wait(wait_time):
                break
            
            events = [
                self._change_sound_type,
                self._change_chord,
                self._change_volume,
                self._change_both
            ]
            
            event = random.choice(events)
            event()
    
    def _bell_scheduler_with_stop(self):
        """Bell scheduler with stop flag support"""
        import random
        import time
        from src.config import BASE_FREQS
        
        while not self.stop_flag.is_set():
            wait_time = random.uniform(self.bell_config.min_interval, 
                                     self.bell_config.max_interval)
            if self.stop_flag.wait(wait_time):
                break
            
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
                    if self.stop_flag.wait(random.uniform(1.0, 4.0)):
                        break
    
    def _drum_scheduler_with_stop(self):
        """Slit drum scheduler with stop flag support"""
        import random
        import time
        from src.config import BASE_FREQS
        
        if self.stop_flag.wait(5):
            return
        
        while not self.stop_flag.is_set():
            wait_time = random.uniform(self.drum_config.min_interval, 
                                     self.drum_config.max_interval)
            if self.stop_flag.wait(wait_time):
                break
            
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
                    if self.stop_flag.wait(random.uniform(1.0, 3.0)):
                        break
    
    def _handpan_scheduler_with_stop(self):
        """Handpan scheduler with stop flag support"""
        import random
        import time
        from src.config import BASE_FREQS
        
        if self.stop_flag.wait(10):
            return
        
        while not self.stop_flag.is_set():
            wait_time = random.uniform(self.handpan_config.min_interval, 
                                     self.handpan_config.max_interval)
            if self.stop_flag.wait(wait_time):
                break
            
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
                    if self.stop_flag.wait(random.uniform(1.0, 3.5)):
                        break
    
    def _crystal_bowl_scheduler_with_stop(self):
        """Crystal singing bowl scheduler with stop flag support"""
        import random
        from src.config import BASE_FREQS
        
        if self.stop_flag.wait(8):
            return
        
        while not self.stop_flag.is_set():
            wait_time = random.uniform(self.crystal_bowl_config.min_interval, 
                                     self.crystal_bowl_config.max_interval)
            if self.stop_flag.wait(wait_time):
                break
            
            with self.lock:
                if self.chord:
                    base_freq = random.choice(self.chord)
                    freq = base_freq * random.choice([1, 1.5, 2, 3])
                else:
                    freq = random.choice(BASE_FREQS) * random.choice([1, 2, 3])
                self.crystal_bowl.trigger(freq)
