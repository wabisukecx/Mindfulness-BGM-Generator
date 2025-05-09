#!/usr/bin/env python3
"""
Mindfulness BGM Generator - Main Entry Point
High Quality Version with Modular Structure
"""

import sounddevice as sd
import time
import random
import argparse

from src.config import SAMPLE_RATE, BUFFER_FRAMES, DEFAULT_AMBIENT_RATIO
from src.instruments_base import InstrumentConfig
from src.generator import MindfulnessBGM
from src.utils import parse_instrument_interval, parse_ambient_value


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Mindfulness BGM Generator - High Quality Version")
    
    parser.add_argument("--bell", type=str, default="15-45",
                      help="Tibetan bell interval in seconds")
    parser.add_argument("--drum", type=str, default="8-25",
                      help="Slit drum interval in seconds")
    parser.add_argument("--handpan", type=str, default="12-30",
                      help="Handpan interval in seconds")
    parser.add_argument("--crystal-bowl", type=str, default="25-60",
                      help="Crystal singing bowl interval in seconds")
    parser.add_argument("--ambient", type=str, default=str(DEFAULT_AMBIENT_RATIO),
                      help=f"Ambient sound ratio (0-1, default: {DEFAULT_AMBIENT_RATIO})")
    parser.add_argument("--instrument", type=str, default=None,
                      help="Specify instrument: bell, drum, handpan, crystal-bowl")
    
    return parser.parse_args()


def setup_instruments(args):
    """Set up instrument configurations based on arguments"""
    # Parse instrument settings
    bell_min, bell_max = parse_instrument_interval(args.bell)
    drum_min, drum_max = parse_instrument_interval(args.drum)
    handpan_min, handpan_max = parse_instrument_interval(args.handpan)
    crystal_bowl_min, crystal_bowl_max = parse_instrument_interval(args.crystal_bowl)
    
    # Available instruments
    instruments = ['bell', 'drum', 'handpan', 'crystal-bowl']
    
    # Choose single instrument
    if args.instrument:
        selected_instrument = args.instrument.lower()
        if selected_instrument not in instruments:
            raise ValueError(f"Invalid instrument: {selected_instrument}. "
                           f"Available instruments: {', '.join(instruments)}")
    else:
        selected_instrument = random.choice(instruments)
    
    # Enable only the selected instrument
    bell_enabled = (selected_instrument == 'bell' and bell_min > 0)
    drum_enabled = (selected_instrument == 'drum' and drum_min > 0)
    handpan_enabled = (selected_instrument == 'handpan' and handpan_min > 0)
    crystal_bowl_enabled = (selected_instrument == 'crystal-bowl' and crystal_bowl_min > 0)
    
    # Create configurations
    configs = {
        'bell': InstrumentConfig(bell_min, bell_max, bell_enabled),
        'drum': InstrumentConfig(drum_min, drum_max, drum_enabled),
        'handpan': InstrumentConfig(handpan_min, handpan_max, handpan_enabled),
        'crystal_bowl': InstrumentConfig(crystal_bowl_min, crystal_bowl_max, crystal_bowl_enabled),
    }
    
    return configs, selected_instrument


def display_settings(instrument_configs, ambient_ratio):
    """Display current settings"""
    print("Starting mindfulness BGM (High Quality Version)...")
    print("\nSettings:")
    print("- Four meditation instruments with enhanced harmonics")
    print("- Enhanced reverb effect (0.3s)")
    print("- CD quality audio (48kHz)")
    print("- Improved stereo separation")
    print("- Smoother transitions (3s)")
    
    print("\nInstrument settings:")
    
    config_names = [
        ("Tibetan bell", 'bell'),
        ("Slit drum", 'drum'),
        ("Handpan", 'handpan'),
        ("Crystal singing bowl", 'crystal_bowl')
    ]
    
    for display_name, config_key in config_names:
        config = instrument_configs[config_key]
        if config.enabled:
            print(f"- {display_name}: {config.min_interval:.1f}-{config.max_interval:.1f} seconds (SELECTED)")
        else:
            print(f"- {display_name}: DISABLED")
    
    if ambient_ratio > 0:
        print(f"- Ambient sounds: {ambient_ratio:.1%} mix ratio")
    else:
        print("- Ambient sounds: DISABLED")
    
    print("\nPress Ctrl+C to stop.")


def main():
    """Main entry point"""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Set up instruments
        instrument_configs, selected_instrument = setup_instruments(args)
        
        # Parse ambient sound settings
        ambient_ratio = parse_ambient_value(args.ambient)
        
        # Display settings
        display_settings(instrument_configs, ambient_ratio)
        
        # Configure audio device
        sd.default.samplerate = SAMPLE_RATE
        sd.default.blocksize = BUFFER_FRAMES
        sd.default.channels = 2
        
        # Create generator
        generator = MindfulnessBGM(
            instrument_configs['bell'],
            instrument_configs['drum'],
            instrument_configs['handpan'],
            instrument_configs['crystal_bowl'],
            ambient_ratio
        )
        
        # Start audio stream
        with sd.OutputStream(callback=generator.callback):
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nGracefully stopping...")
                
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())