#!/usr/bin/env python3
"""
Mindfulness BGM Generator - Main Entry Point
Enhanced Version with Extended Harmonic Progression and Chord Variations
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
    """Parse command line arguments with extended harmonic options"""
    parser = argparse.ArgumentParser(description="Mindfulness BGM Generator - Enhanced Harmonic Version")
    
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
    parser.add_argument("--modulation-mode", type=str, default="balanced",
                      choices=["stable", "balanced", "dynamic"],
                      help="Modulation behavior: stable (calm), balanced, or dynamic (active)")
    parser.add_argument("--harmonic-richness", type=str, default="normal",
                      choices=["minimal", "normal", "rich"],
                      help="Harmonic richness level: minimal (fewer harmonics), normal, rich (more harmonics)")
    parser.add_argument("--chord-variety", type=str, default="normal",
                      choices=["limited", "normal", "extended"],
                      help="Chord variety: limited (fewer chord types), normal, extended (more chord types)")
    parser.add_argument("--progression-style", type=str, default="circle-of-fifths",
                      choices=["circle-of-fifths", "pentatonic", "modal", "random"],
                      help="Chord progression style: circle-of-fifths (traditional), pentatonic (eastern), modal (evocative), random")
    parser.add_argument("--session-evolution", action="store_true", default=True,
                      help="Enable gradual evolution of chord progressions over the session duration")
    parser.add_argument("--binaural-enhancement", action="store_true", default=True,
                      help="Enhance stereo field with subtle binaural-inspired frequency differences")
    
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


def create_harmonic_config(args):
    """Create harmonic configuration based on extended arguments"""
    harmonic_config = {
        'harmonic_richness': args.harmonic_richness,
        'chord_variety': args.chord_variety,
        'progression_style': args.progression_style,
        'session_evolution': args.session_evolution,
        'binaural_enhancement': args.binaural_enhancement
    }
    return harmonic_config


def display_settings(instrument_configs, ambient_ratio, modulation_mode, harmonic_config):
    """Display current settings with extended harmonic information"""
    print("Starting mindfulness BGM (Enhanced Harmonic Version)...")
    print("\nSettings:")
    print("- Four meditation instruments with enhanced harmonics")
    print("- Enhanced reverb effect (0.3s)")
    print("- CD quality audio (48kHz)")
    print("- Improved stereo separation")
    print("- Smoother transitions (3s)")
    print(f"- Modulation mode: {modulation_mode}")
    
    # Modulation mode descriptions
    mode_descriptions = {
        "stable": "Minimal modulation for deep meditation",
        "balanced": "Balanced modulation for regular practice",
        "dynamic": "Active modulation for energetic sessions"
    }
    print(f"  {mode_descriptions[modulation_mode]}")
    
    print("\nHarmonic settings:")
    
    richness_descriptions = {
        "minimal": "Minimal harmonics for pure tones",
        "normal": "Standard harmonic content",
        "rich": "Rich harmonic content for fuller sound"
    }
    print(f"- Harmonic richness: {harmonic_config['harmonic_richness']} ({richness_descriptions[harmonic_config['harmonic_richness']]})")
    
    chord_variety_descriptions = {
        "limited": "Limited chord types for stability",
        "normal": "Standard selection of mindfulness-oriented chords",
        "extended": "Extended chord vocabulary for variety"
    }
    print(f"- Chord variety: {harmonic_config['chord_variety']} ({chord_variety_descriptions[harmonic_config['chord_variety']]})")
    
    progression_descriptions = {
        "circle-of-fifths": "Natural progressions based on circle of fifths",
        "pentatonic": "Pentatonic-based movement for Eastern feel",
        "modal": "Modal progressions for evocative atmosphere",
        "random": "Randomized progressions for maximum variety"
    }
    print(f"- Progression style: {harmonic_config['progression_style']} ({progression_descriptions[harmonic_config['progression_style']]})")
    
    if harmonic_config['session_evolution']:
        print("- Session evolution: ENABLED (progressions evolve over time)")
    else:
        print("- Session evolution: DISABLED")
        
    if harmonic_config['binaural_enhancement']:
        print("- Binaural enhancement: ENABLED (subtle frequency differences between stereo channels)")
    else:
        print("- Binaural enhancement: DISABLED")
    
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
    """Main entry point with enhanced harmonic options"""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Set up instruments
        instrument_configs, selected_instrument = setup_instruments(args)
        
        # Parse ambient sound settings
        ambient_ratio = parse_ambient_value(args.ambient)
        
        # Create harmonic configuration
        harmonic_config = create_harmonic_config(args)
        
        # Display settings
        display_settings(instrument_configs, ambient_ratio, args.modulation_mode, harmonic_config)
        
        # Configure audio device
        sd.default.samplerate = SAMPLE_RATE
        sd.default.blocksize = BUFFER_FRAMES
        sd.default.channels = 2
        
        # Create generator with enhanced options
        generator = MindfulnessBGM(
            instrument_configs['bell'],
            instrument_configs['drum'],
            instrument_configs['handpan'],
            instrument_configs['crystal_bowl'],
            ambient_ratio,
            modulation_mode=args.modulation_mode,
            harmonic_richness=harmonic_config['harmonic_richness'],
            chord_variety=harmonic_config['chord_variety'],
            progression_style=harmonic_config['progression_style'],
            session_evolution=harmonic_config['session_evolution'],
            binaural_enhancement=harmonic_config['binaural_enhancement']
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