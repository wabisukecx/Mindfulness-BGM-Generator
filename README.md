# Mindfulness BGM Generator

A dynamic ambient music generator designed for meditation and mindfulness practices. Creates continuous, evolving soundscapes with harmonious chords, natural ambient sounds, and traditional meditation instruments.

## Features

- **Dynamic Sound Generation**: Continuously evolving harmonies and textures
- **Multiple Sound Types**:
  - Harmonic-rich tones
  - Pure tones
  - Soft pad sounds
  - Warm tones
  - Bell-like sounds
- **Meditation Instruments**:
  - Tibetan bells (Tingsha)
  - Slit drums
  - Handpan sounds
- **Natural Ambient Sounds**: Ocean waves and wind
- **Breathing Rhythm Synchronization**: Subtle volume modulation synced to typical breathing patterns
- **Real-time Audio Processing**: Low-latency sound generation
- **Customizable Parameters**: Adjust instrument intervals and ambient mix ratio
- **Smooth Transitions**: Crossfading between chords and sound types
- **Simple Reverb Effect**: Adds spatial depth to the soundscape

## Requirements

- Python 3.7+
- NumPy
- SoundDevice
- pyaudio (optional, for some systems)

## Installation

1. Clone this repository:

```bash
git clone https://github.com/wabisukecx/Mindfulness-BGM-Generator.git
cd Mindfulness-BGM-Generator
```

1. Install required packages:

```bash
pip install numpy sounddevice
```

## Usage

### Basic Usage

Run with default settings:

```bash
python mindfulness_bgm.py
```

### Command-line Options

- `--bell`: Tibetan bell interval in seconds (e.g., '15-45' or '30' or '0' to disable)
- `--drum`: Slit drum interval in seconds (e.g., '8-25' or '15' or '0' to disable)  
- `--handpan`: Handpan interval in seconds (e.g., '12-30' or '20' or '0' to disable)
- `--ambient`: Ambient sound ratio (0-1, default: 0.15, '0' to disable)

### Examples

Disable Tibetan bells:

```bash
python mindfulness_bgm.py --bell 0
```

Set slit drum interval to 10-20 seconds:

```bash
python mindfulness_bgm.py --drum 10-20
```

Set handpan to fixed 30-second intervals:

```bash
python mindfulness_bgm.py --handpan 30
```

Customize all parameters:

```bash
python mindfulness_bgm.py --bell 20-60 --drum 0 --handpan 15-25
```

Ambient pad sounds only (no percussion):

```bash
python mindfulness_bgm.py --bell 0 --drum 0 --handpan 0
```

No ambient sounds, only musical elements:

```bash
python mindfulness_bgm.py --ambient 0
```

50% ambient sounds:

```bash
python mindfulness_bgm.py --ambient 0.5
```

## Technical Details

### Audio Processing

- **Sample Rate**: 44100 Hz
- **Buffer Size**: 1024 frames
- **Channels**: Stereo
- **Output Format**: Float32

### Musical Elements

- **Scale**: A Pentatonic (A, C, E, A, C)
- **Chord Types**:
  - Open fifths
  - Major/Minor triads
  - Sus2/Sus4 chords
  - 7th chords
  - Add9 chords
  - Quartal harmonies

### Sound Generation

- Pure sine wave synthesis
- Harmonic addition for timbral richness
- Exponential envelope generators
- Phase modulation for stereo width
- Soft limiting to prevent clipping

## Architecture

The application uses a multi-threaded architecture:

- Main thread: Audio callback and signal processing
- Event scheduler thread: Manages chord and sound type changes
- Instrument scheduler threads: Controls percussion timing

## Customization

You can modify the following constants in the code:

- `VOLUME`: Overall output volume
- `MIN_EVENT_INTERVAL`: Minimum time between sound changes
- `MAX_EVENT_INTERVAL`: Maximum time between sound changes
- `FADE_TIME`: Crossfade duration
- `BREATH_CYCLE`: Breathing rhythm frequency
- `BASE_FREQS`: Musical scale frequencies

## Known Issues

- On some systems, you may need to specify the audio device explicitly
- Very low buffer sizes may cause audio glitches on slower systems

## Contributing

Contributions are welcome! Please feel free to submit pull requests or create issues for bugs and feature requests.

## License

[MIT License](LICENSE)

## Acknowledgments

Inspired by traditional meditation music and modern sound therapy techniques. Special thanks to the Python audio processing community.
This README provides comprehensive documentation for users of all technical levels, from basic usage to advanced customization options.
