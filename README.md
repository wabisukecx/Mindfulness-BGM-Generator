# Mindfulness BGM Generator

Dynamic ambient music generator designed for meditation and mindfulness practices. Creates continuous, evolving soundscapes with harmonious chords, natural ambient sounds, and traditional meditation instruments.

## Overview

This application provides a real-time audio generation system that creates continuously evolving ambient music suitable for meditation and mindfulness practices. It combines synthesized harmonies, natural sounds, and traditional meditation instruments to create a peaceful soundscape.

## Features

- Dynamic sound generation with continuously evolving harmonies and textures
- Multiple sound types (harmonic-rich tones, pure tones, soft pad sounds, warm tones, bell-like sounds)
- Four meditation instruments (Tibetan bells, slit drums, handpan sounds, crystal singing bowls)
- Ocean wave sounds with rich textures
- Subtle volume modulation aligned with breath cycles
- Real-time audio processing with low latency
- Customizable parameters (instrument intervals, ambient mix ratio)
- Smooth transitions between sound types and chords
- Reverb effect for spatial depth
- Adaptive modulation modes:
  - Stable: Minimal harmonic changes for deep meditation
  - Balanced: Natural progression for regular practice
  - Dynamic: Active variations for energetic sessions

### Extended Harmonic Features (added in v1.6.0)

- **Harmonic Richness Levels**:
  - minimal: Fewer harmonics for purer tones
  - normal: Standard harmonic content
  - rich: More harmonics for fuller sound

- **Chord Variety Options**:
  - limited: Basic chord types for stability
  - normal: Standard selection of mindfulness-oriented chords
  - extended: Rich chord vocabulary for variety

- **Progression Styles**:
  - circle-of-fifths: Traditional harmonically-related progressions
  - pentatonic: Eastern-inspired scale movements
  - modal: Evocative progressions based on modal harmony
  - random: Free progressions for maximum variety

- **Session Evolution**: Gradual changes in harmonic behavior over 30-minute sessions
- **Binaural Enhancement**: Subtle frequency differences between channels for enhanced stereo experience

## Project Structure

```bash
.
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── packages.txt              # System-level dependencies
├── main.py                   # Command-line version
├── app.py                    # Streamlit web interface
├── main_rpiz2w.py            # Raspberry Pi Zero 2W optimized version
└── src/                      # Source code modules
    ├── __init__.py
    ├── config.py             # High-quality audio settings
    ├── config_rpiz2w.py      # RPi Zero 2W optimized settings
    ├── generator.py          # Main BGM generator class
    ├── generator_rpiz2w.py   # RPi optimized generator
    ├── generator_streamlit.py # Streamlit-compatible generator
    ├── synthesizer.py        # Sound synthesis functions
    ├── synthesizer_rpiz2w.py # RPi optimized synthesizer
    ├── sound_types.py        # Sound type definitions
    ├── instruments_base.py   # Base instrument classes
    ├── instruments.py        # Meditation instrument implementations
    ├── instruments_rpiz2w.py # RPi optimized instruments
    ├── ambient_sounds.py     # Ambient sound generation
    ├── ambient_sounds_rpiz2w.py # RPi optimized ambient
    ├── effects.py            # Audio effects processing
    ├── effects_rpiz2w.py     # RPi optimized effects
    └── utils.py              # Utility functions
```

## Requirements

- Python 3.7+
- Audio output device
- 4GB RAM (recommended)
- Dual-core CPU (minimum)

### Python Dependencies

```python
numpy        # Numerical computing
sounddevice  # Audio I/O
streamlit    # Web application framework (for app.py)
pyaudio      # Audio I/O (optional, for some systems)
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/wabisukecx/Mindfulness-BGM-Generator.git
cd Mindfulness-BGM-Generator
```

### 2. Install System Dependencies

#### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-all-dev
```

#### macOS

```bash
brew install portaudio
```

#### Windows

For Windows users, PyAudio installation may require additional steps:

```bash
pip install pipwin
pipwin install pyaudio
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Command-line Version (main.py)

#### Basic Usage

```bash
python main.py
```

#### Command-line Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--instrument` | Choose specific instrument | Random | `--instrument bell` |
| `--bell` | Tibetan bell interval (seconds) | `15-45` | `--bell 30` or `--bell 20-40` |
| `--drum` | Slit drum interval (seconds) | `8-25` | `--drum 15` or `--drum 10-20` |
| `--handpan` | Handpan interval (seconds) | `12-30` | `--handpan 20` or `--handpan 15-25` |
| `--crystal-bowl` | Crystal bowl interval (seconds) | `25-60` | `--crystal-bowl 40` or `--crystal-bowl 30-50` |
| `--ambient` | Ambient sound ratio (0-1) | `0.3` | `--ambient 0.5` |
| `--modulation-mode` | Harmonic modulation behavior | `balanced` | `--modulation-mode stable` |
| `--harmonic-richness` | Harmonic richness level | `normal` | `--harmonic-richness rich` |
| `--chord-variety` | Chord variety level | `normal` | `--chord-variety extended` |
| `--progression-style` | Chord progression style | `circle-of-fifths` | `--progression-style modal` |
| `--session-evolution` | Enable session evolution | `True` | `--session-evolution` |
| `--binaural-enhancement` | Enable binaural enhancements | `True` | `--binaural-enhancement` |

#### Usage Examples

Use only Tibetan bells:

```bash
python main.py --instrument bell
```

Use only handpan with custom interval:

```bash
python main.py --instrument handpan --handpan 10-20
```

Ambient sounds only (no percussion):

```bash
python main.py --bell 0 --drum 0 --handpan 0 --crystal-bowl 0
```

Deep meditation with minimal harmonic changes:

```bash
python main.py --modulation-mode stable
```

Active meditation with dynamic variations:

```bash
python main.py --modulation-mode dynamic --ambient 0.5
```

Enhanced harmonics for fuller sound:

```bash
python main.py --harmonic-richness rich --chord-variety extended
```

Eastern-inspired meditation experience:

```bash
python main.py --progression-style pentatonic --instrument crystal-bowl
```

Modal progression with binaural enhancement:

```bash
python main.py --progression-style modal --binaural-enhancement
```

### Web Interface (app.py)

Run the Streamlit application:

```bash
streamlit run app.py
```

The web app provides:

- Start/Stop controls
- Instrument selection dropdown
- Ambient sound mix slider
- Advanced settings for instrument intervals

### Raspberry Pi Zero 2W Version (main_rpiz2w.py)

Optimized for lower-power devices:

```bash
python main_rpiz2w.py
```

This version features:

- Lower memory footprint
- Optimized audio processing
- Same command-line options as main.py

## Technical Details

### Audio Processing

- Sample Rate: 16kHz (optimized) / 48kHz (standard)
- Buffer Size: 1024 frames
- Channels: Stereo
- Output Format: Float32

### Musical Elements

- Scale: A Pentatonic (A, C, E, A, C)
- Chord Types (mindfulness-oriented):
  - Single notes (drone)
  - Octaves
  - Perfect fifths
  - Perfect fourths
  - Quartal harmonies
  - Sus2 chords
  - Open voicings
  - Various extended chord types

- Modulation System:
  - Stable Mode: 80% probability of staying on same root, minimal changes
  - Balanced Mode: 60% root stability, preferring fifths and fourths
  - Dynamic Mode: 30% root stability, free modulation across all roots
  - Natural harmonic progressions based on circle of fifths
  - Time-based probability with gradual calming effect

- Harmonic Richness Levels:
  - minimal: 2-4 harmonics per sound type
  - normal: 3-7 harmonics per sound type
  - rich: 5-9 harmonics per sound type

- Chord Variety Levels:
  - limited: 5 basic chord types
  - normal: 10-12 chord types
  - extended: 15-25 chord types

- Progression Styles:
  - circle-of-fifths: Harmonic relationships based on traditional progressions
  - pentatonic: Eastern-inspired scale movements for a more meditative quality
  - modal: Evocative progressions inspired by modal harmony (Dorian, Mixolydian, etc.)
  - random: Free progressions for maximum variety

## Troubleshooting

### Common Issues

1. **No Audio Output**
   - Check audio device connection
   - Verify system audio settings
   - Try specifying output device: `sd.default.device = 'your_device_name'`

2. **Audio Glitches**
   - Increase buffer size: `BUFFER_FRAMES = 2048`
   - Close other audio applications
   - Check CPU usage

3. **Installation Errors**
   - Ensure all system dependencies are installed
   - Try installing PyAudio separately
   - Check Python version compatibility

4. **Streamlit App Issues**
   - Ensure Streamlit is properly installed
   - Check port availability (default: 8501)
   - Try: `streamlit run app.py --server.port 8080`

## Version History

- v1.0.0 - Initial release with basic features
- v1.1.0 - Added Crystal Singing Bowl instrument
- v1.2.0 - Enhanced ambient sounds and volume controls
- v1.3.0 - Added Raspberry Pi optimization
- v1.4.0 - Improved harmonic system and single instrument focus
- v1.5.0 - Added adaptive modulation modes for different meditation styles
- v1.6.0 - Extended harmonic features: richness levels, chord variety, progression styles

## Future Roadmap

- [ ] Additional meditation instruments
- [ ] Binaural beat integration
- [ ] Session recording capability
- [ ] Mobile app development
- [ ] AI-driven composition
- [ ] User preference profiles
