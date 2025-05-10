# Mindfulness BGM Generator

Dynamic ambient music generator designed for meditation and mindfulness practices. Creates continuous, evolving soundscapes with harmonious chords, natural ambient sounds, and traditional meditation instruments.

## Overview

This application provides a real-time audio generation system that creates continuously evolving ambient music suitable for meditation and mindfulness practices. It combines synthesized harmonies, natural sounds, and traditional meditation instruments to create a peaceful soundscape.

## Features

### Core Features
- **Dynamic Sound Generation**: Continuously evolving harmonies and textures
- **Multiple Sound Types**:
  - Harmonic-rich tones
  - Pure tones
  - Soft pad sounds
  - Warm tones
  - Bell-like sounds
- **Four Meditation Instruments**:
  - Tibetan bells (Tingsha)
  - Slit drums
  - Handpan sounds
  - Crystal singing bowls
- **Natural Ambient Sounds**: Ocean waves with enhanced volume and presence
- **Breathing Rhythm Synchronization**: Subtle volume modulation aligned with typical breathing patterns
- **Real-time Audio Processing**: Low-latency sound generation
- **Customizable Parameters**: Adjust instrument intervals and ambient mix ratio
- **Smooth Transitions**: Crossfading between chords and sound types
- **Simple Reverb Effect**: Adds spatial depth to the soundscape

### Musical Features
- **Mindfulness-oriented Harmonies**: Uses perfect fifths, fourths, and open intervals
- **Single Instrument per Session**: Each session focuses on one meditation instrument for deeper meditation
- **Harmonic Percussion**: Instruments play notes that harmonize with the current background chord
- **Natural Rhythm Patterns**: Instruments play 1-3 notes with musical spacing

## Project Structure

```
.
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── packages.txt             # System-level dependencies
├── main.py                  # Command-line version
├── app.py                   # Streamlit web interface
└── main_rpiz2w.py          # Raspberry Pi Zero 2 W optimized version
```

## Requirements

### System Requirements
- Python 3.7+
- Audio output device
- 4GB RAM (recommended)
- Dual-core CPU (minimum)

### Python Dependencies
```
numpy          # Numerical computing
sounddevice    # Audio I/O
streamlit      # Web application framework (for app.py)
pyaudio        # Audio I/O (optional, for some systems)
```

### System Dependencies (Linux/Debian-based)
```
portaudio19-dev    # PortAudio library
python3-all-dev    # Python development headers
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

### Raspberry Pi Zero 2 W Version (main_rpiz2w.py)

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
- **Sample Rate**: 16kHz (optimized) / 44.1kHz (standard)
- **Buffer Size**: 1024 frames
- **Channels**: Stereo
- **Output Format**: Float32

### Musical Elements
- **Scale**: A Pentatonic (A, C, E, A, C)
- **Chord Types** (Mindfulness-oriented):
  - Single notes (drone)
  - Octaves
  - Perfect fifths
  - Perfect fourths
  - Quartal harmonies
  - Sus2 chords
  - Open voicings

### Sound Generation
- Pure sine wave synthesis
- Harmonic addition for timbral richness
- Exponential envelope generators
- Phase modulation for stereo width
- Soft limiting to prevent clipping

## Architecture

The application uses a multi-threaded architecture:

```
Main Thread (Audio Callback)
├── Sound Generation
├── Envelope Processing
└── Effect Processing (Reverb)

Event Scheduler Thread
├── Chord Changes
├── Sound Type Changes
└── Volume Transitions

Instrument Scheduler Threads
├── Bell Scheduler
├── Drum Scheduler  
├── Handpan Scheduler
└── Crystal Bowl Scheduler
```

### Core Classes

| Class | Description |
|-------|-------------|
| `MindfulnessBGM` | Main generator class |
| `TibetanBell` | Tibetan bell sound synthesis |
| `SlitDrum` | Slit drum sound synthesis |
| `Handpan` | Handpan sound synthesis |
| `CrystalSingingBowl` | Crystal bowl sound synthesis |
| `SoundType` | Enum for different sound types |
| `InstrumentConfig` | Configuration for each instrument |

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

### Platform-Specific Notes

#### Raspberry Pi
- Use `main_rpiz2w.py` for better performance
- Consider using a USB audio adapter for better quality
- Monitor CPU temperature during extended use

#### Windows
- May require Microsoft Visual C++ Redistributable
- Use Windows audio troubleshooter for device issues

#### macOS
- Grant microphone permissions if prompted
- Use `brew doctor` to check Homebrew installation

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Follow existing code style
4. Add appropriate documentation
5. Test on multiple platforms if possible
6. Submit a pull request with clear description

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Comment complex algorithms
- Keep functions focused and small

## License

[MIT License](LICENSE)

## Acknowledgments

- Inspired by traditional meditation music and modern sound therapy techniques
- Special thanks to the Python audio processing community
- Sound synthesis techniques based on academic research in psychoacoustics

## Version History

- v1.0.0 - Initial release with basic features
- v1.1.0 - Added Crystal Singing Bowl instrument
- v1.2.0 - Enhanced ambient sounds and volume controls
- v1.3.0 - Added Raspberry Pi optimization
- v1.4.0 - Improved harmonic system and single instrument focus

## Future Roadmap

- [ ] Additional meditation instruments
- [ ] Binaural beat integration
- [ ] Session recording capability
- [ ] Mobile app development
- [ ] AI-driven composition
- [ ] User preference profiles
