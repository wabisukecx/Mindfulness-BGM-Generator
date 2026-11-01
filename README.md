# Mindfulness BGM Generator

English | [日本語](README.ja.md)

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
├── README.ja.md              # Japanese user guide
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── main.py                   # Command-line version
├── app.py                    # PyQt6 desktop interface
├── main_rpiz2w.py            # Raspberry Pi Zero 2W optimized version
└── src/                      # Source code modules
    ├── __init__.py
    ├── config.py             # High-quality audio settings
    ├── i18n.py               # Japanese GUI translations (English source messages)
    ├── config_rpiz2w.py      # RPi Zero 2W optimized settings
    ├── generator.py          # Main BGM generator class
    ├── generator_rpiz2w.py   # RPi optimized generator
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

- Python 3.10+ (validated on Windows with Python 3.14)
- Audio output device
- 4GB RAM (recommended)
- Dual-core CPU (minimum)

### Python Dependencies

```python
numpy        # Numerical computing
sounddevice  # Audio I/O
PyQt6        # Desktop GUI (for app.py)
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

The `sounddevice` Windows package includes PortAudio. PyAudio and pipwin are
not needed by this application.

### 3. Install Python Dependencies

```bash
python -m pip install -r requirements.txt
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
| `--binaural-enhancement` | Enable subtle stereo detuning | `True` | `--binaural-enhancement` |
| `--no-session-evolution` | Disable session evolution | — | `--no-session-evolution` |
| `--no-binaural-enhancement` | Disable stereo detuning | — | `--no-binaural-enhancement` |

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

### Desktop Interface (app.py / PyQt6)

The GUI is a native desktop window. Audio plays on the selected local output
device. A browser or a web server is not required.

On Windows PowerShell:

```powershell
cd Mindfulness-BGM-Generator
.\.venv\Scripts\python.exe app.py
```

For a fresh checkout, create the environment first:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The interface opens in English on first launch. Use **Language** at the top
of the window to switch between **English** and **日本語**. Your selection is
saved for the next launch. You can switch during playback without changing
the sound settings or restarting the session. Labels, presets, descriptions,
status messages, and application error dialogs follow the selected language;
device names and technical error details are supplied by the audio system.

The desktop opens in **Incense & Stillness** with the **Incense & Stillness** preset:
a steady, quiet low open fifth, no percussion, no ocean noise, and no stereo
detuning. The pitch never changes, but the sound is alive, like incense smoke:

- The low fifth (110/165 Hz) is voiced with soft harmonics that each swell and
  thin on their own slow, unsynchronised 20–90 second cycles, and the fifth is
  detuned by 0.08 Hz so coinciding harmonics beat slowly, like a rin's ring.
  The harmonics also keep it audible on laptop and small speakers that cannot
  reproduce the low fundamental.
- Every few minutes the tone recedes over 40 seconds to near-silence, rests
  there for 20–60 seconds, and returns: a space for the stillness itself.
- A rin (660 Hz) opens the sitting; a timed sitting closes with another as the
  sound fades, and ends once its ring has died away.

Four presets are available:

- **Incense & Stillness** (線香と静坐): a steady low background tone for sitting quietly.
- **Sitting with Resonance** (余韻に座る): the same steady bed with a crystal bowl at 90–150 second intervals.
- **Distant Shore** (遠い水辺): a small amount of softened ocean texture behind the steady tone.
- **Shō Resonance** (笙のひびき): in place of the low tone, a voice inspired by the shō (the
  gagaku mouth organ): sustained reed clusters around 440–1320 Hz built from
  A B D E F# (inspired by, not reproducing, the historical aitake). Each pipe is
  detuned by up to 2 cents for a shimmer, the sound swells with a 12-second
  breath, and every 25–45 seconds the cluster shifts one pipe at a time, like
  te-utsuri. The rin and the spaces of near-silence are the same as above.

Choose 15, 20, 30, or 60 minutes, or an untimed session. Playback enters over
8 seconds; a timed session fades out over its final 20 seconds, then stops
without an alarm. These are app timers, independent of incense burn time.
The settings disappear while playing, leaving a still, uncluttered view.
The volume control applies to the entire mix. Fine adjustments are available
under the optional advanced settings; changing the sound settings selects the
custom profile and restores the usual generative behavior.

The bilingual interface provides:

- Begin and End controls, with cleanup when the window closes
- Meditation presets, a session timer, and a master volume control
- Instrument selection, including Random and No Percussion
- Output device selection and device refresh
- Ocean ambient sound mix (0–100%)
- Harmonic modulation, richness, chord variety, and progression settings
- Session evolution and stereo detuning switches
- Minimum and maximum intervals for each instrument (minimum 0 disables it)

Settings are locked during playback. Stop playback to change them. If a device
fails to open or playback is interrupted, the GUI shows an error and returns
to its stopped state.

### Raspberry Pi Zero 2W Version (main_rpiz2w.py)

Optimized for lower-power devices:

```bash
python main_rpiz2w.py
```

This version features:

- Lower memory footprint
- Optimized audio processing
- Basic instrument and ambient options; enhanced harmonic options are available in main.py only

## Technical Details

### Audio Processing

- Sample Rate: 16kHz (optimized) / 48kHz (standard)
- Buffer Size: 1024 frames (optimized) / 2048 frames (standard)
- Channels: Stereo
- Output Format: Float32
- Output Level: default settings average about -20 dBFS RMS with peaks below
  -6 dBFS at full volume, so the limiter only catches unusual peaks; the quiet
  presets sit a few dB lower

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
   - Check Python version compatibility

4. **Desktop GUI Issues**
   - Install dependencies in the same environment used to run `app.py`.
   - Click the output device refresh button after connecting a device.
   - Choose an output device that supports stereo at 48 kHz.

## Validation

Run the regression checks without opening an audio device:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The checks cover worker cleanup, desktop start/stop, window close, and device failure, zero and
invalid intervals, late sound changes, finite stereo output, real channel
detuning, ocean frequency content, and filter/delay continuity between blocks.
They also check that percussion envelopes do not depend on the audio block size,
that no partial aliases above the Nyquist frequency, that every drone timbre stays
audible, and that a modulation lands on the root its bridge chord prepared.
Percussion re-strikes ring over earlier notes (the oldest voice fades rather than
being cut), and the percussion level control and limiter change gain smoothly
within each block instead of stepping between blocks.
Listening quality and Raspberry Pi hardware performance require device testing.

## Version History

- Unreleased - Make English the default GUI language, add a saved English/Japanese language switch and a Japanese user guide, and expand advanced sound settings to the right.
- v1.8.0 - Add quiet incense/sitting presets, session timing, soft entry and timed exit, master volume, warm colors, and an uncluttered sitting view.
- v1.7.0 - Replace the Streamlit interface with a Japanese PyQt6 desktop GUI, output selection, harmonic controls, and playback interruption handling.
- v1.6.1 - Fix duplicate Web schedulers and stop cleanup, reset sound envelopes on sound changes, validate intervals, add disable flags, generate actual stereo detuning, and replace sub-audible ocean output with colored noise.
- v1.6.0 - Extended harmonic features: richness levels, chord variety, progression styles
- v1.5.0 - Added adaptive modulation modes for different meditation styles
- v1.4.0 - Improved harmonic system and single instrument focus
- v1.3.0 - Added Raspberry Pi optimization
- v1.2.0 - Enhanced ambient sounds and volume controls
- v1.1.0 - Added Crystal Singing Bowl instrument
- v1.0.0 - Initial release with basic features

## Future Roadmap

- [ ] Additional meditation instruments
- [ ] Binaural beat integration
- [ ] Session recording capability
- [ ] Mobile app development
- [ ] AI-driven composition
- [ ] User preference profiles

## License

This project is licensed under the [MIT License](LICENSE).
