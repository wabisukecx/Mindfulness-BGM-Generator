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
- **Meditation Instruments** (one instrument selected per session):
  - Tibetan bells (Tingsha)
  - Slit drums
  - Handpan sounds
- **Natural Ambient Sounds**: Ocean waves and wind with enhanced volume
- **Breathing Rhythm Synchronization**: Subtle volume modulation synced to typical breathing patterns
- **Real-time Audio Processing**: Low-latency sound generation
- **Customizable Parameters**: Adjust instrument intervals and ambient mix ratio
- **Smooth Transitions**: Crossfading between chords and sound types
- **Simple Reverb Effect**: Adds spatial depth to the soundscape
- **Mindfulness-oriented Harmonies**: Uses perfect fifths, fourths, and open intervals

## Requirements

- Python 3.7+
- NumPy
- SoundDevice
- pyaudio (optional, for some systems)

For the Streamlit web app:

- Streamlit

## Installation

1. Clone this repository:

```bash
git clone https://github.com/wabisukecx/Mindfulness-BGM-Generator.git
cd Mindfulness-BGM-Generator
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Command-line Version (main.py)

Run with default settings (randomly selects one instrument):

```bash
python main.py
```

### Web App Version (app.py)

Run the Streamlit application:

```bash
streamlit run app.py
```

The web app provides a graphical interface with:

- Start/Stop controls
- Instrument selection
- Ambient sound mix adjustment
- Interval settings for each instrument

### Command-line Options

- `--instrument`: Choose specific instrument ('bell', 'drum', or 'handpan')
- `--bell`: Tibetan bell interval in seconds (e.g., '15-45' or '30' or '0' to disable)
- `--drum`: Slit drum interval in seconds (e.g., '8-25' or '15' or '0' to disable)  
- `--handpan`: Handpan interval in seconds (e.g., '12-30' or '20' or '0' to disable)
- `--ambient`: Ambient sound ratio (0-1, default: 0.3, '0' to disable)

### Examples

Use only Tibetan bells:

```bash
python main.py --instrument bell
```

Use only slit drum:

```bash
python main.py --instrument drum
```

Use only handpan:

```bash
python main.py --instrument handpan
```

Set custom intervals for the selected instrument:

```bash
python main.py --instrument drum --drum 10-20
```

Adjust ambient sound mix:

```bash
python main.py --ambient 0.5
```

Ambient sounds only (no percussion):

```bash
python main.py --bell 0 --drum 0 --handpan 0
```

## New Features

### Single Instrument per Session

Each time you run the program, it automatically selects one of the three meditation instruments (Tibetan bell, slit drum, or handpan) to use throughout the session. This creates a more focused meditative experience.

### Enhanced Musical Harmony

- **Mindfulness-oriented chord progressions**: Uses open intervals, perfect fifths, fourths, and single notes instead of traditional Western harmonies
- **Harmonic percussion**: Instruments play notes that harmonize with the current background chord
- **Natural rhythm patterns**: Instruments play 1-3 notes with musical spacing (quarter note to whole note intervals)

### Improved Ambient Sounds

- Enhanced volume and presence of natural sounds
- Richer ocean wave patterns with multiple frequencies
- More dynamic wind sounds
- Added low-frequency rumble for depth

## Technical Details

### Audio Processing

- **Sample Rate**: 44100 Hz
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

- Main thread: Audio callback and signal processing
- Event scheduler thread: Manages chord and sound type changes
- Single instrument scheduler thread: Controls the selected percussion timing

## Customization

You can modify the following constants in the code:

- `VOLUME`: Overall output volume
- `MIN_EVENT_INTERVAL`: Minimum time between sound changes
- `MAX_EVENT_INTERVAL`: Maximum time between sound changes
- `FADE_TIME`: Crossfade duration
- `BREATH_CYCLE`: Breathing rhythm frequency
- `BASE_FREQS`: Musical scale frequencies
- `DEFAULT_AMBIENT_RATIO`: Default ambient sound mix level (0.3)

## Known Issues

- On some systems, you may need to specify the audio device explicitly
- Very low buffer sizes may cause audio glitches on slower systems

## Contributing

Contributions are welcome! Please feel free to submit pull requests or create issues for bugs and feature requests.

## License

[MIT License](LICENSE)

## Acknowledgments

Inspired by traditional meditation music and modern sound therapy techniques. Special thanks to the Python audio processing community.
