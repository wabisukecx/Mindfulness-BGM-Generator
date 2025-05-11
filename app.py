#!/usr/bin/env python3
"""
Streamlit web interface for Mindfulness BGM Generator
"""

import streamlit as st
import sounddevice as sd
import random

from src.config import SAMPLE_RATE, BUFFER_FRAMES, DEFAULT_AMBIENT_RATIO
from src.instruments_base import InstrumentConfig
from src.generator_streamlit import MindfulnessBGMStreamlit


# Initialize session state
if 'generator' not in st.session_state:
    st.session_state.generator = None
if 'stream' not in st.session_state:
    st.session_state.stream = None
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False
if 'selected_instrument' not in st.session_state:
    st.session_state.selected_instrument = None
if 'current_status' not in st.session_state:
    st.session_state.current_status = "Stopped"
if 'audio_available' not in st.session_state:
    st.session_state.audio_available = None


def check_audio_device():
    """Check if audio device is available"""
    try:
        devices = sd.query_devices()
        return len(devices) > 0
    except Exception:
        return False


def start_bgm():
    """Start the BGM generator"""
    if st.session_state.is_playing:
        return
    
    # Get settings from session state
    selected_instrument = st.session_state.get('instrument_selection', 'Random')
    bell_interval = st.session_state.get('bell_interval', (15, 45))
    drum_interval = st.session_state.get('drum_interval', (8, 25))
    handpan_interval = st.session_state.get('handpan_interval', (12, 30))
    crystal_bowl_interval = st.session_state.get('crystal_bowl_interval', (25, 60))
    ambient_ratio = st.session_state.get('ambient_ratio', 0.3)
    
    # Select instrument
    instruments = ['bell', 'drum', 'handpan', 'crystal-bowl']
    if selected_instrument == 'Random':
        selected = random.choice(instruments)
    else:
        selected = selected_instrument.lower().replace(' ', '-')
    
    # Enable only the selected instrument
    bell_enabled = (selected == 'bell')
    drum_enabled = (selected == 'drum')
    handpan_enabled = (selected == 'handpan')
    crystal_bowl_enabled = (selected == 'crystal-bowl')
    
    # Create instrument configurations
    bell_config = InstrumentConfig(bell_interval[0], bell_interval[1], bell_enabled)
    drum_config = InstrumentConfig(drum_interval[0], drum_interval[1], drum_enabled)
    handpan_config = InstrumentConfig(handpan_interval[0], handpan_interval[1], handpan_enabled)
    crystal_bowl_config = InstrumentConfig(crystal_bowl_interval[0], crystal_bowl_interval[1], crystal_bowl_enabled)
    
    # Create generator
    generator = MindfulnessBGMStreamlit(bell_config, drum_config, handpan_config, crystal_bowl_config, ambient_ratio)
    
    # Configure audio
    sd.default.samplerate = SAMPLE_RATE
    sd.default.blocksize = BUFFER_FRAMES
    sd.default.channels = 2
    
    try:
        # Start audio stream
        stream = sd.OutputStream(callback=generator.callback)
        stream.start()
        
        # Update session state
        st.session_state.generator = generator
        st.session_state.stream = stream
        st.session_state.is_playing = True
        st.session_state.selected_instrument = selected.replace('-', ' ').title()
        st.session_state.current_status = f"Playing with {st.session_state.selected_instrument}"
        
    except sd.PortAudioError as e:
        generator.stop()
        st.session_state.current_status = "Audio device not available"
        st.error("❌ Audio device is not available in this environment.")
        st.warning("Streamlit Cloud does not support audio device access.")
        
        with st.expander("📋 Solution: Run the application locally", expanded=True):
            st.markdown("""
                To use this application with full audio functionality, please run it on your local machine:
                
                1. **Download the required files**:
                    - `app.py` (this application)
                    - `requirements.txt`
                    - `packages.txt` (for Linux/Mac)
                    
                2. **Install dependencies**:
                    ```
                    sudo apt-get install portaudio19-dev python3-all-dev
                    pip install -r requirements.txt
                    ```
                    
                3. **Run the application**:
                    ```
                    streamlit run app.py
                    ```
                    
                For Windows users, you may need to install PyAudio separately:
                    ```
                    pip install pipwin
                    pipwin install pyaudio
                    ```
            """)


def stop_bgm():
    """Stop the BGM generator"""
    if not st.session_state.is_playing:
        return
    
    # Stop audio stream
    if st.session_state.stream:
        st.session_state.stream.stop()
        st.session_state.stream.close()
    
    # Stop generator
    if st.session_state.generator:
        st.session_state.generator.stop()
    
    # Reset session state
    st.session_state.generator = None
    st.session_state.stream = None
    st.session_state.is_playing = False
    st.session_state.current_status = "Stopped"
    st.session_state.selected_instrument = None


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Mindfulness BGM Generator",
        page_icon="🎵",
        layout="wide"
    )
    
    # Check audio availability
    if st.session_state.audio_available is None:
        st.session_state.audio_available = check_audio_device()
    
    # Title and description
    st.title("🧘 Mindfulness BGM Generator")
    st.markdown("Dynamic ambient music generator for meditation and mindfulness practices")
    
    # Audio availability warning
    if not st.session_state.audio_available:
        st.warning("⚠️ Audio device is not available. The application will work but cannot produce sound.")
        st.info("Tip: Run this application locally for full audio functionality.")
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("▶️ Start", use_container_width=True, disabled=st.session_state.is_playing):
            start_bgm()
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop", use_container_width=True, disabled=not st.session_state.is_playing):
            stop_bgm()
            st.rerun()
    
    with col3:
        st.write(f"**Status:** {st.session_state.current_status}")
    
    # Settings
    with st.container():
        st.markdown("---")
        st.subheader("🎛️ Settings")
        
        settings_col1, settings_col2 = st.columns(2)
        
        with settings_col1:
            instrument_options = ['Random', 'Bell', 'Drum', 'Handpan', 'Crystal Bowl']
            selected_instrument = st.selectbox(
                "Select Instrument",
                instrument_options,
                key='instrument_selection',
                disabled=st.session_state.is_playing
            )
        
        with settings_col2:
            ambient_ratio = st.slider(
                "Ambient Sound Mix",
                0.0, 1.0, DEFAULT_AMBIENT_RATIO,
                key='ambient_ratio',
                disabled=st.session_state.is_playing
            )
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings - Instrument Intervals", expanded=False):
            adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
            
            with adv_col1:
                st.markdown("**🔔 Tibetan Bell**")
                bell_interval = st.slider(
                    "Interval (seconds)",
                    0, 60, (15, 45),
                    key='bell_interval',
                    disabled=st.session_state.is_playing
                )
            
            with adv_col2:
                st.markdown("**🥁 Slit Drum**")
                drum_interval = st.slider(
                    "Interval (seconds)",
                    0, 60, (8, 25),
                    key='drum_interval',
                    disabled=st.session_state.is_playing
                )
            
            with adv_col3:
                st.markdown("**🎵 Handpan**")
                handpan_interval = st.slider(
                    "Interval (seconds)",
                    0, 60, (12, 30),
                    key='handpan_interval',
                    disabled=st.session_state.is_playing
                )
            
            with adv_col4:
                st.markdown("**🔮 Crystal Bowl**")
                crystal_bowl_interval = st.slider(
                    "Interval (seconds)",
                    0, 60, (25, 60),
                    key='crystal_bowl_interval',
                    disabled=st.session_state.is_playing
                )
    
    # Information tabs
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["ℹ️ About", "📖 Instructions", "🎯 Features"])
    
    with tab1:
        st.markdown("""
            This app generates dynamic ambient music designed specifically for meditation and mindfulness practices.
            It combines traditional meditation instruments with natural ambient sounds and harmonious chord progressions.
            
            The music evolves continuously, creating a unique experience each time you use it.
            
            The application uses advanced audio synthesis techniques to create realistic instrument sounds
            and smooth transitions between different musical elements.
        """)
    
    with tab2:
        st.markdown("""
            ### How to Use
            
            1. **Select an instrument** or choose 'Random' to let the app choose for you
            2. **Adjust the ambient sound mix** to set the balance between instruments and nature sounds
            3. **Click Start** to begin generating music
            4. **Click Stop** when you want to finish
            
            ### Advanced Options
            
            - Use the expander to access detailed interval settings for each instrument
            - Each instrument plays at random intervals within the specified range
            - Set an interval to 0 to disable that instrument completely
        """)
    
    with tab3:
        st.markdown("""
            ### Key Features
            
            - **Dynamic Sound Generation**: Continuously evolving soundscapes
            - **Multiple Sound Types**: Different timbres and textures
            - **Four Meditation Instruments**:
              - Tibetan Bells (Tingsha)
              - Slit Drums
              - Handpan
              - Crystal Singing Bowls
            - **Natural Ambient Sounds**: Ocean waves with realistic textures
            - **Breathing Rhythm Synchronization**: Subtle volume modulations
            - **Real-time Audio Processing**: Low-latency sound generation
            - **Customizable Parameters**: Fine-tune your experience
            - **Smooth Transitions**: Crossfading between musical elements
            - **Spatial Effects**: Reverb and stereo enhancement
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666;">
            Mindfulness BGM Generator v1.4.0
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
