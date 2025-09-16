"""
Multilingual Translation System - Streamlit App
Main application interface with text and speech I/O
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import tempfile
import os
import base64
from io import BytesIO
import time
import shutil  # for ffmpeg check

# Import custom modules
from models.translator import TranslationPipeline
from models.speech_to_text import SpeechToText, AudioRecorder
from models.text_to_speech import TextToSpeech
from models.transliterator import Transliterator
from utils.language_config import SUPPORTED_LANGUAGES, get_language_pair
from utils.cache_manager import ModelCache

# Page configuration
st.set_page_config(
    page_title="🌐 Multilingual Translator",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .translation-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .history-item {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 3px solid #4CAF50;
    }
    h1 {
        color: #2E7D32;
    }
    .stSelectbox > label {
        font-weight: bold;
        color: #2E7D32;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'translation_history' not in st.session_state:
    st.session_state.translation_history = []
if 'audio_recorder' not in st.session_state:
    st.session_state.audio_recorder = None
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'transcribed_text' not in st.session_state:
    st.session_state.transcribed_text = ""
if 'translator' not in st.session_state:
    st.session_state.translator = None
if 'stt' not in st.session_state:
    st.session_state.stt = None
if 'tts' not in st.session_state:
    st.session_state.tts = None
if 'transliterator' not in st.session_state:
    st.session_state.transliterator = None

# FFmpeg availability and diagnostics
if 'ffmpeg_available' not in st.session_state:
    st.session_state['ffmpeg_available'] = bool(shutil.which("ffmpeg"))
if 'ffmpeg_path' not in st.session_state:
    st.session_state['ffmpeg_path'] = shutil.which("ffmpeg")

# Ensure audio persistence state variables
if 'audio_b64' not in st.session_state:
    st.session_state['audio_b64'] = None
if 'tts_status' not in st.session_state:
    st.session_state['tts_status'] = ""

# Cache models
@st.cache_resource(show_spinner=False)
def load_models():
    """Load and cache all models."""
    with st.spinner("🔄 Loading models... This may take a minute on first run."):
        translator = TranslationPipeline(model_type='nllb-200')
        stt = SpeechToText(model_size='base')
        tts = TextToSpeech(backend='auto')
        transliterator = Transliterator()
    
    return translator, stt, tts, transliterator

# Load models once at startup
if not st.session_state.models_loaded:
    translator, stt, tts, transliterator = load_models()
    st.session_state.translator = translator
    st.session_state.stt = stt
    st.session_state.tts = tts
    st.session_state.transliterator = transliterator
    st.session_state.models_loaded = True

# Header
st.title("🌐 Multilingual Translation System")
st.markdown("**Translate text and speech across multiple languages with optional transliteration**")

# Show FFmpeg diagnostic in sidebar
# Only warn if FFmpeg is missing
if not st.session_state['ffmpeg_available']:
    st.sidebar.warning(
        "⚠️ FFmpeg not found on system PATH. Speech I/O (Whisper/generating audio) may not work."
    )


# Sidebar (settings + history)
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model settings
    st.subheader("Model Configuration")
    model_quality = st.select_slider(
        "Translation Quality",
        options=["Fast", "Balanced", "High Quality"],
        value="Balanced"
    )
    
    # Set beam size based on quality
    beam_size = {"Fast": 2, "Balanced": 4, "High Quality": 8}[model_quality]
    
    # Transliteration settings
    st.subheader("Transliteration")
    enable_transliteration = st.checkbox("Enable Transliteration", value=False)
    
    if enable_transliteration:
        romanization_scheme = st.selectbox(
            "Romanization Scheme",
            ["itrans", "iast", "hk", "wx"],
            help="Choose the romanization scheme for Indic scripts"
        )
    else:
        romanization_scheme = "itrans"
    
    # History section
    st.header("📜 Translation History")
    
    if st.session_state.translation_history:
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.translation_history = []
            st.rerun()
        
        # Display history
        for idx, item in enumerate(reversed(st.session_state.translation_history[-10:])):
            with st.expander(f"Translation {len(st.session_state.translation_history) - idx}"):
                st.text(f"Time: {item['timestamp']}")
                st.text(f"From: {item['source_lang']}")
                st.text(f"To: {item['target_lang']}")
                st.text(f"Input: {item['source_text'][:100]}...")
                st.text(f"Output: {item['translation'][:100]}...")
    else:
        st.info("No translations yet")
    
    # About section
    st.header("ℹ️ About")
    st.markdown("""
    **Features:**
    - 🌍 18+ Language Support
    - 🎤 Speech Input
    - 🔊 Audio Output  
    - 📝 Text Translation
    - 🔤 Transliteration
    - 📜 History Tracking
    """)

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Input")
    
    # Language selection
    source_lang = st.selectbox(
        "Source Language",
        list(SUPPORTED_LANGUAGES.keys()),
        index=0,
        key="source_lang_select"
    )
    
    # Input method tabs
    input_tab1, input_tab2 = st.tabs(["📝 Text Input", "🎤 Speech Input"])
    
    with input_tab1:
        # Text input
        input_text = st.text_area(
            "Enter text to translate",
            height=200,
            placeholder="Type or paste your text here...",
            key="text_input"
        )
        
        # Optional: File upload
        uploaded_file = st.file_uploader(
            "Or upload a text file",
            type=['txt'],
            key="file_upload"
        )
        
        if uploaded_file:
            input_text = uploaded_file.read().decode('utf-8')
            st.text_area("File content:", value=input_text, height=100, disabled=True)
    
    with input_tab2:
        # Speech input
        st.markdown("**Record your speech:**")
        
        col_rec1, col_rec2 = st.columns(2)
        
        with col_rec1:
            # Disable Start Recording if already recording
            if st.button("🎤 Start Recording", disabled=st.session_state.is_recording):
                st.session_state.is_recording = True
                st.session_state.audio_recorder = AudioRecorder()
                st.session_state.audio_recorder.start_recording()
                st.rerun()
        
        with col_rec2:
            if st.button("⏹️ Stop Recording", disabled=not st.session_state.is_recording):
                if st.session_state.audio_recorder:
                    audio_data = st.session_state.audio_recorder.stop_recording()
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                        st.session_state.audio_recorder.save_recording(audio_data, tmp.name)
                        temp_audio_path = tmp.name
                    
                    # Transcribe using the already loaded STT model
                    with st.spinner("Transcribing..."):
                        result = st.session_state.stt.transcribe_audio_file(
                            temp_audio_path,
                            language=SUPPORTED_LANGUAGES[source_lang]
                        )
                        st.session_state.transcribed_text = result.get('text', '')
                    
                    # Clean up
                    try:
                        os.unlink(temp_audio_path)
                    except Exception:
                        pass
                    
                st.session_state.is_recording = False
                st.rerun()
        
        if st.session_state.is_recording:
            st.warning("🔴 Recording in progress...")
        
        # Display transcribed text if available
        if st.session_state.transcribed_text:
            input_text = st.text_area(
                "Transcribed text:", 
                value=st.session_state.transcribed_text, 
                height=100,
                key="transcribed_display"
            )
    
    # Transliteration option for input
    if enable_transliteration and source_lang in ["Hindi", "Bengali", "Tamil", "Telugu", "Marathi", "Gujarati"]:
        if st.checkbox("Show romanized version of input"):
            if input_text:
                romanized = st.session_state.transliterator.to_roman(
                    input_text,
                    SUPPORTED_LANGUAGES[source_lang],
                    romanization_scheme
                )
                st.text_area("Romanized input:", value=romanized, height=100, disabled=True)

with col2:
    st.subheader("📤 Output")
    
    # Target language selection
    target_lang = st.selectbox(
        "Target Language",
        list(SUPPORTED_LANGUAGES.keys()),
        index=1 if len(SUPPORTED_LANGUAGES) > 1 else 0,
        key="target_lang_select"
    )
    
    # Translate button
    if st.button("🔄 Translate", type="primary", use_container_width=True):
        if input_text:
            # Perform translation
            with st.spinner("Translating..."):
                start_time = time.time()
                
                translation, success = st.session_state.translator.translate_with_fallback(
                    input_text,
                    SUPPORTED_LANGUAGES[source_lang],
                    SUPPORTED_LANGUAGES[target_lang],
                    num_beams=beam_size
                )
                
                translation_time = time.time() - start_time
            
            if success:
                # Display translation
                st.success(f"✅ Translation completed in {translation_time:.2f} seconds")
                
                # Translation output
                st.text_area(
                    "Translation:",
                    value=translation,
                    height=200,
                    key="translation_output"
                )
                
                # Transliteration option for output
                if enable_transliteration and target_lang in ["Hindi", "Bengali", "Tamil", "Telugu", "Marathi", "Gujarati"]:
                    romanized_output = st.session_state.transliterator.to_roman(
                        translation,
                        SUPPORTED_LANGUAGES[target_lang],
                        romanization_scheme
                    )
                    st.text_area(
                        "Romanized translation:",
                        value=romanized_output,
                        height=100,
                        key="romanized_output"
                    )
                
                # Text-to-Speech
                st.subheader("🔊 Audio Output")
                
                col_audio1, col_audio2 = st.columns(2)
                
                # ---- Persistent TTS generation and player ----
                with col_audio1:
                    # Disable generate audio button if ffmpeg not available
                    gen_disabled = not st.session_state['ffmpeg_available']
                    
                    if st.button("🔊 Generate Audio", disabled=gen_disabled):
                        # Reset status
                        st.session_state['tts_status'] = "Generating..."
                        st.session_state['audio_b64'] = None

                        try:
                            with st.spinner("Generating audio..."):
                                audio_result = st.session_state.tts.synthesize(
                                    translation,
                                    SUPPORTED_LANGUAGES[target_lang]
                                )

                            # If audio_result is a dict with success flag
                            if not audio_result or not (isinstance(audio_result, dict) and audio_result.get('success', False)):
                                # Log and show error
                                if isinstance(audio_result, dict):
                                    err_msg = audio_result.get('error', 'Unknown error')
                                else:
                                    err_msg = 'TTS returned invalid response'
                                st.session_state['tts_status'] = f"Failed: {err_msg}"
                                st.error("Failed to generate audio: " + str(err_msg))
                            else:
                                # Prefer in-memory audio_data, fallback to saved file
                                audio_b64_local = None
                                if 'audio_data' in audio_result and audio_result['audio_data']:
                                    b = audio_result['audio_data']
                                    if isinstance(b, (bytes, bytearray)):
                                        audio_b64_local = base64.b64encode(b).decode()
                                    else:
                                        # if it's a path string
                                        try:
                                            with open(b, 'rb') as f:
                                                audio_b64_local = base64.b64encode(f.read()).decode()
                                        except Exception as e:
                                            st.session_state['tts_status'] = f"Failed to read audio_data: {e}"
                                            st.error("Failed to read generated audio: " + str(e))
                                elif 'save_path' in audio_result and audio_result['save_path']:
                                    try:
                                        with open(audio_result['save_path'], 'rb') as f:
                                            audio_b64_local = base64.b64encode(f.read()).decode()
                                    except Exception as e:
                                        st.session_state['tts_status'] = f"Failed to read save_path: {e}"
                                        st.error("Failed to read audio file: " + str(e))
                                else:
                                    st.session_state['tts_status'] = "No audio returned"
                                    st.error("TTS returned no audio data")

                                if audio_b64_local:
                                    st.session_state['audio_b64'] = audio_b64_local
                                    st.session_state['tts_status'] = "Ready"
                                    st.success("Audio generated successfully")

                        except Exception as e:
                            st.session_state['tts_status'] = f"Exception: {e}"
                            st.error("Exception during TTS: " + str(e))

                    # Show current TTS status
                    if st.session_state.get('tts_status'):
                        st.caption(st.session_state['tts_status'])

                    # Persistent audio player (display if audio_b64 exists)
                    if st.session_state.get('audio_b64'):
                        audio_html = f"""
                        <audio controls>
                            <source src="data:audio/mp3;base64,{st.session_state['audio_b64']}" type="audio/mp3">
                            Your browser does not support the audio element.
                        </audio>
                        """
                        st.markdown(audio_html, unsafe_allow_html=True)
                # -------------------------------------------------
                
                with col_audio2:
                    # Download button for translation
                    st.download_button(
                        label="📥 Download Translation",
                        data=translation,
                        file_name=f"translation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                
                # Add to history
                st.session_state.translation_history.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'source_lang': source_lang,
                    'target_lang': target_lang,
                    'source_text': input_text,
                    'translation': translation
                })
                
            else:
                st.error("❌ Translation failed. Please try again.")
        else:
            st.warning("⚠️ Please enter text to translate")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit, Hugging Face Transformers, and OpenAI Whisper</p>
        <p>© 2025 Multilingual Translation System</p>
    </div>
    """,
    unsafe_allow_html=True
)
