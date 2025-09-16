"""
Sidebar Components Module
UI components for the Streamlit sidebar
"""

import streamlit as st
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd

def render_sidebar():
    """
    Render the complete sidebar with all settings and options.
    
    Returns:
        Dict containing all sidebar settings
    """
    settings = {}
    
    with st.sidebar:
        # App header
        st.markdown("""
        <div style='text-align: center'>
            <h1>🌐</h1>
            <h2>Translation Settings</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model settings section
        settings['model'] = render_model_settings()
        
        # Translation settings
        settings['translation'] = render_translation_settings()
        
        # Transliteration settings
        settings['transliteration'] = render_transliteration_settings()
        
        # Audio settings
        settings['audio'] = render_audio_settings()
        
        # History section
        render_history()
        
        # About section
        render_about()
        
        # Footer
        render_footer()
    
    return settings

def render_model_settings() -> Dict:
    """
    Render model configuration settings.
    
    Returns:
        Dictionary with model settings
    """
    st.subheader("⚙️ Model Configuration")
    
    model_settings = {}
    
    # Model selection
    model_settings['model_type'] = st.selectbox(
        "Translation Model",
        ["NLLB-200 (Recommended)", "mBART-50", "mT5-Base"],
        index=0,
        help="Choose the translation model. NLLB-200 supports 200+ languages."
    )
    
    # Map display names to model codes
    model_map = {
        "NLLB-200 (Recommended)": "nllb-200",
        "mBART-50": "mbart-50",
        "mT5-Base": "mt5-base"
    }
    model_settings['model_code'] = model_map[model_settings['model_type']]
    
    # Quality settings
    model_settings['quality'] = st.select_slider(
        "Translation Quality",
        options=["Fast", "Balanced", "High Quality"],
        value="Balanced",
        help="Higher quality uses more computational resources"
    )
    
    # Beam size based on quality
    quality_beam_map = {"Fast": 2, "Balanced": 4, "High Quality": 8}
    model_settings['beam_size'] = quality_beam_map[model_settings['quality']]
    
    # Advanced settings (collapsed by default)
    with st.expander("Advanced Settings"):
        model_settings['temperature'] = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Controls randomness in translation"
        )
        
        model_settings['max_length'] = st.number_input(
            "Max Output Length",
            min_value=50,
            max_value=1000,
            value=512,
            step=50,
            help="Maximum length of translated text"
        )
        
        model_settings['use_gpu'] = st.checkbox(
            "Use GPU Acceleration",
            value=True,
            help="Enable GPU if available for faster processing"
        )
    
    return model_settings

def render_translation_settings() -> Dict:
    """
    Render translation-specific settings.
    
    Returns:
        Dictionary with translation settings
    """
    st.subheader("🔄 Translation Options")
    
    trans_settings = {}
    
    # Batch translation
    trans_settings['batch_mode'] = st.checkbox(
        "Enable Batch Mode",
        value=False,
        help="Process multiple sentences at once"
    )
    
    if trans_settings['batch_mode']:
        trans_settings['batch_size'] = st.slider(
            "Batch Size",
            min_value=2,
            max_value=10,
            value=5,
            help="Number of sentences to process together"
        )
    
    # Preserve formatting
    trans_settings['preserve_formatting'] = st.checkbox(
        "Preserve Formatting",
        value=True,
        help="Maintain original text formatting"
    )
    
    # Auto-detect language
    trans_settings['auto_detect'] = st.checkbox(
        "Auto-detect Source Language",
        value=False,
        help="Automatically detect the source language"
    )
    
    return trans_settings

def render_transliteration_settings() -> Dict:
    """
    Render transliteration settings.
    
    Returns:
        Dictionary with transliteration settings
    """
    st.subheader("🔤 Transliteration")
    
    translit_settings = {}
    
    # Enable transliteration
    translit_settings['enabled'] = st.checkbox(
        "Enable Transliteration",
        value=False,
        help="Convert between native scripts and romanized text"
    )
    
    if translit_settings['enabled']:
        # Romanization scheme
        translit_settings['scheme'] = st.selectbox(
            "Romanization Scheme",
            ["ITRANS", "IAST", "Harvard-Kyoto", "WX"],
            index=0,
            help="Choose the romanization standard"
        )
        
        # Direction
        translit_settings['direction'] = st.radio(
            "Direction",
            ["Native to Roman", "Roman to Native", "Auto"],
            index=2,
            help="Conversion direction"
        )
        
        # Languages to apply
        translit_settings['languages'] = st.multiselect(
            "Apply to Languages",
            ["Hindi", "Bengali", "Tamil", "Telugu", "Marathi", "Gujarati"],
            default=["Hindi"],
            help="Select languages for transliteration"
        )
    
    return translit_settings

def render_audio_settings() -> Dict:
    """
    Render audio settings for STT and TTS.
    
    Returns:
        Dictionary with audio settings
    """
    st.subheader("🔊 Audio Settings")
    
    audio_settings = {}
    
    # STT settings
    st.markdown("**Speech Recognition**")
    audio_settings['stt_model'] = st.selectbox(
        "STT Model",
        ["Whisper Base", "Whisper Small", "Whisper Medium"],
        index=0,
        help="Larger models are more accurate but slower"
    )
    
    audio_settings['noise_reduction'] = st.checkbox(
        "Enable Noise Reduction",
        value=True,
        help="Apply noise reduction to recorded audio"
    )
    
    # TTS settings
    st.markdown("**Text-to-Speech**")
    audio_settings['tts_backend'] = st.selectbox(
        "TTS Engine",
        ["Google TTS (Online)", "pyttsx3 (Offline)", "Auto"],
        index=2,
        help="Choose TTS engine"
    )
    
    audio_settings['speech_rate'] = st.slider(
        "Speech Rate",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Speed of synthesized speech"
    )
    
    audio_settings['voice_gender'] = st.radio(
        "Voice Gender",
        ["Neutral", "Male", "Female"],
        index=0,
        help="Preferred voice gender (if available)"
    )
    
    return audio_settings

def render_history():
    """Render translation history section."""
    st.markdown("---")
    st.subheader("📜 Translation History")
    
    if 'translation_history' not in st.session_state:
        st.session_state.translation_history = []
    
    if st.session_state.translation_history:
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.translation_history = []
                st.rerun()
        
        with col2:
            if st.button("📥 Export", use_container_width=True):
                export_history()
        
        # Filter options
        filter_lang = st.selectbox(
            "Filter by Language",
            ["All"] + list(set([
                item['source_lang'] for item in st.session_state.translation_history
            ])),
            key="history_filter"
        )
        
        # Display history items
        filtered_history = st.session_state.translation_history
        if filter_lang != "All":
            filtered_history = [
                item for item in filtered_history 
                if item['source_lang'] == filter_lang
            ]
        
        # Show recent items (max 10)
        for idx, item in enumerate(reversed(filtered_history[-10:]), 1):
            render_history_item(item, len(filtered_history) - idx + 1)
    else:
        st.info("No translations yet. Start translating to see history!")

def render_history_item(item: Dict, index: int):
    """
    Render a single history item.
    
    Args:
        item: History item dictionary
        index: Item index
    """
    with st.expander(f"#{index} - {item['timestamp']}", expanded=False):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"**From:** {item['source_lang']}")
            st.text_area(
                "Source",
                value=item['source_text'][:200] + ("..." if len(item['source_text']) > 200 else ""),
                height=80,
                disabled=True,
                key=f"hist_src_{index}"
            )
        
        with col2:
            st.markdown(f"**To:** {item['target_lang']}")
            st.text_area(
                "Translation",
                value=item['translation'][:200] + ("..." if len(item['translation']) > 200 else ""),
                height=80,
                disabled=True,
                key=f"hist_tgt_{index}"
            )
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📋 Copy", key=f"copy_{index}", use_container_width=True):
                st.write(f"```{item['translation']}```")
        
        with col2:
            if st.button("🔄 Reuse", key=f"reuse_{index}", use_container_width=True):
                st.session_state.reuse_text = item['source_text']
                st.rerun()
        
        with col3:
            if st.button("🗑️", key=f"delete_{index}", use_container_width=True):
                st.session_state.translation_history.remove(item)
                st.rerun()

def export_history():
    """Export translation history to CSV."""
    if st.session_state.translation_history:
        df = pd.DataFrame(st.session_state.translation_history)
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download History (CSV)",
            data=csv,
            file_name=f"translation_history_{datetime.now():%Y%m%d_%H%M%S}.csv",
            mime="text/csv"
        )

def render_about():
    """Render about section."""
    st.markdown("---")
    st.subheader("ℹ️ About")
    
    with st.expander("Features & Capabilities"):
        st.markdown("""
        **Core Features:**
        - 🌍 **18+ Languages** supported
        - 🎤 **Speech Input** via microphone
        - 🔊 **Audio Output** with TTS
        - 📝 **Text Translation** with high accuracy
        - 🔤 **Transliteration** for Indic scripts
        - 📜 **History Tracking** with export
        - ⚡ **GPU Acceleration** support
        - 💾 **Smart Caching** for performance
        
        **Supported Languages:**
        English, Hindi, Spanish, French, German,
        Bengali, Tamil, Telugu, Marathi, Gujarati,
        Urdu, Arabic, Chinese, Japanese, Korean,
        Russian, Portuguese, Italian
        
        **Models Used:**
        - Translation: NLLB-200, mBART, mT5
        - STT: OpenAI Whisper
        - TTS: Google TTS, pyttsx3
        """)
    
    with st.expander("Keyboard Shortcuts"):
        st.markdown("""
        - `Ctrl + Enter` - Translate
        - `Ctrl + R` - Record audio
        - `Ctrl + S` - Stop recording
        - `Ctrl + D` - Clear input
        - `Ctrl + H` - Toggle history
        """)
    
    with st.expander("Tips & Tricks"):
        st.markdown("""
        1. **Better Translations:**
           - Use complete sentences
           - Avoid slang and idioms
           - Check punctuation
        
        2. **Speech Input:**
           - Speak clearly and slowly
           - Minimize background noise
           - Use headset for better quality
        
        3. **Performance:**
           - Enable GPU if available
           - Use batch mode for multiple texts
           - Clear cache if experiencing issues
        """)

def render_footer():
    """Render sidebar footer."""
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; font-size: 0.8em; color: #888'>
            <p>Version 1.0.0 | © 2025</p>
            <p>Built with ❤️ using Streamlit</p>
            <p>
                <a href='#'>Documentation</a> |
                <a href='#'>GitHub</a> |
                <a href='#'>Report Issue</a>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

def get_theme_settings() -> Dict:
    """
    Get theme/appearance settings.
    
    Returns:
        Dictionary with theme settings
    """
    theme = {}
    
    with st.expander("🎨 Appearance", expanded=False):
        theme['dark_mode'] = st.checkbox("Dark Mode", value=False)
        theme['compact_view'] = st.checkbox("Compact View", value=False)
        theme['show_flags'] = st.checkbox("Show Language Flags", value=True)
        theme['animation'] = st.checkbox("Enable Animations", value=True)
    
    return theme