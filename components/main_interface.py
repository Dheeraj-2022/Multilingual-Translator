"""
Main Interface Components Module
UI components for the main translation interface
"""

import streamlit as st
import time
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import base64
from io import BytesIO
import tempfile
import os

def render_translation_interface(translator, stt, tts, transliterator, settings: Dict):
    """
    Render the main translation interface.
    
    Args:
        translator: Translation model instance
        stt: Speech-to-text model instance
        tts: Text-to-speech model instance
        transliterator: Transliterator instance
        settings: Settings from sidebar
    
    Returns:
        Translation results if any
    """
    # Create two columns for input and output
    col_input, col_output = st.columns(2, gap="large")
    
    # Input section
    with col_input:
        input_data = render_input_section(stt, settings)
    
    # Output section
    with col_output:
        output_data = render_output_section()
    
    # Translation button and processing
    if st.button("🔄 **Translate**", type="primary", use_container_width=True):
        if input_data['text']:
            translation_result = process_translation(
                translator,
                input_data,
                output_data,
                settings
            )
            
            if translation_result['success']:
                # Display results
                display_translation_results(
                    translation_result,
                    tts,
                    transliterator,
                    settings
                )
                
                # Add to history
                add_to_history(input_data, output_data, translation_result)
                
                return translation_result
        else:
            st.warning("⚠️ Please enter text or record audio to translate")
    
    return None

def render_input_section(stt, settings: Dict) -> Dict:
    """
    Render the input section with text and speech options.
    
    Args:
        stt: Speech-to-text model instance
        settings: Application settings
        
    Returns:
        Dictionary with input data
    """
    st.subheader("📥 Input")
    
    input_data = {}
    
    # Language selection with flags
    input_data['source_lang'] = render_language_selector(
        "Source Language",
        key="source_lang",
        auto_detect=settings.get('translation', {}).get('auto_detect', False)
    )
    
    # Input method tabs
    tab_text, tab_speech, tab_file = st.tabs([
        "📝 Text Input",
        "🎤 Speech Input",
        "📁 File Upload"
    ])
    
    with tab_text:
        input_data['text'] = render_text_input()
    
    with tab_speech:
        speech_text = render_speech_input(stt, input_data['source_lang'])
        if speech_text:
            input_data['text'] = speech_text
    
    with tab_file:
        file_text = render_file_input()
        if file_text:
            input_data['text'] = file_text
    
    # Character count and language detection
    if input_data.get('text'):
        render_input_info(input_data['text'], settings)
    
    return input_data

def render_output_section() -> Dict:
    """
    Render the output section.
    
    Returns:
        Dictionary with output settings
    """
    st.subheader("📤 Output")
    
    output_data = {}
    
    # Target language selection
    output_data['target_lang'] = render_language_selector(
        "Target Language",
        key="target_lang",
        default_index=1
    )
    
    # Output options
    col1, col2 = st.columns(2)
    with col1:
        output_data['show_alternatives'] = st.checkbox(
            "Show alternatives",
            value=False,
            help="Display alternative translations if available"
        )
    
    with col2:
        output_data['show_confidence'] = st.checkbox(
            "Show confidence",
            value=False,
            help="Display translation confidence score"
        )
    
    # Placeholder for translation output
    output_data['container'] = st.container()
    
    return output_data

def render_language_selector(label: str, key: str, 
                            auto_detect: bool = False,
                            default_index: int = 0) -> str:
    """
    Render a language selector with optional flags.
    
    Args:
        label: Label for the selector
        key: Unique key for the widget
        auto_detect: Whether to include auto-detect option
        default_index: Default selection index
        
    Returns:
        Selected language code
    """
    from utils.language_config import SUPPORTED_LANGUAGES
    
    # Language flags mapping
    FLAGS = {
        'English': '🇬🇧',
        'Hindi': '🇮🇳',
        'Spanish': '🇪🇸',
        'French': '🇫🇷',
        'German': '🇩🇪',
        'Bengali': '🇧🇩',
        'Tamil': '🇮🇳',
        'Telugu': '🇮🇳',
        'Marathi': '🇮🇳',
        'Gujarati': '🇮🇳',
        'Urdu': '🇵🇰',
        'Arabic': '🇸🇦',
        'Chinese (Simplified)': '🇨🇳',
        'Japanese': '🇯🇵',
        'Korean': '🇰🇷',
        'Russian': '🇷🇺',
        'Portuguese': '🇵🇹',
        'Italian': '🇮🇹'
    }
    
    options = list(SUPPORTED_LANGUAGES.keys())
    
    if auto_detect:
        options = ["Auto-detect"] + options
        display_options = ["🔍 Auto-detect"] + [
            f"{FLAGS.get(lang, '🌐')} {lang}" for lang in SUPPORTED_LANGUAGES.keys()
        ]
    else:
        display_options = [
            f"{FLAGS.get(lang, '🌐')} {lang}" for lang in options
        ]
    
    selected = st.selectbox(
        label,
        options=range(len(display_options)),
        format_func=lambda x: display_options[x],
        index=default_index,
        key=key
    )
    
    if auto_detect and selected == 0:
        return "auto"
    
    actual_index = selected - 1 if auto_detect else selected
    return SUPPORTED_LANGUAGES[options[actual_index]]

def render_text_input() -> str:
    """
    Render text input area.
    
    Returns:
        Input text
    """
    # Check if there's reused text from history
    if 'reuse_text' in st.session_state:
        initial_text = st.session_state.reuse_text
        del st.session_state.reuse_text
    else:
        initial_text = ""
    
    text = st.text_area(
        "Enter text to translate",
        value=initial_text,
        height=200,
        placeholder="Type or paste your text here...",
        key="text_input_area",
        help="Maximum 5000 characters"
    )
    
    # Quick actions
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📋 Paste", use_container_width=True):
            # This would require JavaScript integration
            st.info("Use Ctrl+V to paste")
    
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.text_input_area = ""
            st.rerun()
    
    with col3:
        if st.button("📝 Example", use_container_width=True):
            st.session_state.text_input_area = "Hello! This is an example text for translation. How are you today?"
            st.rerun()
    
    return text

def render_speech_input(stt, source_lang: str) -> Optional[str]:
    """
    Render speech input interface.
    
    Args:
        stt: Speech-to-text model
        source_lang: Source language code
        
    Returns:
        Transcribed text if any
    """
    st.markdown("**🎤 Record your speech:**")
    
    # Initialize session state for recording
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    if 'audio_recorder' not in st.session_state:
        st.session_state.audio_recorder = None
    if 'transcribed_text' not in st.session_state:
        st.session_state.transcribed_text = ""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button(
            "🔴 Start Recording" if not st.session_state.is_recording else "⏸️ Recording...",
            disabled=st.session_state.is_recording,
            use_container_width=True
        ):
            from models.speech_to_text import AudioRecorder
            st.session_state.is_recording = True
            st.session_state.audio_recorder = AudioRecorder()
            st.session_state.audio_recorder.start_recording()
            st.rerun()
    
    with col2:
        if st.button(
            "⏹️ Stop Recording",
            disabled=not st.session_state.is_recording,
            use_container_width=True
        ):
            if st.session_state.audio_recorder:
                audio_data = st.session_state.audio_recorder.stop_recording()
                
                # Save and transcribe
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    st.session_state.audio_recorder.save_recording(audio_data, tmp.name)
                    temp_path = tmp.name
                
                with st.spinner("Transcribing..."):
                    result = stt.transcribe_audio_file(temp_path, language=source_lang)
                    st.session_state.transcribed_text = result['text']
                
                os.unlink(temp_path)
                st.session_state.is_recording = False
                st.rerun()
    
    with col3:
        recording_time = st.number_input(
            "Duration (sec)",
            min_value=3,
            max_value=60,
            value=10,
            key="recording_duration"
        )
    
    # Status indicator
    if st.session_state.is_recording:
        st.error("🔴 Recording in progress...")
        
        # Add visual recording indicator
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
    
    # Display transcribed text
    if st.session_state.transcribed_text:
        st.success("✅ Speech transcribed successfully!")
        transcribed = st.text_area(
            "Transcribed text:",
            value=st.session_state.transcribed_text,
            height=100,
            key="transcribed_display"
        )
        
        if st.button("Use this text", use_container_width=True):
            return transcribed
    
    return None

def render_file_input() -> Optional[str]:
    """
    Render file upload interface.
    
    Returns:
        Text from uploaded file if any
    """
    uploaded_file = st.file_uploader(
        "Choose a text file",
        type=['txt', 'doc', 'docx', 'pdf'],
        help="Supported formats: TXT, DOC, DOCX, PDF"
    )
    
    if uploaded_file is not None:
        file_text = process_uploaded_file(uploaded_file)
        
        if file_text:
            st.success(f"✅ File loaded: {uploaded_file.name}")
            
            # Display preview
            preview = st.text_area(
                "File content (preview):",
                value=file_text[:1000] + ("..." if len(file_text) > 1000 else ""),
                height=150,
                disabled=True
            )
            
            # File info
            col1, col2 = st.columns(2)
            with col1:
                st.metric("File size", f"{uploaded_file.size / 1024:.1f} KB")
            with col2:
                st.metric("Character count", len(file_text))
            
            if st.button("Use this content", use_container_width=True):
                return file_text
    
    return None

def process_uploaded_file(uploaded_file) -> Optional[str]:
    """
    Process uploaded file and extract text.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Extracted text or None
    """
    try:
        if uploaded_file.type == "text/plain":
            return uploaded_file.read().decode('utf-8')
        
        elif uploaded_file.type == "application/pdf":
            # Would need PyPDF2 or similar
            st.warning("PDF processing not yet implemented")
            return None
        
        elif uploaded_file.type in ["application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            # Would need python-docx
            st.warning("Word document processing not yet implemented")
            return None
        
        else:
            st.error("Unsupported file type")
            return None
            
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

def render_input_info(text: str, settings: Dict):
    """
    Display information about the input text.
    
    Args:
        text: Input text
        settings: Application settings
    """
    col1, col2, col3 = st.columns(3)
    
    with col1:
        char_count = len(text)
        word_count = len(text.split())
        st.metric("Characters", char_count)
    
    with col2:
        st.metric("Words", word_count)
    
    with col3:
        # Estimate translation time
        est_time = estimate_translation_time(word_count, settings)
        st.metric("Est. time", f"{est_time:.1f}s")
    
    # Language detection if enabled
    if settings.get('translation', {}).get('auto_detect', False):
        detected_lang = detect_language(text)
        if detected_lang:
            st.info(f"🔍 Detected language: {detected_lang}")

def process_translation(translator, input_data: Dict, output_data: Dict, 
                       settings: Dict) -> Dict:
    """
    Process the translation request.
    
    Args:
        translator: Translation model
        input_data: Input data dictionary
        output_data: Output settings dictionary
        settings: Application settings
        
    Returns:
        Translation results dictionary
    """
    result = {
        'success': False,
        'translation': '',
        'time_taken': 0,
        'confidence': None,
        'alternatives': []
    }
    
    try:
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Preprocessing
        status_text.text("Preprocessing text...")
        progress_bar.progress(20)
        
        from utils.preprocessing import TextPreprocessor
        preprocessor = TextPreprocessor()
        cleaned_text = preprocessor.clean_for_translation(
            input_data['text'],
            input_data['source_lang'],
            settings.get('translation', {}).get('preserve_formatting', True)
        )
        
        # Step 2: Translation
        status_text.text("Translating...")
        progress_bar.progress(50)
        
        start_time = time.time()
        
        translation, success = translator.translate_with_fallback(
            cleaned_text,
            input_data['source_lang'],
            output_data['target_lang'],
            num_beams=settings.get('model', {}).get('beam_size', 4),
            temperature=settings.get('model', {}).get('temperature', 1.0),
            max_length=settings.get('model', {}).get('max_length', 512)
        )
        
        result['time_taken'] = time.time() - start_time
        
        # Step 3: Postprocessing
        status_text.text("Finalizing...")
        progress_bar.progress(90)
        
        if success:
            result['success'] = True
            result['translation'] = translation
            
            # Get alternatives if requested
            if output_data.get('show_alternatives', False):
                # This would require modifying the translation function
                result['alternatives'] = []
            
            # Calculate confidence if requested
            if output_data.get('show_confidence', False):
                result['confidence'] = 0.95  # Placeholder
        
        # Complete
        progress_bar.progress(100)
        status_text.text("✅ Translation complete!")
        time.sleep(0.5)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.error(f"Translation error: {e}")
        result['error'] = str(e)
    
    return result

def display_translation_results(result: Dict, tts, transliterator, settings: Dict):
    """
    Display translation results.
    
    Args:
        result: Translation results dictionary
        tts: Text-to-speech model
        transliterator: Transliterator model
        settings: Application settings
    """
    if result['success']:
        # Success message
        st.success(f"✅ Translation completed in {result['time_taken']:.2f} seconds")
        
        # Main translation
        st.text_area(
            "Translation:",
            value=result['translation'],
            height=200,
            key="translation_output"
        )
        
        # Confidence score if available
        if result.get('confidence'):
            st.metric("Confidence", f"{result['confidence']*100:.1f}%")
        
        # Alternatives if available
        if result.get('alternatives'):
            with st.expander("Alternative translations"):
                for i, alt in enumerate(result['alternatives'], 1):
                    st.write(f"{i}. {alt}")
        
        # Transliteration if enabled
        if settings.get('transliteration', {}).get('enabled', False):
            render_transliteration(result['translation'], transliterator, settings)
        
        # Audio controls
        render_audio_controls(result['translation'], tts, settings)
        
        # Export options
        render_export_options(result)

def render_transliteration(text: str, transliterator, settings: Dict):
    """
    Render transliteration output.
    
    Args:
        text: Text to transliterate
        transliterator: Transliterator model
        settings: Application settings
    """
    st.subheader("🔤 Transliteration")
    
    # Perform transliteration based on settings
    scheme = settings['transliteration'].get('scheme', 'ITRANS').lower()
    direction = settings['transliteration'].get('direction', 'Auto')
    
    if direction == "Native to Roman":
        transliterated = transliterator.to_roman(text, 'hi', scheme)
    elif direction == "Roman to Native":
        transliterated = transliterator.from_roman(text, 'hi', scheme)
    else:
        # Auto-detect and convert
        detected_script = transliterator.detect_script(text)
        if detected_script == 'roman':
            transliterated = transliterator.from_roman(text, 'hi', scheme)
        else:
            transliterated = transliterator.to_roman(text, detected_script, scheme)
    
    st.text_area(
        "Transliterated text:",
        value=transliterated,
        height=100,
        key="transliteration_output"
    )

def render_audio_controls(text: str, tts, settings: Dict):
    """
    Render audio playback controls.
    
    Args:
        text: Text to synthesize
        tts: Text-to-speech model
        settings: Application settings
    """
    st.subheader("🔊 Audio Output")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("▶️ Play Audio", use_container_width=True):
            with st.spinner("Generating audio..."):
                audio_result = tts.synthesize(
                    text,
                    language='en',  # Should be target language
                    backend=settings.get('audio', {}).get('tts_backend', 'auto')
                )
                
                if audio_result['success']:
                    # Create audio player
                    if 'audio_data' in audio_result:
                        audio_bytes = audio_result['audio_data'].getvalue()
                    elif 'save_path' in audio_result:
                        with open(audio_result['save_path'], 'rb') as f:
                            audio_bytes = f.read()
                    
                    audio_b64 = base64.b64encode(audio_bytes).decode()
                    audio_html = f"""
                    <audio controls autoplay>
                        <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                    </audio>
                    """
                    st.markdown(audio_html, unsafe_allow_html=True)
    
    with col2:
        if st.button("⏸️ Stop", use_container_width=True):
            st.info("Audio stopped")
    
    with col3:
        if st.button("💾 Download Audio", use_container_width=True):
            st.info("Preparing download...")

def render_export_options(result: Dict):
    """
    Render export options for translation results.
    
    Args:
        result: Translation results
    """
    st.subheader("📥 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            label="📄 Download as TXT",
            data=result['translation'],
            file_name=f"translation_{datetime.now():%Y%m%d_%H%M%S}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col2:
        # Create JSON export
        import json
        json_data = json.dumps({
            'translation': result['translation'],
            'time_taken': result['time_taken'],
            'timestamp': datetime.now().isoformat()
        }, indent=2)
        
        st.download_button(
            label="📊 Download as JSON",
            data=json_data,
            file_name=f"translation_{datetime.now():%Y%m%d_%H%M%S}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col3:
        if st.button("📋 Copy to Clipboard", use_container_width=True):
            st.code(result['translation'])
            st.info("Text displayed above - use Ctrl+C to copy")

def add_to_history(input_data: Dict, output_data: Dict, result: Dict):
    """
    Add translation to history.
    
    Args:
        input_data: Input data
        output_data: Output settings
        result: Translation result
    """
    if 'translation_history' not in st.session_state:
        st.session_state.translation_history = []
    
    history_item = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'source_lang': input_data['source_lang'],
        'target_lang': output_data['target_lang'],
        'source_text': input_data['text'],
        'translation': result['translation'],
        'time_taken': result['time_taken']
    }
    
    st.session_state.translation_history.append(history_item)
    
    # Keep only last 100 items
    if len(st.session_state.translation_history) > 100:
        st.session_state.translation_history = st.session_state.translation_history[-100:]

def estimate_translation_time(word_count: int, settings: Dict) -> float:
    """
    Estimate translation time based on word count and settings.
    
    Args:
        word_count: Number of words
        settings: Application settings
        
    Returns:
        Estimated time in seconds
    """
    # Base time per word (rough estimate)
    base_time_per_word = 0.05
    
    # Adjust based on quality setting
    quality = settings.get('model', {}).get('quality', 'Balanced')
    quality_multiplier = {'Fast': 0.5, 'Balanced': 1.0, 'High Quality': 2.0}
    
    # Calculate estimate
    estimated_time = word_count * base_time_per_word * quality_multiplier.get(quality, 1.0)
    
    # Add overhead
    estimated_time += 0.5
    
    return estimated_time

def detect_language(text: str) -> Optional[str]:
    """
    Detect language of input text.
    
    Args:
        text: Input text
        
    Returns:
        Detected language name or None
    """
    try:
        # Simple placeholder - would use langdetect in production
        # from langdetect import detect
        # lang_code = detect(text)
        
        # For demo purposes
        if any(char >= 'अ' and char <= 'ह' for char in text):
            return "Hindi"
        elif any(char >= 'あ' and char <= 'ん' for char in text):
            return "Japanese"
        elif any(char >= '一' and char <= '龯' for char in text):
            return "Chinese"
        else:
            return "English"
    except:
        return None