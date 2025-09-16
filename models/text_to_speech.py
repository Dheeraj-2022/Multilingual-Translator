"""
Text-to-Speech Module
Handles speech synthesis using gTTS and pyttsx3
"""

from gtts import gTTS
import pyttsx3
import pygame
import tempfile
import os
import logging
from typing import Optional, Dict, List
from io import BytesIO
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextToSpeech:
    """
    Text-to-Speech converter with multiple backend support.
    """
    
    # Language support mapping
    LANGUAGE_SUPPORT = {
        'en': {'gtts': 'en', 'pyttsx3': True, 'name': 'English'},
        'hi': {'gtts': 'hi', 'pyttsx3': True, 'name': 'Hindi'},
        'es': {'gtts': 'es', 'pyttsx3': True, 'name': 'Spanish'},
        'fr': {'gtts': 'fr', 'pyttsx3': True, 'name': 'French'},
        'de': {'gtts': 'de', 'pyttsx3': True, 'name': 'German'},
        'bn': {'gtts': 'bn', 'pyttsx3': False, 'name': 'Bengali'},
        'ta': {'gtts': 'ta', 'pyttsx3': False, 'name': 'Tamil'},
        'te': {'gtts': 'te', 'pyttsx3': False, 'name': 'Telugu'},
        'mr': {'gtts': 'mr', 'pyttsx3': False, 'name': 'Marathi'},
        'gu': {'gtts': 'gu', 'pyttsx3': False, 'name': 'Gujarati'},
        'ur': {'gtts': 'ur', 'pyttsx3': False, 'name': 'Urdu'},
        'ar': {'gtts': 'ar', 'pyttsx3': True, 'name': 'Arabic'},
        'zh': {'gtts': 'zh', 'pyttsx3': True, 'name': 'Chinese'},
        'ja': {'gtts': 'ja', 'pyttsx3': True, 'name': 'Japanese'},
        'ko': {'gtts': 'ko', 'pyttsx3': False, 'name': 'Korean'},
        'ru': {'gtts': 'ru', 'pyttsx3': True, 'name': 'Russian'},
        'pt': {'gtts': 'pt', 'pyttsx3': True, 'name': 'Portuguese'},
        'it': {'gtts': 'it', 'pyttsx3': True, 'name': 'Italian'}
    }
    
    def __init__(self, backend: str = 'auto', cache_dir: str = './data/cache'):
        """
        Initialize TTS module.
        
        Args:
            backend: TTS backend ('gtts', 'pyttsx3', or 'auto')
            cache_dir: Directory to cache audio files
        """
        self.backend = backend
        self.cache_dir = os.path.join(cache_dir, 'tts')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize pygame for audio playback
        pygame.mixer.init()
        
        # Initialize pyttsx3 engine if needed
        if backend in ['pyttsx3', 'auto']:
            try:
                self.pyttsx3_engine = pyttsx3.init()
                self._configure_pyttsx3()
                logger.info("pyttsx3 engine initialized")
            except Exception as e:
                logger.warning(f"Could not initialize pyttsx3: {e}")
                self.pyttsx3_engine = None
        else:
            self.pyttsx3_engine = None
            
        logger.info(f"TTS initialized with backend: {backend}")
    
    def _configure_pyttsx3(self):
        """Configure pyttsx3 engine settings."""
        if self.pyttsx3_engine:
            # Set properties
            self.pyttsx3_engine.setProperty('rate', 150)  # Speed
            self.pyttsx3_engine.setProperty('volume', 0.9)  # Volume
            
            # Get available voices
            voices = self.pyttsx3_engine.getProperty('voices')
            self.available_voices = {v.id: v for v in voices}
    
    def text_to_speech_gtts(self, 
                           text: str, 
                           language: str = 'en',
                           slow: bool = False) -> BytesIO:
        """
        Convert text to speech using gTTS.
        
        Args:
            text: Text to convert
            language: Language code
            slow: Whether to speak slowly
            
        Returns:
            Audio data as BytesIO object
        """
        try:
            # Get gTTS language code
            lang_code = self.LANGUAGE_SUPPORT.get(language, {}).get('gtts', 'en')
            
            # Create gTTS object
            tts = gTTS(text=text, lang=lang_code, slow=slow)
            
            # Save to BytesIO
            audio_buffer = BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            return audio_buffer
            
        except Exception as e:
            logger.error(f"gTTS error: {e}")
            raise
    
    def text_to_speech_pyttsx3(self, 
                              text: str,
                              language: str = 'en',
                              output_file: Optional[str] = None) -> Optional[str]:
        """
        Convert text to speech using pyttsx3.
        
        Args:
            text: Text to convert
            language: Language code
            output_file: Optional output file path
            
        Returns:
            Path to saved audio file if output_file specified
        """
        if not self.pyttsx3_engine:
            raise RuntimeError("pyttsx3 engine not available")
            
        try:
            # Set voice based on language if available
            self._set_voice_for_language(language)
            
            if output_file:
                # Save to file
                self.pyttsx3_engine.save_to_file(text, output_file)
                self.pyttsx3_engine.runAndWait()
                return output_file
            else:
                # Speak directly
                self.pyttsx3_engine.say(text)
                self.pyttsx3_engine.runAndWait()
                return None
                
        except Exception as e:
            logger.error(f"pyttsx3 error: {e}")
            raise
    
    def _set_voice_for_language(self, language: str):
        """Set appropriate voice for language in pyttsx3."""
        if not self.pyttsx3_engine or not self.available_voices:
            return
            
        # Simple language-based voice selection
        # This can be enhanced with better voice mapping
        for voice_id, voice in self.available_voices.items():
            if language.lower() in voice.id.lower():
                self.pyttsx3_engine.setProperty('voice', voice_id)
                break
    
    def synthesize(self, 
                  text: str,
                  language: str = 'en',
                  backend: Optional[str] = None,
                  save_path: Optional[str] = None) -> Dict:
        """
        Main synthesis method with automatic backend selection.
        
        Args:
            text: Text to synthesize
            language: Language code
            backend: Override backend selection
            save_path: Optional path to save audio
            
        Returns:
            Dictionary with audio data and metadata
        """
        if not text.strip():
            return {'success': False, 'error': 'Empty text'}
            
        # Select backend
        use_backend = backend or self.backend
        
        # Check language support
        if language not in self.LANGUAGE_SUPPORT:
            logger.warning(f"Language {language} not fully supported, defaulting to English")
            language = 'en'
            
        lang_info = self.LANGUAGE_SUPPORT[language]
        
        # Determine which backend to use
        if use_backend == 'auto':
            # Prefer gTTS for better quality if internet available
            use_backend = 'gtts'
            
        try:
            if use_backend == 'gtts':
                # Use gTTS
                audio_buffer = self.text_to_speech_gtts(text, language)
                
                if save_path:
                    # Save to file
                    with open(save_path, 'wb') as f:
                        f.write(audio_buffer.getvalue())
                        
                return {
                    'success': True,
                    'backend': 'gtts',
                    'audio_data': audio_buffer,
                    'save_path': save_path,
                    'language': language
                }
                
            elif use_backend == 'pyttsx3' and lang_info.get('pyttsx3', False):
                # Use pyttsx3
                if save_path:
                    self.text_to_speech_pyttsx3(text, language, save_path)
                else:
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                        save_path = tmp.name
                        self.text_to_speech_pyttsx3(text, language, save_path)
                        
                return {
                    'success': True,
                    'backend': 'pyttsx3',
                    'save_path': save_path,
                    'language': language
                }
                
            else:
                # Fallback to gTTS
                return self.synthesize(text, language, 'gtts', save_path)
                
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def play_audio(self, audio_source):
        """
        Play audio from file or BytesIO object.
        
        Args:
            audio_source: File path or BytesIO object
        """
        try:
            if isinstance(audio_source, str):
                # Play from file
                pygame.mixer.music.load(audio_source)
            elif isinstance(audio_source, BytesIO):
                # Play from BytesIO
                pygame.mixer.music.load(audio_source)
            else:
                raise ValueError("Invalid audio source")
                
            pygame.mixer.music.play()
            
            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                
        except Exception as e:
            logger.error(f"Audio playback error: {e}")
    
    def get_audio_base64(self, audio_source) -> str:
        """
        Convert audio to base64 for web playback.
        
        Args:
            audio_source: File path or BytesIO object
            
        Returns:
            Base64 encoded audio string
        """
        try:
            if isinstance(audio_source, str):
                with open(audio_source, 'rb') as f:
                    audio_data = f.read()
            elif isinstance(audio_source, BytesIO):
                audio_data = audio_source.getvalue()
            else:
                raise ValueError("Invalid audio source")
                
            return base64.b64encode(audio_data).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Base64 encoding error: {e}")
            return ""
    
    def cleanup_cache(self, max_files: int = 100):
        """
        Clean up old cached audio files.
        
        Args:
            max_files: Maximum number of files to keep
        """
        try:
            cache_files = os.listdir(self.cache_dir)
            if len(cache_files) > max_files:
                # Sort by modification time
                cache_files.sort(
                    key=lambda x: os.path.getmtime(os.path.join(self.cache_dir, x))
                )
                
                # Remove oldest files
                for file in cache_files[:-max_files]:
                    os.remove(os.path.join(self.cache_dir, file))
                    
                logger.info(f"Cleaned up {len(cache_files) - max_files} cache files")
                
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")