"""
Speech-to-Text Module
Handles speech recognition using Whisper and SpeechRecognition
"""

import whisper
import speech_recognition as sr
import numpy as np
import torch
import logging
from typing import Optional, Dict, Tuple
import tempfile
import soundfile as sf
import os
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeechToText:
    """
    Speech-to-Text converter using Whisper and fallback to Google Speech Recognition.
    """
    
    WHISPER_MODELS = {
        'tiny': 'tiny',
        'base': 'base',
        'small': 'small',
        'medium': 'medium',
        'large': 'large'
    }
    
    def __init__(self, 
                 model_size: str = 'base',
                 device: Optional[str] = None,
                 cache_dir: str = './data/cache'):
        """
        Initialize STT module.
        
        Args:
            model_size: Size of Whisper model to use
            device: Device to run model on
            cache_dir: Directory to cache models
        """
        self.model_size = model_size
        self.cache_dir = cache_dir
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Initialize recognizer for fallback
        self.recognizer = sr.Recognizer()
        
        # Load Whisper model
        self._load_whisper_model()
        
        logger.info(f"STT initialized with Whisper {model_size} on {self.device}")
    
    @lru_cache(maxsize=1)
    def _load_whisper_model(self):
        """Load Whisper model with caching."""
        try:
            logger.info(f"Loading Whisper {self.model_size} model...")
            self.whisper_model = whisper.load_model(
                self.model_size,
                device=self.device,
                download_root=self.cache_dir
            )
            logger.info("Whisper model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            self.whisper_model = None
    
    def transcribe_audio_file(self, 
                             audio_path: str,
                             language: Optional[str] = None,
                             task: str = 'transcribe') -> Dict:
        """
        Transcribe audio from file using Whisper.
        
        Args:
            audio_path: Path to audio file
            language: Language code (None for auto-detect)
            task: 'transcribe' or 'translate' (to English)
            
        Returns:
            Dictionary with transcription results
        """
        if self.whisper_model is None:
            raise RuntimeError("Whisper model not loaded")
            
        try:
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(
                audio_path,
                language=language,
                task=task,
                fp16=self.device == 'cuda'
            )
            
            return {
                'text': result['text'].strip(),
                'language': result.get('language', language),
                'segments': result.get('segments', [])
            }
            
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            raise
    
    def transcribe_microphone(self, 
                             duration: int = 5,
                             language: Optional[str] = None) -> Tuple[str, str]:
        """
        Record and transcribe from microphone.
        
        Args:
            duration: Recording duration in seconds
            language: Language code for transcription
            
        Returns:
            Tuple of (transcribed_text, detected_language)
        """
        try:
            # Record audio from microphone
            with sr.Microphone() as source:
                logger.info(f"Recording for {duration} seconds...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=duration)
                
            # Save to temporary file for Whisper
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                tmp_file.write(audio.get_wav_data())
                
            # Transcribe with Whisper
            if self.whisper_model:
                result = self.transcribe_audio_file(tmp_path, language)
                text = result['text']
                detected_lang = result['language']
            else:
                # Fallback to Google Speech Recognition
                text = self.recognizer.recognize_google(audio, language=language or 'en')
                detected_lang = language or 'en'
                
            # Clean up temporary file
            os.unlink(tmp_path)
            
            return text, detected_lang
            
        except sr.RequestError as e:
            logger.error(f"Speech recognition request error: {e}")
            return "", ""
        except sr.UnknownValueError:
            logger.warning("Could not understand audio")
            return "", ""
        except Exception as e:
            logger.error(f"Microphone transcription error: {e}")
            return "", ""
    
    def transcribe_audio_array(self,
                              audio_array: np.ndarray,
                              sample_rate: int = 16000,
                              language: Optional[str] = None) -> Dict:
        """
        Transcribe audio from numpy array.
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate of audio
            language: Language code
            
        Returns:
            Transcription results
        """
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            sf.write(tmp_path, audio_array, sample_rate)
            
        try:
            # Transcribe
            result = self.transcribe_audio_file(tmp_path, language)
            return result
        finally:
            # Clean up
            os.unlink(tmp_path)
    
    def detect_language_from_audio(self, audio_path: str) -> str:
        """
        Detect language from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Detected language code
        """
        if self.whisper_model is None:
            return 'en'  # Default to English
            
        try:
            # Use Whisper to detect language
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            
            # Make log-Mel spectrogram
            mel = whisper.log_mel_spectrogram(audio).to(self.device)
            
            # Detect language
            _, probs = self.whisper_model.detect_language(mel)
            detected_lang = max(probs, key=probs.get)
            
            return detected_lang
            
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return 'en'


class AudioRecorder:
    """
    Audio recording utility for web interface.
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize audio recorder.
        
        Args:
            sample_rate: Sample rate for recording
        """
        self.sample_rate = sample_rate
        self.recording = False
        self.audio_data = []
        
    def start_recording(self):
        """Start recording audio."""
        import sounddevice as sd
        
        self.recording = True
        self.audio_data = []
        
        def callback(indata, frames, time, status):
            if self.recording:
                self.audio_data.append(indata.copy())
                
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=callback
        )
        self.stream.start()
        logger.info("Recording started")
        
    def stop_recording(self) -> np.ndarray:
        """
        Stop recording and return audio data.
        
        Returns:
            Recorded audio as numpy array
        """
        self.recording = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
            
        if self.audio_data:
            audio_array = np.concatenate(self.audio_data, axis=0)
            logger.info(f"Recording stopped. Duration: {len(audio_array)/self.sample_rate:.2f}s")
            return audio_array.flatten()
        else:
            return np.array([])
    
    def save_recording(self, audio_array: np.ndarray, output_path: str):
        """
        Save recorded audio to file.
        
        Args:
            audio_array: Audio data
            output_path: Path to save audio file
        """
        sf.write(output_path, audio_array, self.sample_rate)
        logger.info(f"Audio saved to {output_path}")