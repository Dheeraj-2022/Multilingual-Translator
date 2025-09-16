"""
Audio Handler Module
Utilities for audio processing and manipulation
"""

import numpy as np
import librosa
import soundfile as sf
import io
import wave
import struct
from typing import Tuple, Optional, Union
import logging
from scipy import signal
import tempfile
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Audio processing utilities for speech input/output.
    """
    
    # Standard sample rates
    SAMPLE_RATES = {
        'whisper': 16000,
        'standard': 44100,
        'high': 48000,
        'telephone': 8000
    }
    
    def __init__(self, target_sample_rate: int = 16000):
        """
        Initialize audio processor.
        
        Args:
            target_sample_rate: Target sample rate for processing
        """
        self.target_sample_rate = target_sample_rate
        logger.info(f"AudioProcessor initialized with {target_sample_rate} Hz")
    
    def load_audio(self, 
                   audio_source: Union[str, bytes, io.BytesIO],
                   sample_rate: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Load audio from various sources.
        
        Args:
            audio_source: Audio file path, bytes, or BytesIO
            sample_rate: Desired sample rate
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            if isinstance(audio_source, str):
                # Load from file
                audio, sr = librosa.load(audio_source, sr=sample_rate)
            elif isinstance(audio_source, bytes):
                # Load from bytes
                audio, sr = sf.read(io.BytesIO(audio_source))
                if sample_rate and sr != sample_rate:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
                    sr = sample_rate
            elif isinstance(audio_source, io.BytesIO):
                # Load from BytesIO
                audio_source.seek(0)
                audio, sr = sf.read(audio_source)
                if sample_rate and sr != sample_rate:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
                    sr = sample_rate
            else:
                raise ValueError(f"Unsupported audio source type: {type(audio_source)}")
            
            return audio, sr
            
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise
    
    def resample_audio(self, 
                      audio: np.ndarray,
                      orig_sr: int,
                      target_sr: int) -> np.ndarray:
        """
        Resample audio to target sample rate.
        
        Args:
            audio: Audio array
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio array
        """
        if orig_sr == target_sr:
            return audio
            
        try:
            resampled = librosa.resample(
                audio,
                orig_sr=orig_sr,
                target_sr=target_sr
            )
            return resampled
        except Exception as e:
            logger.error(f"Error resampling audio: {e}")
            raise
    
    def normalize_audio(self, 
                       audio: np.ndarray,
                       target_db: float = -20.0) -> np.ndarray:
        """
        Normalize audio volume.
        
        Args:
            audio: Audio array
            target_db: Target dB level
            
        Returns:
            Normalized audio array
        """
        try:
            # Calculate current RMS
            rms = np.sqrt(np.mean(audio**2))
            
            # Avoid division by zero
            if rms == 0:
                return audio
            
            # Calculate scaling factor
            target_rms = 10**(target_db / 20)
            scaling_factor = target_rms / rms
            
            # Apply scaling
            normalized = audio * scaling_factor
            
            # Clip to prevent overflow
            normalized = np.clip(normalized, -1.0, 1.0)
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing audio: {e}")
            return audio
    
    def remove_silence(self,
                      audio: np.ndarray,
                      sample_rate: int,
                      top_db: int = 20) -> np.ndarray:
        """
        Remove silence from audio.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            top_db: Threshold for silence removal
            
        Returns:
            Audio with silence removed
        """
        try:
            # Get non-silent intervals
            intervals = librosa.effects.split(audio, top_db=top_db)
            
            # Concatenate non-silent parts
            if len(intervals) > 0:
                non_silent = []
                for start, end in intervals:
                    non_silent.append(audio[start:end])
                
                if non_silent:
                    return np.concatenate(non_silent)
            
            return audio
            
        except Exception as e:
            logger.error(f"Error removing silence: {e}")
            return audio
    
    def apply_noise_reduction(self,
                            audio: np.ndarray,
                            sample_rate: int,
                            noise_factor: float = 0.01) -> np.ndarray:
        """
        Apply basic noise reduction.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            noise_factor: Noise reduction factor
            
        Returns:
            Audio with reduced noise
        """
        try:
            # Simple spectral subtraction
            # Get magnitude spectrogram
            D = librosa.stft(audio)
            magnitude = np.abs(D)
            phase = np.angle(D)
            
            # Estimate noise (using first few frames)
            noise_profile = np.mean(magnitude[:, :10], axis=1, keepdims=True)
            
            # Subtract noise
            magnitude_clean = magnitude - noise_factor * noise_profile
            magnitude_clean = np.maximum(magnitude_clean, 0)
            
            # Reconstruct signal
            D_clean = magnitude_clean * np.exp(1j * phase)
            audio_clean = librosa.istft(D_clean)
            
            return audio_clean
            
        except Exception as e:
            logger.error(f"Error in noise reduction: {e}")
            return audio
    
    def convert_to_wav(self,
                      audio: np.ndarray,
                      sample_rate: int,
                      output_path: Optional[str] = None) -> Union[str, bytes]:
        """
        Convert audio array to WAV format.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            output_path: Optional output file path
            
        Returns:
            WAV file path or bytes
        """
        try:
            # Ensure audio is in correct format
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Scale to 16-bit range
            audio_scaled = np.int16(audio * 32767)
            
            if output_path:
                # Save to file
                sf.write(output_path, audio_scaled, sample_rate, subtype='PCM_16')
                return output_path
            else:
                # Return as bytes
                buffer = io.BytesIO()
                sf.write(buffer, audio_scaled, sample_rate, format='WAV', subtype='PCM_16')
                buffer.seek(0)
                return buffer.getvalue()
                
        except Exception as e:
            logger.error(f"Error converting to WAV: {e}")
            raise
    
    def create_spectrogram(self,
                          audio: np.ndarray,
                          sample_rate: int,
                          n_fft: int = 2048,
                          hop_length: int = 512) -> np.ndarray:
        """
        Create spectrogram from audio.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            n_fft: FFT window size
            hop_length: Hop length for STFT
            
        Returns:
            Spectrogram array
        """
        try:
            # Compute spectrogram
            D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            spectrogram = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            return spectrogram
            
        except Exception as e:
            logger.error(f"Error creating spectrogram: {e}")
            raise
    
    def detect_voice_activity(self,
                            audio: np.ndarray,
                            sample_rate: int,
                            frame_duration: float = 0.025,
                            energy_threshold: float = 0.01) -> np.ndarray:
        """
        Detect voice activity in audio.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            frame_duration: Frame duration in seconds
            energy_threshold: Energy threshold for voice detection
            
        Returns:
            Boolean array indicating voice activity
        """
        try:
            # Calculate frame size
            frame_size = int(sample_rate * frame_duration)
            num_frames = len(audio) // frame_size
            
            # Calculate energy for each frame
            voice_activity = np.zeros(num_frames, dtype=bool)
            
            for i in range(num_frames):
                start = i * frame_size
                end = start + frame_size
                frame = audio[start:end]
                
                # Calculate frame energy
                energy = np.sum(frame ** 2) / frame_size
                
                # Check if voice is present
                voice_activity[i] = energy > energy_threshold
            
            return voice_activity
            
        except Exception as e:
            logger.error(f"Error in voice activity detection: {e}")
            raise
    
    def enhance_speech(self,
                      audio: np.ndarray,
                      sample_rate: int) -> np.ndarray:
        """
        Enhance speech quality.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            
        Returns:
            Enhanced audio
        """
        try:
            # Apply multiple enhancement techniques
            
            # 1. Remove silence
            audio = self.remove_silence(audio, sample_rate)
            
            # 2. Apply noise reduction
            audio = self.apply_noise_reduction(audio, sample_rate)
            
            # 3. Normalize volume
            audio = self.normalize_audio(audio)
            
            # 4. Apply high-pass filter to remove low-frequency noise
            nyquist = sample_rate / 2
            cutoff = 80 / nyquist  # 80 Hz cutoff
            b, a = signal.butter(5, cutoff, btype='high')
            audio = signal.filtfilt(b, a, audio)
            
            return audio
            
        except Exception as e:
            logger.error(f"Error enhancing speech: {e}")
            return audio
    
    def chunk_audio(self,
                   audio: np.ndarray,
                   sample_rate: int,
                   chunk_duration: float = 10.0,
                   overlap: float = 0.5) -> list:
        """
        Split audio into chunks for processing.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap between chunks (0-1)
            
        Returns:
            List of audio chunks
        """
        try:
            chunk_size = int(sample_rate * chunk_duration)
            overlap_size = int(chunk_size * overlap)
            step_size = chunk_size - overlap_size
            
            chunks = []
            for i in range(0, len(audio) - chunk_size + 1, step_size):
                chunk = audio[i:i + chunk_size]
                chunks.append(chunk)
            
            # Add final chunk if there's remaining audio
            if len(audio) % step_size != 0:
                final_chunk = audio[-chunk_size:]
                chunks.append(final_chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking audio: {e}")
            return [audio]
    
    def merge_audio_chunks(self,
                          chunks: list,
                          sample_rate: int,
                          overlap: float = 0.5) -> np.ndarray:
        """
        Merge audio chunks with crossfade.
        
        Args:
            chunks: List of audio chunks
            sample_rate: Sample rate
            overlap: Overlap used in chunking
            
        Returns:
            Merged audio array
        """
        if not chunks:
            return np.array([])
        
        if len(chunks) == 1:
            return chunks[0]
        
        try:
            # Calculate overlap size
            chunk_size = len(chunks[0])
            overlap_size = int(chunk_size * overlap)
            
            # Initialize output
            total_length = sum(len(chunk) for chunk in chunks) - overlap_size * (len(chunks) - 1)
            merged = np.zeros(total_length)
            
            # Merge chunks with crossfade
            position = 0
            for i, chunk in enumerate(chunks):
                if i == 0:
                    merged[:len(chunk)] = chunk
                    position = len(chunk) - overlap_size
                else:
                    # Apply crossfade
                    fade_in = np.linspace(0, 1, overlap_size)
                    fade_out = np.linspace(1, 0, overlap_size)
                    
                    # Crossfade overlap region
                    merged[position:position + overlap_size] *= fade_out
                    merged[position:position + overlap_size] += chunk[:overlap_size] * fade_in
                    
                    # Add non-overlapping part
                    if i < len(chunks) - 1:
                        merged[position + overlap_size:position + len(chunk) - overlap_size] = chunk[overlap_size:-overlap_size]
                        position += len(chunk) - overlap_size
                    else:
                        merged[position + overlap_size:position + len(chunk)] = chunk[overlap_size:]
            
            return merged
            
        except Exception as e:
            logger.error(f"Error merging audio chunks: {e}")
            return np.concatenate(chunks)