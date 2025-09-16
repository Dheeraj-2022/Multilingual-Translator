"""
Cache Manager Module
Handles model caching and memory management
"""

import os
import json
import pickle
import hashlib
import shutil
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelCache:
    """
    Manages caching of models and translations.
    """
    
    def __init__(self, cache_dir: str = './data/cache', max_size_gb: float = 5.0):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for cache storage
            max_size_gb: Maximum cache size in GB
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        
        # Create cache directories
        self.model_cache_dir = self.cache_dir / 'models'
        self.translation_cache_dir = self.cache_dir / 'translations'
        self.audio_cache_dir = self.cache_dir / 'audio'
        
        for dir_path in [self.model_cache_dir, self.translation_cache_dir, self.audio_cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load cache metadata
        self.metadata_file = self.cache_dir / 'cache_metadata.json'
        self.metadata = self._load_metadata()
        
        logger.info(f"Cache initialized at {self.cache_dir}")
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
        return {
            'translations': {},
            'audio': {},
            'models': {},
            'stats': {
                'hits': 0,
                'misses': 0,
                'total_size': 0
            }
        }
    
    def _save_metadata(self):
        """Save cache metadata to file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def _get_cache_key(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        Generate cache key for translation.
        
        Args:
            text: Source text
            src_lang: Source language
            tgt_lang: Target language
            
        Returns:
            Cache key
        """
        content = f"{text}_{src_lang}_{tgt_lang}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_translation(self, text: str, src_lang: str, tgt_lang: str) -> Optional[str]:
        """
        Get cached translation if available.
        
        Args:
            text: Source text
            src_lang: Source language
            tgt_lang: Target language
            
        Returns:
            Cached translation or None
        """
        cache_key = self._get_cache_key(text, src_lang, tgt_lang)
        
        if cache_key in self.metadata['translations']:
            cache_info = self.metadata['translations'][cache_key]
            cache_file = self.translation_cache_dir / f"{cache_key}.txt"
            
            if cache_file.exists():
                # Check if cache is still valid (24 hours)
                if datetime.fromisoformat(cache_info['timestamp']) > datetime.now() - timedelta(hours=24):
                    try:
                        with open(cache_file, 'r', encoding='utf-8') as f:
                            translation = f.read()
                        
                        self.metadata['stats']['hits'] += 1
                        self._save_metadata()
                        logger.info(f"Cache hit for translation: {cache_key}")
                        return translation
                    except Exception as e:
                        logger.error(f"Error reading cache: {e}")
        
        self.metadata['stats']['misses'] += 1
        self._save_metadata()
        return None
    
    def save_translation(self, text: str, src_lang: str, tgt_lang: str, translation: str):
        """
        Save translation to cache.
        
        Args:
            text: Source text
            src_lang: Source language
            tgt_lang: Target language
            translation: Translation result
        """
        cache_key = self._get_cache_key(text, src_lang, tgt_lang)
        cache_file = self.translation_cache_dir / f"{cache_key}.txt"
        
        try:
            # Save translation
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(translation)
            
            # Update metadata
            self.metadata['translations'][cache_key] = {
                'timestamp': datetime.now().isoformat(),
                'src_lang': src_lang,
                'tgt_lang': tgt_lang,
                'size': len(translation)
            }
            self._save_metadata()
            
            logger.info(f"Translation cached: {cache_key}")
            
            # Check cache size
            self._manage_cache_size()
            
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
    
    def get_audio(self, text: str, language: str, backend: str) -> Optional[bytes]:
        """
        Get cached audio if available.
        
        Args:
            text: Text content
            language: Language code
            backend: TTS backend used
            
        Returns:
            Cached audio data or None
        """
        cache_key = hashlib.md5(f"{text}_{language}_{backend}".encode()).hexdigest()
        
        if cache_key in self.metadata['audio']:
            cache_file = self.audio_cache_dir / f"{cache_key}.mp3"
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        return f.read()
                except Exception as e:
                    logger.error(f"Error reading audio cache: {e}")
        
        return None
    
    def save_audio(self, text: str, language: str, backend: str, audio_data: bytes):
        """
        Save audio to cache.
        
        Args:
            text: Text content
            language: Language code
            backend: TTS backend used
            audio_data: Audio data
        """
        cache_key = hashlib.md5(f"{text}_{language}_{backend}".encode()).hexdigest()
        cache_file = self.audio_cache_dir / f"{cache_key}.mp3"
        
        try:
            with open(cache_file, 'wb') as f:
                f.write(audio_data)
            
            self.metadata['audio'][cache_key] = {
                'timestamp': datetime.now().isoformat(),
                'language': language,
                'backend': backend,
                'size': len(audio_data)
            }
            self._save_metadata()
            
            logger.info(f"Audio cached: {cache_key}")
            
        except Exception as e:
            logger.error(f"Error saving audio to cache: {e}")
    
    def _get_cache_size(self) -> int:
        """
        Calculate total cache size in bytes.
        
        Returns:
            Total size in bytes
        """
        total_size = 0
        for cache_dir in [self.translation_cache_dir, self.audio_cache_dir, self.model_cache_dir]:
            for file_path in cache_dir.iterdir():
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        return total_size
    
    def _manage_cache_size(self):
        """Manage cache size by removing old entries if needed."""
        current_size = self._get_cache_size()
        
        if current_size > self.max_size_bytes:
            logger.info(f"Cache size ({current_size / 1024 / 1024:.2f} MB) exceeds limit. Cleaning...")
            
            # Collect all cache entries with timestamps
            all_entries = []
            
            # Translation entries
            for key, info in self.metadata['translations'].items():
                all_entries.append({
                    'type': 'translation',
                    'key': key,
                    'timestamp': datetime.fromisoformat(info['timestamp']),
                    'size': info.get('size', 0)
                })
            
            # Audio entries
            for key, info in self.metadata['audio'].items():
                all_entries.append({
                    'type': 'audio',
                    'key': key,
                    'timestamp': datetime.fromisoformat(info['timestamp']),
                    'size': info.get('size', 0)
                })
            
            # Sort by timestamp (oldest first)
            all_entries.sort(key=lambda x: x['timestamp'])
            
            # Remove oldest entries until size is acceptable
            for entry in all_entries:
                if current_size <= self.max_size_bytes * 0.8:  # Keep 80% full
                    break
                
                if entry['type'] == 'translation':
                    cache_file = self.translation_cache_dir / f"{entry['key']}.txt"
                    if cache_file.exists():
                        cache_file.unlink()
                        current_size -= entry['size']
                    del self.metadata['translations'][entry['key']]
                    
                elif entry['type'] == 'audio':
                    cache_file = self.audio_cache_dir / f"{entry['key']}.mp3"
                    if cache_file.exists():
                        cache_file.unlink()
                        current_size -= entry['size']
                    del self.metadata['audio'][entry['key']]
            
            self._save_metadata()
            logger.info(f"Cache cleaned. New size: {current_size / 1024 / 1024:.2f} MB")
    
    def clear_cache(self, cache_type: Optional[str] = None):
        """
        Clear cache.
        
        Args:
            cache_type: Type of cache to clear ('translations', 'audio', 'all')
        """
        if cache_type in ['translations', 'all']:
            shutil.rmtree(self.translation_cache_dir)
            self.translation_cache_dir.mkdir(exist_ok=True)
            self.metadata['translations'] = {}
            logger.info("Translation cache cleared")
        
        if cache_type in ['audio', 'all']:
            shutil.rmtree(self.audio_cache_dir)
            self.audio_cache_dir.mkdir(exist_ok=True)
            self.metadata['audio'] = {}
            logger.info("Audio cache cleared")
        
        if cache_type == 'all':
            self.metadata['stats'] = {'hits': 0, 'misses': 0, 'total_size': 0}
        
        self._save_metadata()
    
    def get_stats(self) -> Dict:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics dictionary
        """
        stats = self.metadata['stats'].copy()
        stats['total_size_mb'] = self._get_cache_size() / 1024 / 1024
        stats['translation_count'] = len(self.metadata['translations'])
        stats['audio_count'] = len(self.metadata['audio'])
        
        if stats['hits'] + stats['misses'] > 0:
            stats['hit_rate'] = stats['hits'] / (stats['hits'] + stats['misses']) * 100
        else:
            stats['hit_rate'] = 0
        
        return stats


class TranslationBuffer:
    """
    Buffer for batch translation processing.
    """
    
    def __init__(self, max_size: int = 10, max_wait_time: float = 2.0):
        """
        Initialize translation buffer.
        
        Args:
            max_size: Maximum buffer size before processing
            max_wait_time: Maximum wait time in seconds
        """
        self.max_size = max_size
        self.max_wait_time = max_wait_time
        self.buffer = []
        self.last_add_time = datetime.now()
    
    def add(self, text: str, src_lang: str, tgt_lang: str) -> bool:
        """
        Add text to buffer.
        
        Args:
            text: Text to translate
            src_lang: Source language
            tgt_lang: Target language
            
        Returns:
            True if buffer should be processed
        """
        self.buffer.append({
            'text': text,
            'src_lang': src_lang,
            'tgt_lang': tgt_lang,
            'timestamp': datetime.now()
        })
        
        self.last_add_time = datetime.now()
        
        # Check if buffer should be processed
        if len(self.buffer) >= self.max_size:
            return True
        
        if (datetime.now() - self.last_add_time).total_seconds() > self.max_wait_time:
            return True
        
        return False
    
    def get_batch(self) -> List[Dict]:
        """
        Get and clear buffer.
        
        Returns:
            List of buffered items
        """
        batch = self.buffer.copy()
        self.buffer = []
        return batch
    
    def clear(self):
        """Clear the buffer."""
        self.buffer = []