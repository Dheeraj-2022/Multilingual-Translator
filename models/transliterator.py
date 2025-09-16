"""
Transliteration Module
Handles script conversion for Indic languages
"""

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import logging
from typing import Dict, List, Optional, Tuple
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Transliterator:
    """
    Transliteration handler for Indic scripts.
    Supports bidirectional conversion between native scripts and Roman.
    """
    
    # Script mappings
    SCRIPT_MAPPINGS = {
        'hi': {'native': sanscript.DEVANAGARI, 'name': 'Hindi'},
        'bn': {'native': sanscript.BENGALI, 'name': 'Bengali'},
        'ta': {'native': sanscript.TAMIL, 'name': 'Tamil'},
        'te': {'native': sanscript.TELUGU, 'name': 'Telugu'},
        'mr': {'native': sanscript.DEVANAGARI, 'name': 'Marathi'},
        'gu': {'native': sanscript.GUJARATI, 'name': 'Gujarati'},
        'pa': {'native': sanscript.GURMUKHI, 'name': 'Punjabi'},
        'kn': {'native': sanscript.KANNADA, 'name': 'Kannada'},
        'ml': {'native': sanscript.MALAYALAM, 'name': 'Malayalam'},
        'or': {'native': sanscript.ORIYA, 'name': 'Oriya'},
        'sa': {'native': sanscript.DEVANAGARI, 'name': 'Sanskrit'}
    }
    
    # Common transliteration schemes
    ROMAN_SCHEMES = {
        'iast': sanscript.IAST,
        'itrans': sanscript.ITRANS,
        'hk': sanscript.HK,
        'slp1': sanscript.SLP1,
        'wx': sanscript.WX,
        'velthuis': sanscript.VELTHUIS
    }
    
    def __init__(self, default_scheme: str = 'itrans'):
        """
        Initialize transliterator.
        
        Args:
            default_scheme: Default romanization scheme
        """
        self.default_scheme = self.ROMAN_SCHEMES.get(default_scheme, sanscript.ITRANS)
        logger.info(f"Transliterator initialized with {default_scheme} scheme")
        
        # Initialize custom mappings for common variations
        self._init_custom_mappings()
    
    def _init_custom_mappings(self):
        """Initialize custom character mappings for common variations."""
        self.custom_mappings = {
            # Common Hindi romanization variations
            'hindi_common': {
                'aa': 'ā', 'ee': 'ī', 'oo': 'ū',
                'sh': 'ś', 'ch': 'c', 'chh': 'ch',
                'jh': 'jh', 'th': 'ṭh', 'dh': 'ḍh',
                'ph': 'ph', 'bh': 'bh', 'gh': 'gh'
            },
            # Simplified mappings
            'simplified': {
                'ā': 'a', 'ī': 'i', 'ū': 'u',
                'ṛ': 'ri', 'ṝ': 'ri', 'ḷ': 'li',
                'ṃ': 'm', 'ḥ': 'h', 'ñ': 'n',
                'ṅ': 'n', 'ṇ': 'n', 'ś': 'sh',
                'ṣ': 'sh', 'ṭ': 't', 'ḍ': 'd'
            }
        }
    
    def transliterate_text(self,
                          text: str,
                          source_lang: str,
                          target_lang: str,
                          scheme: Optional[str] = None) -> str:
        """
        Transliterate text between scripts.
        
        Args:
            text: Input text
            source_lang: Source language code
            target_lang: Target language code
            scheme: Romanization scheme (if applicable)
            
        Returns:
            Transliterated text
        """
        if not text.strip():
            return ""
            
        try:
            # Determine source and target scripts
            source_script = self._get_script(source_lang, scheme)
            target_script = self._get_script(target_lang, scheme)
            
            if source_script == target_script:
                return text
                
            # Apply preprocessing
            text = self._preprocess_text(text, source_lang)
            
            # Perform transliteration
            result = transliterate(text, source_script, target_script)
            
            # Apply postprocessing
            result = self._postprocess_text(result, target_lang)
            
            return result
            
        except Exception as e:
            logger.error(f"Transliteration error: {e}")
            return text
    
    def _get_script(self, lang_code: str, scheme: Optional[str] = None) -> str:
        """
        Get script identifier for language.
        
        Args:
            lang_code: Language code
            scheme: Optional romanization scheme
            
        Returns:
            Script identifier
        """
        if lang_code == 'roman' or lang_code == 'en':
            return self.ROMAN_SCHEMES.get(scheme, self.default_scheme)
        elif lang_code in self.SCRIPT_MAPPINGS:
            return self.SCRIPT_MAPPINGS[lang_code]['native']
        else:
            # Default to ITRANS for unknown
            return sanscript.ITRANS
    
    def _preprocess_text(self, text: str, source_lang: str) -> str:
        """
        Preprocess text before transliteration.
        
        Args:
            text: Input text
            source_lang: Source language
            
        Returns:
            Preprocessed text
        """
        # Handle common variations in romanized text
        if source_lang in ['roman', 'en']:
            # Apply common mappings
            for old, new in self.custom_mappings.get('hindi_common', {}).items():
                text = text.replace(old, new)
                
        # Normalize Unicode
        import unicodedata
        text = unicodedata.normalize('NFC', text)
        
        return text
    
    def _postprocess_text(self, text: str, target_lang: str) -> str:
        """
        Postprocess text after transliteration.
        
        Args:
            text: Transliterated text
            target_lang: Target language
            
        Returns:
            Postprocessed text
        """
        # Clean up extra spaces
        text = ' '.join(text.split())
        
        # Handle script-specific issues
        if target_lang in ['hi', 'mr', 'sa']:
            # Fix Devanagari-specific issues
            text = self._fix_devanagari_issues(text)
        elif target_lang == 'ta':
            # Fix Tamil-specific issues
            text = self._fix_tamil_issues(text)
            
        return text
    
    def _fix_devanagari_issues(self, text: str) -> str:
        """Fix common Devanagari rendering issues."""
        # Fix halant (्) placement
        text = re.sub(r'([क-ह])्\s+', r'\1् ', text)
        
        # Fix nukta (़) placement
        text = re.sub(r'([क-ह])\s+़', r'\1़', text)
        
        return text
    
    def _fix_tamil_issues(self, text: str) -> str:
        """Fix common Tamil rendering issues."""
        # Tamil doesn't have certain consonant clusters
        # Apply simplifications as needed
        return text
    
    def to_roman(self, 
                 text: str,
                 source_lang: str,
                 scheme: str = 'itrans') -> str:
        """
        Convert text from native script to Roman.
        
        Args:
            text: Input text in native script
            source_lang: Source language code
            scheme: Romanization scheme
            
        Returns:
            Romanized text
        """
        return self.transliterate_text(text, source_lang, 'roman', scheme)
    
    def from_roman(self,
                   text: str,
                   target_lang: str,
                   scheme: str = 'itrans') -> str:
        """
        Convert text from Roman to native script.
        
        Args:
            text: Input romanized text
            target_lang: Target language code
            scheme: Romanization scheme
            
        Returns:
            Text in native script
        """
        return self.transliterate_text(text, 'roman', target_lang, scheme)
    
    def detect_script(self, text: str) -> str:
        """
        Detect the script of input text.
        
        Args:
            text: Input text
            
        Returns:
            Detected script/language code
        """
        # Simple script detection based on Unicode ranges
        for char in text:
            code = ord(char)
            
            # Devanagari
            if 0x0900 <= code <= 0x097F:
                return 'hi'
            # Bengali
            elif 0x0980 <= code <= 0x09FF:
                return 'bn'
            # Tamil
            elif 0x0B80 <= code <= 0x0BFF:
                return 'ta'
            # Telugu
            elif 0x0C00 <= code <= 0x0C7F:
                return 'te'
            # Gujarati
            elif 0x0A80 <= code <= 0x0AFF:
                return 'gu'
            # Gurmukhi
            elif 0x0A00 <= code <= 0x0A7F:
                return 'pa'
            # Kannada
            elif 0x0C80 <= code <= 0x0CFF:
                return 'kn'
            # Malayalam
            elif 0x0D00 <= code <= 0x0D7F:
                return 'ml'
            # Oriya
            elif 0x0B00 <= code <= 0x0B7F:
                return 'or'
                
        # Default to Roman if no Indic script detected
        return 'roman'
    
    def mixed_script_handler(self, text: str, target_script: str) -> str:
        """
        Handle text with mixed scripts (code-mixed text).
        
        Args:
            text: Input text with mixed scripts
            target_script: Target script for conversion
            
        Returns:
            Text with unified script
        """
        # Split text into tokens
        tokens = text.split()
        result_tokens = []
        
        for token in tokens:
            # Detect script of token
            token_script = self.detect_script(token)
            
            if token_script != 'roman' and target_script == 'roman':
                # Convert to Roman
                converted = self.to_roman(token, token_script)
            elif token_script == 'roman' and target_script != 'roman':
                # Convert from Roman
                converted = self.from_roman(token, target_script)
            else:
                # Keep as is
                converted = token
                
            result_tokens.append(converted)
            
        return ' '.join(result_tokens)
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get dictionary of supported languages for transliteration."""
        return {
            self.SCRIPT_MAPPINGS[code]['name']: code 
            for code in self.SCRIPT_MAPPINGS
        }
    
    def get_romanization_schemes(self) -> List[str]:
        """Get list of available romanization schemes."""
        return list(self.ROMAN_SCHEMES.keys())