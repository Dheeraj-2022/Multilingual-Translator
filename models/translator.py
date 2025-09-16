"""
Translation Model Module
Handles multilingual translation using Hugging Face models
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    pipeline
)
from typing import List, Optional, Dict, Tuple
import logging
from functools import lru_cache
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationModel:
    """
    Wrapper for neural machine translation models.
    Supports NLLB, mBART, mT5, and other seq2seq models.
    """
    
    # Model configurations
    MODEL_CONFIGS = {
        'nllb-200': {
            'model_name': 'facebook/nllb-200-distilled-600M',
            'max_length': 512,
            'supports_languages': 200
        },
        'mbart-50': {
            'model_name': 'facebook/mbart-large-50-many-to-many-mmt',
            'max_length': 1024,
            'supports_languages': 50
        },
        'mt5-base': {
            'model_name': 'google/mt5-base',
            'max_length': 512,
            'supports_languages': 101
        }
    }
    
    # Language code mappings for NLLB
    NLLB_LANG_CODES = {
        'en': 'eng_Latn',
        'hi': 'hin_Deva',
        'es': 'spa_Latn',
        'fr': 'fra_Latn',
        'de': 'deu_Latn',
        'bn': 'ben_Beng',
        'ta': 'tam_Taml',
        'te': 'tel_Telu',
        'mr': 'mar_Deva',
        'gu': 'guj_Gujr',
        'ur': 'urd_Arab',
        'ar': 'arb_Arab',
        'zh': 'zho_Hans',
        'ja': 'jpn_Jpan',
        'ko': 'kor_Hang',
        'ru': 'rus_Cyrl',
        'pt': 'por_Latn',
        'it': 'ita_Latn'
    }
    
    def __init__(self, model_type: str = 'nllb-200', 
                 device: Optional[str] = None,
                 cache_dir: str = './data/cache'):
        """
        Initialize translation model.
        
        Args:
            model_type: Type of model to use
            device: Device to run model on (cuda/cpu)
            cache_dir: Directory to cache models
        """
        self.model_type = model_type
        self.config = self.MODEL_CONFIGS[model_type]
        self.cache_dir = cache_dir
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load model and tokenizer
        self._load_model()
        
    @lru_cache(maxsize=1)
    def _load_model(self):
        """Load model and tokenizer with caching."""
        logger.info(f"Loading {self.model_type} model...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['model_name'],
                cache_dir=self.cache_dir,
                use_fast=True
            )
            
            # Load model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config['model_name'],
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            ).to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def translate(self, 
                  text: str, 
                  src_lang: str, 
                  tgt_lang: str,
                  max_length: Optional[int] = None,
                  num_beams: int = 4,
                  temperature: float = 1.0) -> str:
        """
        Translate text from source to target language.
        
        Args:
            text: Input text to translate
            src_lang: Source language code
            tgt_lang: Target language code
            max_length: Maximum length of translation
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            
        Returns:
            Translated text
        """
        if not text.strip():
            return ""
            
        try:
            # Convert language codes for NLLB
            if self.model_type == 'nllb-200':
                src_lang = self.NLLB_LANG_CODES.get(src_lang, src_lang)
                tgt_lang = self.NLLB_LANG_CODES.get(tgt_lang, tgt_lang)
                
                # Set source language
                self.tokenizer.src_lang = src_lang
            
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config['max_length']
            ).to(self.device)
            
            # Set target language token
            if self.model_type == 'nllb-200':
                forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
            else:
                forced_bos_token_id = None
            
            # Generate translation
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=max_length or self.config['max_length'],
                    num_beams=num_beams,
                    temperature=temperature,
                    do_sample=temperature > 1.0,
                    early_stopping=True
                )
            
            # Decode output
            translation = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            return translation.strip()
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return f"Translation error: {str(e)}"
    
    def translate_batch(self, 
                       texts: List[str], 
                       src_lang: str, 
                       tgt_lang: str,
                       **kwargs) -> List[str]:
        """
        Translate multiple texts in batch.
        
        Args:
            texts: List of texts to translate
            src_lang: Source language code
            tgt_lang: Target language code
            **kwargs: Additional arguments for translation
            
        Returns:
            List of translated texts
        """
        translations = []
        for text in texts:
            translation = self.translate(text, src_lang, tgt_lang, **kwargs)
            translations.append(translation)
        return translations
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get dictionary of supported languages."""
        return {
            'English': 'en',
            'Hindi': 'hi',
            'Spanish': 'es',
            'French': 'fr',
            'German': 'de',
            'Bengali': 'bn',
            'Tamil': 'ta',
            'Telugu': 'te',
            'Marathi': 'mr',
            'Gujarati': 'gu',
            'Urdu': 'ur',
            'Arabic': 'ar',
            'Chinese': 'zh',
            'Japanese': 'ja',
            'Korean': 'ko',
            'Russian': 'ru',
            'Portuguese': 'pt',
            'Italian': 'it'
        }
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of input text.
        
        Args:
            text: Input text
            
        Returns:
            Detected language code
        """
        # For simplicity, using a basic approach
        # In production, use langdetect or fasttext
        try:
            from langdetect import detect
            lang_code = detect(text)
            return lang_code
        except:
            # Default to English if detection fails
            return 'en'


class TranslationPipeline:
    """
    High-level pipeline for translation with preprocessing and postprocessing.
    """
    
    def __init__(self, model_type: str = 'nllb-200'):
        """Initialize translation pipeline."""
        self.model = TranslationModel(model_type=model_type)
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text before translation.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        # Handle special characters if needed
        # text = text.replace('...', '…')
        
        return text.strip()
    
    def postprocess_text(self, text: str, target_lang: str) -> str:
        """
        Postprocess translated text.
        
        Args:
            text: Translated text
            target_lang: Target language code
            
        Returns:
            Postprocessed text
        """
        # Fix spacing around punctuation
        text = text.replace(' .', '.')
        text = text.replace(' ,', ',')
        text = text.replace(' ?', '?')
        text = text.replace(' !', '!')
        
        # Language-specific postprocessing
        if target_lang == 'fr':
            # French spacing rules
            text = text.replace('«', '« ')
            text = text.replace('»', ' »')
            
        return text.strip()
    
    def translate_with_fallback(self, 
                               text: str,
                               src_lang: str,
                               tgt_lang: str,
                               **kwargs) -> Tuple[str, bool]:
        """
        Translate with fallback mechanism.
        
        Args:
            text: Input text
            src_lang: Source language
            tgt_lang: Target language
            
        Returns:
            Tuple of (translated_text, success_flag)
        """
        try:
            # Preprocess
            text = self.preprocess_text(text)
            
            # Translate
            translation = self.model.translate(text, src_lang, tgt_lang, **kwargs)
            
            # Postprocess
            translation = self.postprocess_text(translation, tgt_lang)
            
            return translation, True
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            # Return original text as fallback
            return text, False