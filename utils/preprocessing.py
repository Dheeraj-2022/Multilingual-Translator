"""
Text Preprocessing Module
Utilities for cleaning and preprocessing text
"""

import re
import unicodedata
from typing import List, Dict, Optional, Tuple
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """
    Text preprocessing utilities for translation.
    """
    
    def __init__(self):
        """Initialize text preprocessor."""
        # Common abbreviation expansions
        self.abbreviations = {
            "Dr.": "Doctor",
            "Mr.": "Mister",
            "Mrs.": "Mistress",
            "Ms.": "Miss",
            "Jr.": "Junior",
            "Sr.": "Senior",
            "Ph.D.": "Doctor of Philosophy",
            "M.D.": "Doctor of Medicine",
            "B.A.": "Bachelor of Arts",
            "M.A.": "Master of Arts",
            "B.S.": "Bachelor of Science",
            "M.S.": "Master of Science"
        }
        
        # Emoji patterns
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        
        logger.info("TextPreprocessor initialized")
    
    def normalize_unicode(self, text: str, form: str = 'NFC') -> str:
        """
        Normalize Unicode text.
        
        Args:
            text: Input text
            form: Normalization form ('NFC', 'NFD', 'NFKC', 'NFKD')
            
        Returns:
            Normalized text
        """
        return unicodedata.normalize(form, text)
    
    def remove_control_characters(self, text: str) -> str:
        """
        Remove control characters from text.
        
        Args:
            text: Input text
            
        Returns:
            Text without control characters
        """
        # Remove all control characters except spaces, tabs, and newlines
        cleaned = ''.join(
            char for char in text 
            if unicodedata.category(char)[0] != 'C' or char in '\t\n\r'
        )
        return cleaned
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in text.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace
        """
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    
    def fix_punctuation_spacing(self, text: str, language: str = 'en') -> str:
        """
        Fix spacing around punctuation marks.
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            Text with fixed punctuation spacing
        """
        if language in ['en', 'es', 'de', 'it', 'pt']:
            # Standard punctuation rules
            text = re.sub(r'\s+([.!?,;:])', r'\1', text)
            text = re.sub(r'([.!?])\s*', r'\1 ', text)
            text = re.sub(r'\s+$', '', text)
            
        elif language == 'fr':
            # French punctuation rules
            text = re.sub(r'\s+([.!?])', r'\1', text)
            text = re.sub(r'\s*([;:!?])', r' \1', text)
            text = re.sub(r'«\s*', '« ', text)
            text = re.sub(r'\s*»', ' »', text)
            
        elif language in ['zh', 'ja', 'ko']:
            # Asian languages - minimal spacing
            text = re.sub(r'\s*([。！？、，])\s*', r'\1', text)
            
        return text
    
    def expand_abbreviations(self, text: str) -> str:
        """
        Expand common abbreviations.
        
        Args:
            text: Input text
            
        Returns:
            Text with expanded abbreviations
        """
        for abbr, expansion in self.abbreviations.items():
            text = text.replace(abbr, expansion)
        return text
    
    def remove_html_tags(self, text: str) -> str:
        """
        Remove HTML tags from text.
        
        Args:
            text: Input text
            
        Returns:
            Text without HTML tags
        """
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Decode HTML entities
        import html
        text = html.unescape(text)
        return text
    
    def handle_urls(self, text: str, action: str = 'remove') -> str:
        """
        Handle URLs in text.
        
        Args:
            text: Input text
            action: 'remove', 'replace', or 'keep'
            
        Returns:
            Processed text
        """
        url_pattern = r'https?://[^\s]+'
        
        if action == 'remove':
            text = re.sub(url_pattern, '', text)
        elif action == 'replace':
            text = re.sub(url_pattern, '[URL]', text)
        # 'keep' does nothing
        
        return text
    
    def handle_emails(self, text: str, action: str = 'remove') -> str:
        """
        Handle email addresses in text.
        
        Args:
            text: Input text
            action: 'remove', 'replace', or 'keep'
            
        Returns:
            Processed text
        """
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        if action == 'remove':
            text = re.sub(email_pattern, '', text)
        elif action == 'replace':
            text = re.sub(email_pattern, '[EMAIL]', text)
        # 'keep' does nothing
        
        return text
    
    def handle_numbers(self, text: str, action: str = 'keep') -> str:
        """
        Handle numbers in text.
        
        Args:
            text: Input text
            action: 'keep', 'remove', or 'replace'
            
        Returns:
            Processed text
        """
        if action == 'remove':
            text = re.sub(r'\d+', '', text)
        elif action == 'replace':
            text = re.sub(r'\d+', '[NUM]', text)
        # 'keep' does nothing
        
        return text
    
    def handle_emojis(self, text: str, action: str = 'keep') -> str:
        """
        Handle emojis in text.
        
        Args:
            text: Input text
            action: 'keep', 'remove', or 'replace'
            
        Returns:
            Processed text
        """
        if action == 'remove':
            text = self.emoji_pattern.sub('', text)
        elif action == 'replace':
            text = self.emoji_pattern.sub('[EMOJI]', text)
        # 'keep' does nothing
        
        return text
    
    def segment_sentences(self, text: str, language: str = 'en') -> List[str]:
        """
        Segment text into sentences.
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            List of sentences
        """
        # Simple sentence segmentation
        # For production, use nltk.sent_tokenize or spacy
        
        if language in ['en', 'es', 'fr', 'de', 'it', 'pt']:
            # Use periods, exclamation marks, and question marks
            sentences = re.split(r'(?<=[.!?])\s+', text)
        elif language in ['zh', 'ja']:
            # Use Chinese/Japanese sentence markers
            sentences = re.split(r'[。！？]', text)
        else:
            # Default splitting
            sentences = re.split(r'(?<=[.!?।])\s+', text)
        
        # Filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def clean_for_translation(self, 
                            text: str,
                            source_lang: str = 'en',
                            preserve_formatting: bool = False) -> str:
        """
        Main cleaning function for translation.
        
        Args:
            text: Input text
            source_lang: Source language code
            preserve_formatting: Whether to preserve formatting
            
        Returns:
            Cleaned text
        """
        # Apply cleaning pipeline
        text = self.normalize_unicode(text)
        text = self.remove_control_characters(text)
        
        if not preserve_formatting:
            text = self.remove_html_tags(text)
            text = self.handle_urls(text, action='replace')
            text = self.handle_emails(text, action='replace')
        
        text = self.normalize_whitespace(text)
        text = self.fix_punctuation_spacing(text, source_lang)
        
        return text
    
    def prepare_for_tts(self, text: str, language: str = 'en') -> str:
        """
        Prepare text for text-to-speech.
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            Prepared text
        """
        # Expand abbreviations for better pronunciation
        text = self.expand_abbreviations(text)
        
        # Remove URLs and emails
        text = self.handle_urls(text, action='remove')
        text = self.handle_emails(text, action='remove')
        
        # Remove emojis
        text = self.handle_emojis(text, action='remove')
        
        # Clean up
        text = self.normalize_whitespace(text)
        text = self.fix_punctuation_spacing(text, language)
        
        return text
    
    def prepare_for_display(self, text: str, max_length: Optional[int] = None) -> str:
        """
        Prepare text for display in UI.
        
        Args:
            text: Input text
            max_length: Maximum display length
            
        Returns:
            Display-ready text
        """
        # Clean text
        text = self.normalize_whitespace(text)
        
        # Truncate if needed
        if max_length and len(text) > max_length:
            text = text[:max_length-3] + "..."
        
        return text
    
    def detect_script_mixing(self, text: str) -> Dict[str, float]:
        """
        Detect mixed scripts in text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of script percentages
        """
        script_counts = Counter()
        total_chars = 0
        
        for char in text:
            if char.isspace():
                continue
                
            script_name = unicodedata.name(char, '').split()[0]
            script_counts[script_name] += 1
            total_chars += 1
        
        if total_chars == 0:
            return {}
        
        # Calculate percentages
        script_percentages = {
            script: (count / total_chars) * 100
            for script, count in script_counts.items()
        }
        
        return script_percentages
    
    def split_by_script(self, text: str) -> List[Tuple[str, str]]:
        """
        Split text by script changes.
        
        Args:
            text: Input text
            
        Returns:
            List of (text_segment, script_name) tuples
        """
        segments = []
        current_segment = []
        current_script = None
        
        for char in text:
            if char.isspace():
                current_segment.append(char)
                continue
            
            try:
                script_name = unicodedata.name(char, '').split()[0]
            except:
                script_name = 'UNKNOWN'
            
            if current_script is None:
                current_script = script_name
            
            if script_name != current_script:
                # Script change detected
                if current_segment:
                    segments.append((''.join(current_segment), current_script))
                current_segment = [char]
                current_script = script_name
            else:
                current_segment.append(char)
        
        # Add final segment
        if current_segment:
            segments.append((''.join(current_segment), current_script))
        
        return segments
    
    def normalize_quotes(self, text: str) -> str:
        """
        Normalize quote marks.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized quotes
        """
        # Convert various quote styles to standard
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        return text
    
    def remove_duplicate_punctuation(self, text: str) -> str:
        """
        Remove duplicate punctuation marks.
        
        Args:
            text: Input text
            
        Returns:
            Text without duplicate punctuation
        """
        # Remove duplicate periods, commas, etc.
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r',{2,}', ',', text)
        text = re.sub(r'!{2,}', '!', text)
        text = re.sub(r'\?{2,}', '?', text)
        
        return text