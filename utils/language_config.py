"""
Language Configuration Module
Central configuration for supported languages and their properties
"""

from typing import Dict, List, Tuple, Optional

# Main language configuration
SUPPORTED_LANGUAGES = {
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
    'Chinese (Simplified)': 'zh',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Russian': 'ru',
    'Portuguese': 'pt',
    'Italian': 'it'
}

# Reverse mapping
LANGUAGE_CODES = {v: k for k, v in SUPPORTED_LANGUAGES.items()}

# Language families
LANGUAGE_FAMILIES = {
    'Indo-European': {
        'Germanic': ['en', 'de'],
        'Romance': ['es', 'fr', 'pt', 'it'],
        'Indo-Aryan': ['hi', 'bn', 'mr', 'gu', 'ur'],
        'Slavic': ['ru']
    },
    'Dravidian': ['ta', 'te'],
    'Semitic': ['ar'],
    'Sino-Tibetan': ['zh'],
    'Japonic': ['ja'],
    'Koreanic': ['ko']
}

# Script information
LANGUAGE_SCRIPTS = {
    'en': {'script': 'Latin', 'direction': 'ltr'},
    'hi': {'script': 'Devanagari', 'direction': 'ltr'},
    'es': {'script': 'Latin', 'direction': 'ltr'},
    'fr': {'script': 'Latin', 'direction': 'ltr'},
    'de': {'script': 'Latin', 'direction': 'ltr'},
    'bn': {'script': 'Bengali', 'direction': 'ltr'},
    'ta': {'script': 'Tamil', 'direction': 'ltr'},
    'te': {'script': 'Telugu', 'direction': 'ltr'},
    'mr': {'script': 'Devanagari', 'direction': 'ltr'},
    'gu': {'script': 'Gujarati', 'direction': 'ltr'},
    'ur': {'script': 'Arabic', 'direction': 'rtl'},
    'ar': {'script': 'Arabic', 'direction': 'rtl'},
    'zh': {'script': 'Chinese', 'direction': 'ltr'},
    'ja': {'script': 'Japanese', 'direction': 'ltr'},
    'ko': {'script': 'Korean', 'direction': 'ltr'},
    'ru': {'script': 'Cyrillic', 'direction': 'ltr'},
    'pt': {'script': 'Latin', 'direction': 'ltr'},
    'it': {'script': 'Latin', 'direction': 'ltr'}
}

# Languages that support transliteration
TRANSLITERATION_SUPPORTED = [
    'hi', 'bn', 'ta', 'te', 'mr', 'gu', 'pa', 'kn', 'ml', 'or'
]

# Model-specific language codes (for NLLB-200)
NLLB_LANGUAGE_CODES = {
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

# mBART language codes
MBART_LANGUAGE_CODES = {
    'en': 'en_XX',
    'hi': 'hi_IN',
    'es': 'es_XX',
    'fr': 'fr_XX',
    'de': 'de_DE',
    'bn': 'bn_IN',
    'ta': 'ta_IN',
    'te': 'te_IN',
    'mr': 'mr_IN',
    'gu': 'gu_IN',
    'ur': 'ur_PK',
    'ar': 'ar_AR',
    'zh': 'zh_CN',
    'ja': 'ja_XX',
    'ko': 'ko_KR',
    'ru': 'ru_RU',
    'pt': 'pt_XX',
    'it': 'it_IT'
}

# Language pairs with special handling
SPECIAL_PAIRS = {
    ('hi', 'ur'): {
        'transliteration_only': True,
        'note': 'Hindi and Urdu are linguistically similar, mainly differ in script'
    },
    ('zh', 'ja'): {
        'shared_characters': True,
        'note': 'Chinese and Japanese share some characters (Kanji)'
    }
}

# Quality scores for language pairs (0-100)
PAIR_QUALITY_SCORES = {
    ('en', 'es'): 95,
    ('en', 'fr'): 95,
    ('en', 'de'): 93,
    ('en', 'hi'): 88,
    ('en', 'zh'): 90,
    ('en', 'ja'): 87,
    ('en', 'ar'): 85,
    ('hi', 'en'): 88,
    ('hi', 'bn'): 82,
    ('hi', 'ta'): 75,
    ('hi', 'te'): 75,
    # Add more pairs as needed
}

def get_language_name(code: str) -> str:
    """
    Get language name from code.
    
    Args:
        code: Language code
        
    Returns:
        Language name
    """
    return LANGUAGE_CODES.get(code, code)

def get_language_code(name: str) -> str:
    """
    Get language code from name.
    
    Args:
        name: Language name
        
    Returns:
        Language code
    """
    return SUPPORTED_LANGUAGES.get(name, 'en')

def get_language_script(code: str) -> Dict:
    """
    Get script information for language.
    
    Args:
        code: Language code
        
    Returns:
        Script information dictionary
    """
    return LANGUAGE_SCRIPTS.get(code, {'script': 'Unknown', 'direction': 'ltr'})

def supports_transliteration(code: str) -> bool:
    """
    Check if language supports transliteration.
    
    Args:
        code: Language code
        
    Returns:
        True if transliteration is supported
    """
    return code in TRANSLITERATION_SUPPORTED

def get_language_pair(src: str, tgt: str) -> Tuple[str, str]:
    """
    Get normalized language pair.
    
    Args:
        src: Source language code
        tgt: Target language code
        
    Returns:
        Normalized language pair
    """
    return (src, tgt)

def get_pair_quality(src: str, tgt: str) -> int:
    """
    Get translation quality score for language pair.
    
    Args:
        src: Source language code
        tgt: Target language code
        
    Returns:
        Quality score (0-100)
    """
    pair = (src, tgt)
    if pair in PAIR_QUALITY_SCORES:
        return PAIR_QUALITY_SCORES[pair]
    # Try reverse pair
    reverse_pair = (tgt, src)
    if reverse_pair in PAIR_QUALITY_SCORES:
        return PAIR_QUALITY_SCORES[reverse_pair]
    # Default score for unknown pairs
    return 70

def get_special_handling(src: str, tgt: str) -> Optional[Dict]:
    """
    Get special handling requirements for language pair.
    
    Args:
        src: Source language code
        tgt: Target language code
        
    Returns:
        Special handling dictionary or None
    """
    pair = (src, tgt)
    if pair in SPECIAL_PAIRS:
        return SPECIAL_PAIRS[pair]
    reverse_pair = (tgt, src)
    if reverse_pair in SPECIAL_PAIRS:
        return SPECIAL_PAIRS[reverse_pair]
    return None

def get_language_family(code: str) -> Tuple[str, str]:
    """
    Get language family and subfamily.
    
    Args:
        code: Language code
        
    Returns:
        Tuple of (family, subfamily)
    """
    for family, subfamilies in LANGUAGE_FAMILIES.items():
        if isinstance(subfamilies, dict):
            for subfamily, languages in subfamilies.items():
                if code in languages:
                    return (family, subfamily)
        elif isinstance(subfamilies, list):
            if code in subfamilies:
                return (family, 'Main')
    return ('Unknown', 'Unknown')

def are_related_languages(code1: str, code2: str) -> bool:
    """
    Check if two languages are related.
    
    Args:
        code1: First language code
        code2: Second language code
        
    Returns:
        True if languages are in same family
    """
    family1, _ = get_language_family(code1)
    family2, _ = get_language_family(code2)
    return family1 == family2 and family1 != 'Unknown'

def get_model_language_code(code: str, model_type: str = 'nllb') -> str:
    """
    Get model-specific language code.
    
    Args:
        code: Standard language code
        model_type: Model type ('nllb' or 'mbart')
        
    Returns:
        Model-specific language code
    """
    if model_type == 'nllb':
        return NLLB_LANGUAGE_CODES.get(code, code)
    elif model_type == 'mbart':
        return MBART_LANGUAGE_CODES.get(code, code)
    else:
        return code

def validate_language_pair(src: str, tgt: str) -> Tuple[bool, str]:
    """
    Validate if language pair is supported.
    
    Args:
        src: Source language code
        tgt: Target language code
        
    Returns:
        Tuple of (is_valid, message)
    """
    if src not in LANGUAGE_CODES:
        return False, f"Source language '{src}' not supported"
    if tgt not in LANGUAGE_CODES:
        return False, f"Target language '{tgt}' not supported"
    if src == tgt:
        return False, "Source and target languages are the same"
    return True, "Valid language pair"