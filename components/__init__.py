from .sidebar import (
    render_sidebar,
    render_history,
    render_model_settings,
    render_translation_settings,
    render_transliteration_settings,
    render_audio_settings,
    render_about,
    render_footer,
    get_theme_settings
)

from .main_interface import (
    render_translation_interface,
    render_input_section,
    render_output_section,
    render_audio_controls,
    render_language_selector,
    render_text_input,
    render_speech_input,
    render_file_input,
    process_translation,
    display_translation_results,
    add_to_history
)

__all__ = [
    # Sidebar components
    'render_sidebar',
    'render_history',
    'render_model_settings',
    'render_translation_settings',
    'render_transliteration_settings',
    'render_audio_settings',
    'render_about',
    'render_footer',
    'get_theme_settings',
    # Main interface components
    'render_translation_interface',
    'render_input_section',
    'render_output_section',
    'render_audio_controls',
    'render_language_selector',
    'render_text_input',
    'render_speech_input',
    'render_file_input',
    'process_translation',
    'display_translation_results',
    'add_to_history'
]

__version__ = '1.0.0'