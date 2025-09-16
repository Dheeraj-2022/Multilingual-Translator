"""
Setup script for Multilingual Translation System
This script can be used to install the package and create necessary directories
"""

import os
import sys
from pathlib import Path

def create_project_structure():
    """Create the complete project directory structure."""
    
    # Define directory structure
    directories = [
        'models',
        'utils',
        'components',
        'data',
        'data/temp',
        'data/cache',
        'data/cache/models',
        'data/cache/translations',
        'data/cache/audio',
        'data/cache/tts',
        'logs',
        'backups'
    ]
    
    # Create directories
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Create empty __init__.py files where needed
    init_files = [
        'models/__init__.py',
        'utils/__init__.py',
        'components/__init__.py'
    ]
    
    for init_file in init_files:
        Path(init_file).touch(exist_ok=True)
        print(f"✅ Created file: {init_file}")
    
    # Create .gitkeep files for empty directories
    gitkeep_dirs = [
        'data/temp',
        'data/cache/models',
        'data/cache/translations',
        'data/cache/audio',
        'data/cache/tts',
        'logs',
        'backups'
    ]
    
    for directory in gitkeep_dirs:
        gitkeep_path = Path(directory) / '.gitkeep'
        gitkeep_path.touch(exist_ok=True)
        print(f"✅ Created .gitkeep in: {directory}")
    
    print("\n✅ Project structure created successfully!")

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'streamlit',
        'transformers',
        'torch',
        'whisper',
        'gtts',
        'pyttsx3',
        'numpy',
        'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is NOT installed")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies are installed!")
    return True

def download_models():
    """Download required models."""
    print("\n📥 Downloading models (this may take a while)...")
    
    try:
        # Import after checking dependencies
        from models.translator import TranslationModel
        from models.speech_to_text import SpeechToText
        
        # Download translation model
        print("Downloading translation model...")
        translator = TranslationModel(
            model_type='nllb-200',
            cache_dir='./data/cache/models'
        )
        print("✅ Translation model downloaded")
        
        # Download STT model
        print("Downloading speech recognition model...")
        stt = SpeechToText(
            model_size='base',
            cache_dir='./data/cache/models'
        )
        print("✅ Speech recognition model downloaded")
        
        print("\n✅ All models downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading models: {e}")
        return False

def create_env_file():
    """Create a sample .env file if it doesn't exist."""
    if not Path('.env').exists():
        env_content = """# Multilingual Translation System - Environment Variables
MODEL_TYPE=nllb-200
MODEL_CACHE_DIR=./data/cache/models
DEFAULT_SOURCE_LANG=en
DEFAULT_TARGET_LANG=hi
STREAMLIT_SERVER_PORT=8501
LOG_LEVEL=INFO
"""
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Created .env file")
    else:
        print("ℹ️  .env file already exists")

def main():
    """Main setup function."""
    print("=" * 60)
    print("   MULTILINGUAL TRANSLATION SYSTEM - SETUP")
    print("=" * 60)
    
    # Create project structure
    print("\n📁 Creating project structure...")
    create_project_structure()
    
    # Create .env file
    print("\n📝 Creating configuration file...")
    create_env_file()
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    deps_ok = check_dependencies()
    
    if deps_ok:
        # Ask user if they want to download models
        response = input("\n📥 Download models now? (y/n): ").lower()
        if response == 'y':
            download_models()
        else:
            print("ℹ️  Models will be downloaded on first run")
    
    print("\n" + "=" * 60)
    print("   SETUP COMPLETE!")
    print("=" * 60)
    print("\nTo run the application:")
    print("1. Activate virtual environment: source venv/bin/activate")
    print("2. Run: streamlit run app.py")
    print("\nOr use the run script: ./run.sh")

if __name__ == "__main__":
    main()