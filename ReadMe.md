# 🌍 Multilingual Translation System  

## 📖 Overview  
The **Multilingual Translation System** is an end-to-end application designed to break down language barriers by enabling **text and speech translation** across **18+ languages**, including support for **Indic scripts** with **transliteration**.  

It integrates **speech-to-text (STT)**, **machine translation (MT)**, and **text-to-speech (TTS)** in a single pipeline with a **Streamlit-based UI** for seamless interaction.  

---

## 🚀 Features  
- ✍️ **Text Input & File Upload** – Translate typed text or uploaded `.txt` files.  
- 🎤 **Speech Input** – Record audio, transcribe with **Whisper**, and translate.  
- 🔤 **Indic Transliteration** – Convert scripts into romanized format.  
- 🔊 **Text-to-Speech Output** – Listen to translations using **gTTS / pyttsx3**.  
- 📜 **History Panel** – View and manage past translations.  
- ⚡ **Caching & Beam Search** – Optimized performance with adjustable translation quality.  

---

## 🛠️ Tech Stack  
- **Frontend/UI:** Streamlit  
- **Translation Model:** Hugging Face NLLB-200  
- **Speech-to-Text (STT):** OpenAI Whisper  
- **Text-to-Speech (TTS):** gTTS / pyttsx3  
- **Audio Processing:** FFmpeg  
- **Backend:** Python (Transformers, Torch)  

