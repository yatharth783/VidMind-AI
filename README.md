
# 🎬 AI Video Assistant

An AI-powered video assistant that converts YouTube videos and audio into transcripts, generates summaries, and enables users to chat with video content.

## 🚀 Features

- YouTube video audio extraction
- Audio transcription using Sarvam AI
- English transcription using OpenAI Whisper
- Hindi and Hinglish transcription support
- Automatic audio chunking
- AI-powered summarization
- Local LLM integration using Ollama
- Video-based question answering (RAG)

## 🛠️ Tech Stack

- Python
- Streamlit
- Sarvam AI
- OpenAI Whisper
- Ollama
- Llama 3.2
- LangChain
- PyDub
- FFmpeg
- YouTube Transcript / Audio Processing

## 📂 Project Structure

```text
video_assistant/
│
├── app.py
├── core/
│   └── transcriber.py
├── utils/
├── downloads/
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## ⚙️ Installation

### 1. Clone repository

```bash
git clone https://github.com/yatharth783/AI-Video-Assistant.git
cd AI-Video-Assistant
```

### 2. Create virtual environment

```bash
python -m venv .venv
```

### 3. Activate environment

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a `.env` file:

```env
SARVAM_API_KEY=your_api_key
SARVAM_STT_MODEL=saaras:v4
WHISPER_MODEL=small
```

### 6. Install and start Ollama

Download Ollama from:

https://ollama.com/

Pull model:

```bash
ollama pull llama3.2
```

### 7. Run application

```bash
streamlit run app.py
```

## 🔐 Security

Never upload API keys, `.env` files, private credentials, or downloaded media files.

## 👨‍💻 Author

Yatharth Pathak

GitHub: https://github.com/yatharth783
