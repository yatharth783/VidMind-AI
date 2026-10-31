from dotenv import load_dotenv
load_dotenv()

import whisper

audio_path = r"D:\video_assistant\downloads\test_audio.wav"

print("Loading Whisper model...")
model = whisper.load_model("base")

print("Transcribing audio...")
result = model.transcribe(
    audio_path,
    fp16=False,
    verbose=False
)

print("\n===== TRANSCRIPT =====\n")
print(result["text"])
print("\n===== LANGUAGE =====")
print(result["language"])
