import whisper

audio_path = r"D:\video_assistant\downloads\hindi_boosted.wav"

print("Loading Whisper...")
model = whisper.load_model("base")

print("Transcribing Hindi audio...")

result = model.transcribe(
    audio_path,
    language="hi",
    task="transcribe",
    fp16=False,
    temperature=0,
    condition_on_previous_text=False
)

print("\n===== HINDI TRANSCRIPT =====\n")
print(result["text"])
print("\n===== LANGUAGE =====")
print(result["language"])
