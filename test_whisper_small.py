import whisper

audio_path = r"D:\video_assistant\downloads\test_audio.wav"

print("Loading Whisper Small model...")
model = whisper.load_model("small")

print("Transcribing...")
result = model.transcribe(
    audio_path,
    fp16=False,
    verbose=False,
    task="transcribe"
)

print("\n===== TRANSCRIPT =====\n")
print(result["text"])

print("\n===== LANGUAGE =====")
print(result["language"])
