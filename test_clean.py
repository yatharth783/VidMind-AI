import whisper

audio_path = r"D:\video_assistant\downloads\clean_audio.wav"

print("Loading Whisper Small Model...")
model = whisper.load_model("small")

print("Starting transcription...")

result = model.transcribe(
    audio_path,
    task="transcribe",
    fp16=False,
    temperature=0,
    condition_on_previous_text=False,
    initial_prompt="?? ????? ???? ??? ???? ?? ?????? ???"
)

print("\n========== TRANSCRIPT ==========\n")
print(result["text"])

print("\nDetected Language:", result["language"])
