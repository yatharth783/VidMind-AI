import whisper

audio_path = r"D:\video_assistant\downloads\clean_audio.wav"

print("Loading Whisper Small...")
model = whisper.load_model("small")

print("Transcribing in Hindi...")

result = model.transcribe(
    audio_path,
    language="hi",
    task="transcribe",
    fp16=False,
    temperature=0,
    condition_on_previous_text=False,
    no_speech_threshold=0.6
)

print("\n========== HINDI TRANSCRIPT ==========\n")
print(result["text"])

print("\nDetected Language:", result["language"])

print("\n========== SEGMENTS ==========\n")
for segment in result["segments"]:
    print(segment["text"])
