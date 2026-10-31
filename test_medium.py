import whisper

audio_path = r"D:\video_assistant\downloads\clean_audio.wav"
model_path = r"D:\video_assistant\models\whisper"

print("Loading Whisper Medium model...")
model = whisper.load_model("medium", download_root=model_path)

print("Transcribing Hindi audio...")

result = model.transcribe(
    audio_path,
    language="hi",
    task="transcribe",
    fp16=False,
    temperature=0,
    condition_on_previous_text=False,
    initial_prompt="नमस्ते, यह हिंदी में बोली गई बातचीत है।"
)

print("\n========== TRANSCRIPT ==========\n")
print(result["text"])

print("\nDetected Language:", result["language"])
