from core.transcriber import transcribe_chunk_sarvam

audio_path = "downloads/slow_audio.wav"

print("Transcribing slow audio with Sarvam v3...")

transcript = transcribe_chunk_sarvam(audio_path)

print("\nSLOW AUDIO TRANSCRIPT:\n")
print(transcript)

with open("downloads/slow_hindi_transcript.txt", "w", encoding="utf-8") as f:
    f.write(transcript)

print("\nSaved: downloads/slow_hindi_transcript.txt")
