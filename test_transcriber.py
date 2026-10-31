from core.transcriber import transcribe_all

audio_chunks = [
    "downloads/sarvam_chunks/chunk_00.wav",
    "downloads/sarvam_chunks/chunk_01.wav"
]

transcript = transcribe_all(
    audio_chunks,
    language="hindi"
)

print("\nFINAL HINDI TRANSCRIPT:\n")
print(transcript)

with open(
    "downloads/final_hindi_transcript.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(transcript)

print("\nSaved successfully!")
