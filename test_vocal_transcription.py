import os
from pathlib import Path
from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv(override=True)

api_key = os.getenv("SARVAM_API_KEY")

if not api_key:
    raise RuntimeError("SARVAM_API_KEY not found in .env")

client = SarvamAI(api_subscription_key=api_key)

chunk_dir = Path("downloads/vocal_chunks_5s")
chunks = sorted(chunk_dir.glob("chunk_*.wav"))

if not chunks:
    raise FileNotFoundError("No audio chunks found!")

all_transcripts = []

for i, chunk_path in enumerate(chunks, start=1):
    print(f"\nTranscribing vocal chunk {i}/{len(chunks)}...")

    with open(chunk_path, "rb") as audio:
        response = client.speech_to_text.transcribe(
            file=audio,
            model="saaras:v4",
            mode="transcribe",
            language_code="hi-IN"
        )

    transcript = getattr(response, "transcript", "") or ""

    print("Chunk transcript:", transcript)
    all_transcripts.append(transcript.strip())

final_transcript = " ".join(t for t in all_transcripts if t)

os.makedirs("downloads", exist_ok=True)

with open("downloads/vocal_hindi_transcript.txt", "w", encoding="utf-8") as f:
    f.write(final_transcript)

print("\nFINAL ISOLATED VOCAL TRANSCRIPT:\n")
print(final_transcript)

print("\nSaved: downloads/vocal_hindi_transcript.txt")





