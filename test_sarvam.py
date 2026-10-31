import os
import wave
from pathlib import Path
from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv(override=True)

api_key = os.getenv("SARVAM_API_KEY")
if not api_key:
    raise ValueError("SARVAM_API_KEY missing in .env")

client = SarvamAI(api_subscription_key=api_key)

folder = Path("downloads/sarvam_chunks")
chunks = sorted(folder.glob("chunk_*.wav"))

if not chunks:
    raise FileNotFoundError("No audio chunks found!")

all_transcripts = []

for i, chunk in enumerate(chunks, start=1):
    with wave.open(str(chunk), "rb") as audio:
        duration = audio.getnframes() / audio.getframerate()

    print(f"\n{'=' * 15} CHUNK {i}/{len(chunks)} {'=' * 15}")
    print("File:", chunk.name)
    print("Duration:", round(duration, 2), "seconds")

    with open(chunk, "rb") as audio:
        response = client.speech_to_text.transcribe(
            file=audio,
            model="saaras:v4",
            mode="transcribe",
            language_code="hi-IN"
        )

    data = response.model_dump() if hasattr(response, "model_dump") else response
    transcript = data.get("transcript", "") or ""

    print("Full response:", data)
    print("Language:", data.get("language_code"))
    print("Transcript:", repr(transcript))
    print("Transcript length:", len(transcript))

    all_transcripts.append(transcript)

final_transcript = "\n".join(
    t.strip() for t in all_transcripts if t.strip()
)

output = Path("downloads/sarvam_transcript.txt")
output.write_text(final_transcript, encoding="utf-8")

print("\nFINAL TRANSCRIPT:\n")
print(final_transcript)
print("\nSaved to:", output)

