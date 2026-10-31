import os
from pathlib import Path
from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv(override=True)

client = SarvamAI(
    api_subscription_key=os.getenv("SARVAM_API_KEY")
)

audio_path = Path("downloads/sarvam_chunks/chunk_00.wav")

with open(audio_path, "rb") as audio:
    response = client.speech_to_text.transcribe(
        file=audio,
        model="saaras:v3",
        mode="transcribe",
        language_code="hi-IN"
    )

transcript = response.transcript or ""

Path("downloads/sarvam_v3_test.txt").write_text(
    transcript,
    encoding="utf-8"
)

print("\nSARVAM V3 TRANSCRIPT:\n")
print(transcript)
print("\nSaved: downloads/sarvam_v3_test.txt")
