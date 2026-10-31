
import os
from pathlib import Path
from dotenv import load_dotenv
from sarvamai import SarvamAI

# Load environment variables
load_dotenv(override=True)

api_key = os.getenv("SARVAM_API_KEY")

if not api_key:
    raise RuntimeError("SARVAM_API_KEY missing in .env")

client = SarvamAI(api_subscription_key=api_key)

# Find available chunks
chunk_dir = Path("downloads/sarvam_chunks")

chunks = sorted(chunk_dir.glob("chunk_*.wav"))

if not chunks:
    raise FileNotFoundError(
        f"No audio chunks found in {chunk_dir}"
    )

print(f"\nFound {len(chunks)} audio chunks.")

# Test both models
models = ["saaras:v3", "saaras:v4"]

for chunk_path in chunks:

    print("\n" + "=" * 60)
    print("AUDIO FILE:", chunk_path.name)
    print("=" * 60)

    for model_name in models:

        print(f"\nMODEL: {model_name}")

        try:
            with open(chunk_path, "rb") as audio:

                response = client.speech_to_text.transcribe(
                    file=audio,
                    model=model_name,
                    mode="transcribe",
                    language_code="hi-IN"
                )

            transcript = getattr(
                response, "transcript", ""
            ) or ""

            print("TRANSCRIPT:", repr(transcript))
            print(
                "LANGUAGE:",
                getattr(response, "language_code", None)
            )

        except Exception as e:
            print("ERROR:", str(e))

print("\nAll chunk comparisons completed!")
