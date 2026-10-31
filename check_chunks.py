import wave
from pathlib import Path

for path in sorted(Path("downloads/sarvam_chunks").glob("*.wav")):
    with wave.open(str(path), "rb") as audio:
        frames = audio.getnframes()
        sr = audio.getframerate()
        channels = audio.getnchannels()
        duration = frames / sr

        print("\nFile:", path.name)
        print("Duration:", round(duration, 2), "seconds")
        print("Sample rate:", sr)
        print("Channels:", channels)
        print("Audio frames:", frames)
