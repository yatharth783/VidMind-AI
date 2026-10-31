from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title

with open(
    "downloads/final_hindi_transcript.txt",
    "r",
    encoding="utf-8"
) as f:
    transcript = f.read()

print("\nGenerating song title...")
title = generate_title(transcript)
print("\nSONG TITLE:\n", title)

print("\nAnalyzing song lyrics...")
summary = summarize(transcript)
print("\nSONG ANALYSIS:\n", summary)
