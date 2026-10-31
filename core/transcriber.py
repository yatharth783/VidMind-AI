
import os
import tempfile
import unicodedata
from pathlib import Path
from functools import lru_cache

from pydub import AudioSegment
from dotenv import load_dotenv
from sarvamai import SarvamAI


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env", override=True)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

SARVAM_MODEL = os.getenv(
    "SARVAM_STT_MODEL",
    "saaras:v4"
)

SARVAM_PIECE_SECONDS = 25
SARVAM_OVERLAP_SECONDS = 2

DOWNLOAD_DIR = BASE_DIR / "downloads"

_whisper_model = None


# =========================================================
# LANGUAGE CONFIGURATION
# =========================================================

LANGUAGE_ALIASES = {
    "english": "english",
    "en": "english",

    "hindi": "hindi",
    "hi": "hindi",
    "hi-in": "hindi",

    "hinglish": "hinglish"
}


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_transcript(text: str) -> str:

    if not text:
        return ""

    text = unicodedata.normalize(
        "NFC",
        str(text)
    )

    text = text.replace("\ufeff", "").strip()

    bad_patterns = [
        "αñ",
        "Ã",
        "Â",
        "Ð"
    ]

    if any(pattern in text for pattern in bad_patterns):

        raise ValueError(
            "Transcript encoding corruption detected."
        )

    return text


def normalize_language(language: str) -> str:

    language = str(language).lower().strip()

    if language not in LANGUAGE_ALIASES:

        raise ValueError(
            f"Unsupported language: {language}"
        )

    return LANGUAGE_ALIASES[language]


# =========================================================
# WHISPER MODEL
# =========================================================

def load_model():

    global _whisper_model

    if _whisper_model is None:

        print(
            f"Loading Whisper model: {WHISPER_MODEL}"
        )

        try:

            import whisper

            _whisper_model = whisper.load_model(
                WHISPER_MODEL
            )

        except Exception as e:

            raise RuntimeError(
                "Whisper could not be loaded. "
                "Check your PyTorch installation and DLL permissions. "
                f"Original error: {e}"
            ) from e

        print(
            "Whisper loaded successfully."
        )

    return _whisper_model


# =========================================================
# SARVAM CLIENT
# =========================================================

@lru_cache(maxsize=1)
def get_sarvam_client():

    api_key = os.getenv(
        "SARVAM_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "SARVAM_API_KEY missing in .env file."
        )

    return SarvamAI(
        api_subscription_key=api_key
    )


# =========================================================
# WHISPER TRANSCRIPTION
# =========================================================

def transcribe_chunk_whisper(
    chunk_path: str,
    language: str = "en"
) -> str:

    if not os.path.exists(chunk_path):

        raise FileNotFoundError(
            chunk_path
        )

    model = load_model()

    print(
        f"Whisper transcription started: {language}"
    )

    result = model.transcribe(
        chunk_path,
        task="transcribe",
        language=language,
        fp16=False,
        verbose=False,
        condition_on_previous_text=True
    )

    transcript = clean_transcript(
        result.get("text", "")
    )

    print(
        "Whisper transcription completed."
    )

    return transcript


# =========================================================
# SARVAM API REQUEST
# =========================================================

def _send_to_sarvam(
    piece_path: str
) -> str:

    client = get_sarvam_client()

    with open(
        piece_path,
        "rb"
    ) as audio:

        response = client.speech_to_text.transcribe(
            file=audio,
            model=SARVAM_MODEL,
            mode="transcribe",
            language_code="hi-IN"
        )

    transcript = getattr(
        response,
        "transcript",
        ""
    ) or ""

    detected_language = getattr(
        response,
        "language_code",
        None
    )

    print(
        "Detected language:",
        detected_language
    )

    print(
        "Raw transcript:",
        repr(transcript)
    )

    return clean_transcript(
        transcript
    )


# =========================================================
# TRANSCRIPT MERGING
# =========================================================

def merge_transcripts(
    previous: str,
    current: str
) -> str:

    previous = previous.strip()
    current = current.strip()

    if not previous:
        return current

    if not current:
        return previous

    previous_words = previous.split()
    current_words = current.split()

    def normalize(words):

        return [
            word.strip(
                "।,?!;:\"'()[]{}"
            ).lower()

            for word in words
        ]

    max_overlap = min(
        len(previous_words),
        len(current_words),
        12
    )

    for size in range(
        max_overlap,
        0,
        -1
    ):

        left = normalize(
            previous_words[-size:]
        )

        right = normalize(
            current_words[:size]
        )

        if left == right:

            return " ".join(
                previous_words +
                current_words[size:]
            )

    return previous + " " + current


# =========================================================
# SARVAM HINDI / HINGLISH TRANSCRIPTION
# =========================================================

def transcribe_chunk_sarvam(
    chunk_path: str
) -> str:

    if not os.path.exists(chunk_path):

        raise FileNotFoundError(
            chunk_path
        )

    audio = AudioSegment.from_file(
        chunk_path
    )

    if len(audio) == 0:

        raise ValueError(
            "Audio file is empty."
        )

    # Standardize audio format.

    audio = (
        audio
        .set_channels(1)
        .set_frame_rate(16000)
        .set_sample_width(2)
    )

    piece_ms = (
        SARVAM_PIECE_SECONDS * 1000
    )

    overlap_ms = (
        SARVAM_OVERLAP_SECONDS * 1000
    )

    step_ms = (
        piece_ms - overlap_ms
    )

    total_pieces = (
        len(audio) + step_ms - 1
    ) // step_ms

    final_transcript = ""

    errors = []

    for i, start in enumerate(
        range(0, len(audio), step_ms),
        start=1
    ):

        piece = audio[
            start:start + piece_ms
        ]

        if len(piece) == 0:
            continue

        print(
            f"\nSarvam piece {i}/{total_pieces}"
        )

        temp_path = None

        try:

            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False
            ) as temp_file:

                temp_path = temp_file.name

            piece.export(
                temp_path,
                format="wav"
            )

            transcript = _send_to_sarvam(
                temp_path
            )

            if transcript:

                print(
                    "SUCCESS:",
                    transcript
                )

                final_transcript = merge_transcripts(
                    final_transcript,
                    transcript
                )

            else:

                print(
                    f"WARNING: Empty transcript in piece {i}"
                )

        except Exception as e:

            error_message = (
                f"Piece {i} failed: {e}"
            )

            errors.append(error_message)

            print(
                "SARVAM ERROR:",
                error_message
            )

        finally:

            if (
                temp_path
                and os.path.exists(temp_path)
            ):

                try:
                    os.remove(temp_path)

                except OSError:
                    pass

    final_transcript = clean_transcript(
        final_transcript
    )

    if not final_transcript:

        raise RuntimeError(
            "Sarvam transcription failed. "
            + "; ".join(errors)
        )

    if errors:

        print(
            f"WARNING: {len(errors)} audio pieces failed."
        )

    return final_transcript


# =========================================================
# TRANSCRIPTION ROUTER
# =========================================================

def transcribe_chunk(
    chunk_path: str,
    language: str = "english"
) -> str:

    language = normalize_language(
        language
    )

    if language in [
        "hindi",
        "hinglish"
    ]:

        print(
            f"Using Sarvam AI for {language}"
        )

        return transcribe_chunk_sarvam(
            chunk_path
        )

    print(
        "Using Whisper for English"
    )

    return transcribe_chunk_whisper(
        chunk_path,
        language="en"
    )


# =========================================================
# TRANSCRIBE ALL CHUNKS
# =========================================================

def transcribe_all(
    chunks: list,
    language: str = "english"
) -> str:

    if not chunks:

        raise ValueError(
            "No audio chunks received."
        )

    language = normalize_language(
        language
    )

    is_hindi = (
        language == "hindi"
    )

    is_hinglish = (
        language == "hinglish"
    )

    is_sarvam = (
        is_hindi or is_hinglish
    )

    # Select engine.

    if is_hinglish:

        engine = (
            f"Sarvam AI ({SARVAM_MODEL}) - Hinglish"
        )

    elif is_hindi:

        engine = (
            f"Sarvam AI ({SARVAM_MODEL}) - Hindi"
        )

    else:

        engine = (
            f"Whisper ({WHISPER_MODEL}) - English"
        )

    print("\n" + "=" * 55)

    print(
        "TRANSCRIPTION STARTED"
    )

    print(
        "Engine:",
        engine
    )

    print(
        "Language:",
        language
    )

    print(
        "Total chunks:",
        len(chunks)
    )

    print("=" * 55)

    all_transcripts = []

    for i, chunk in enumerate(
        chunks,
        start=1
    ):

        chunk_path = str(chunk)

        print(
            f"\nTranscribing chunk {i}/{len(chunks)}..."
        )

        try:

            if is_sarvam:

                text = transcribe_chunk_sarvam(
                    chunk_path
                )

            else:

                text = transcribe_chunk_whisper(
                    chunk_path,
                    language="en"
                )

            text = clean_transcript(
                text
            )

            if text:

                all_transcripts.append(
                    text
                )

            else:

                print(
                    f"WARNING: Chunk {i} returned empty."
                )

        except Exception as e:

            print(
                f"ERROR processing chunk {i}: {e}"
            )

            # Fallback for Hindi / Hinglish.

            if is_sarvam:

                print(
                    "Trying Whisper Hindi fallback..."
                )

                try:

                    text = transcribe_chunk_whisper(
                        chunk_path,
                        language="hi"
                    )

                    text = clean_transcript(
                        text
                    )

                    if text:

                        all_transcripts.append(
                            text
                        )

                except Exception as fallback_error:

                    print(
                        "Whisper fallback failed:",
                        fallback_error
                    )

    full_transcript = " ".join(
        all_transcripts
    ).strip()

    if not full_transcript:

        raise RuntimeError(
            "Transcription returned empty text. "
            "Check API key, audio quality, "
            "model availability and dependencies."
        )

    # =====================================================
    # SAVE TRANSCRIPT
    # =====================================================

    DOWNLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if is_hinglish:

        filename = "hinglish_transcript.txt"

    elif is_hindi:

        filename = "sarvam_transcript.txt"

    else:

        filename = "whisper_transcript.txt"

    output_file = (
        DOWNLOAD_DIR / filename
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            full_transcript
        )

    # =====================================================
    # FINAL OUTPUT
    # =====================================================

    print("\n" + "=" * 55)

    print(
        "TRANSCRIPTION COMPLETED"
    )

    print("=" * 55)

    print(
        "Model:",
        engine
    )

    print(
        "Transcript length:",
        len(full_transcript)
    )

    print(
        "Transcript preview:",
        repr(full_transcript[:500])
    )

    print(
        "Saved:",
        output_file
    )

    print("=" * 55)

    return full_transcript
