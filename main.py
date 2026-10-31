
import os
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title

from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions
)

from core.rag_engine import (
    build_rag_chain,
    ask_question
)


load_dotenv(override=True)


# ==========================================
# CONFIGURATION
# ==========================================

SUPPORTED_LANGUAGES = [
    "english",
    "hindi"
]


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def validate_input(source, language):

    if not source:
        raise ValueError("Video URL or file path is required.")

    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            "Unsupported language. Choose english or hindi."
        )

    if not source.startswith(("http://", "https://")):

        if not os.path.isfile(source):
            raise FileNotFoundError(
                f"Input file not found: {source}"
            )


def safe_execute(function, transcript, stage_name):

    try:
        print(f"\nProcessing {stage_name}...")

        result = function(transcript)

        if result is None:
            return "Not enough information available."

        return result

    except Exception as e:

        print(f"WARNING: {stage_name} failed: {e}")

        return f"{stage_name} could not be generated."


# ==========================================
# MAIN PIPELINE
# ==========================================

def run_pipeline(source: str, language: str = "english") -> dict:

    print("\nStarting AI Video Assistant...")

    language = language.lower().strip()

    validate_input(source, language)

    # ======================================
    # STEP 1: AUDIO PROCESSING
    # ======================================

    print("\n[1/7] Processing input...")

    chunks = process_input(source)

    if not chunks:
        raise ValueError(
            "No audio chunks generated."
        )

    print(f"Audio ready: {len(chunks)} chunk(s)")

    # ======================================
    # STEP 2: TRANSCRIPTION
    # ======================================

    print("\n[2/7] Transcribing audio...")

    transcript = transcribe_all(
        chunks,
        language
    )

    if not transcript or len(transcript.strip()) < 10:

        raise ValueError(
            "Transcription failed or returned insufficient text."
        )

    transcript = transcript.strip()

    print("\nTranscription completed.")

    print("Transcript length:", len(transcript))

    print(
        "Transcript preview:",
        repr(transcript[:300])
    )

    # ======================================
    # STEP 3: TITLE GENERATION
    # ======================================

    print("\n[3/7] Generating title...")

    title = safe_execute(
        generate_title,
        transcript,
        "Title generation"
    )

    # ======================================
    # STEP 4: SUMMARY GENERATION
    # ======================================

    print("\n[4/7] Generating summary...")

    meeting_summary = safe_execute(
        summarize,
        transcript,
        "Summary generation"
    )

    # ======================================
    # STEP 5: ACTION ITEMS
    # ======================================

    print("\n[5/7] Extracting action items...")

    action_items = safe_execute(
        extract_action_items,
        transcript,
        "Action item extraction"
    )

    # ======================================
    # STEP 6: DECISIONS AND QUESTIONS
    # ======================================

    print("\n[6/7] Extracting decisions and questions...")

    decisions = safe_execute(
        extract_key_decisions,
        transcript,
        "Decision extraction"
    )

    questions = safe_execute(
        extract_questions,
        transcript,
        "Question extraction"
    )

    # ======================================
    # STEP 7: RAG
    # ======================================

    print("\n[7/7] Building RAG...")

    rag_chain = None

    try:

        rag_chain = build_rag_chain(transcript)

        print("RAG initialized successfully.")

    except Exception as e:

        print("WARNING: RAG initialization failed:", e)

    # ======================================
    # FINAL RESULT
    # ======================================

    print("\nPipeline completed!")

    return {

        "title": title,

        "transcript": transcript,

        "summary": meeting_summary,

        "action_items": action_items,

        "key_decisions": decisions,

        "open_questions": questions,

        "rag_chain": rag_chain

    }


# ==========================================
# DISPLAY RESULTS
# ==========================================

def display_results(result):

    print("\n" + "=" * 65)

    print("VIDEO TITLE")

    print("=" * 65)

    print(result["title"])

    print("\n" + "=" * 65)

    print("VIDEO SUMMARY")

    print("=" * 65)

    print(result["summary"])

    print("\n" + "=" * 65)

    print("ACTION ITEMS")

    print("=" * 65)

    print(result["action_items"])

    print("\n" + "=" * 65)

    print("KEY DECISIONS")

    print("=" * 65)

    print(result["key_decisions"])

    print("\n" + "=" * 65)

    print("OPEN QUESTIONS")

    print("=" * 65)

    print(result["open_questions"])

    print("\n" + "=" * 65)

    print("FULL TRANSCRIPT")

    print("=" * 65)

    print(result["transcript"])


# ==========================================
# RAG CHAT
# ==========================================

def chat_with_video(rag_chain):

    if rag_chain is None:

        print("\nRAG is unavailable.")

        return

    print("\nChat with your video!")

    print("Ask anything about the video.")

    print("Type 'exit' to quit.\n")

    while True:

        try:

            question = input("You: ").strip()

            if question.lower() in [
                "exit",
                "quit",
                "q"
            ]:

                print("Goodbye!")

                break

            if not question:
                continue

            print("\nThinking...\n")

            answer = ask_question(
                rag_chain,
                question
            )

            print("Assistant:", answer)

            print()

        except KeyboardInterrupt:

            print("\nChat stopped.")

            break

        except Exception as e:

            print("\nChat Error:", e)


# ==========================================
# APPLICATION ENTRY POINT
# ==========================================

if __name__ == "__main__":

    print("=" * 65)

    print("AI VIDEO ASSISTANT")

    print("=" * 65)

    source = input(
        "\nEnter YouTube URL or local file path: "
    ).strip()

    language = input(
        "\nLanguage (english/hindi): "
    ).strip().lower() or "english"

    try:

        result = run_pipeline(
            source,
            language
        )

        display_results(result)

        chat_with_video(
            result["rag_chain"]
        )

    except Exception as e:

        print("\nAPPLICATION ERROR:", str(e))

        print(
            "\nPlease check the input URL, "
            "audio processing and API configuration."
        )
