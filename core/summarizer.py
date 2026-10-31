
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==========================================
# LLM CONFIGURATION
# ==========================================

def get_llm():
    return ChatOllama(
        model="llama3.2",
        temperature=0,
        num_predict=2048
    )


def invoke_llm(system_prompt, human_prompt, data):

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])

    chain = prompt | get_llm() | StrOutputParser()

    return chain.invoke(data).strip()


# ==========================================
# CONTENT TYPE DETECTION
# ==========================================

def detect_content_type(transcript):

    system_prompt = """
You are an expert video content classifier.

Classify the transcript into exactly ONE category:

1. Meeting
2. Educational Lecture
3. Tutorial
4. Interview
5. Podcast
6. News
7. Recipe
8. Entertainment
9. Other

Rules:
- Focus on actual spoken content.
- Do not classify based only on keywords.
- Return only the category name.
- Never invent context.
"""

    result = invoke_llm(
        system_prompt,
        "Classify this transcript:\n\n{text}",
        {"text": transcript[:5000]}
    )

    allowed = [
        "Meeting",
        "Educational Lecture",
        "Tutorial",
        "Interview",
        "Podcast",
        "News",
        "Recipe",
        "Entertainment",
        "Other"
    ]

    for category in allowed:
        if result.lower() == category.lower():
            print("Detected content type:", category)
            return category

    return "Other"


# ==========================================
# CONTENT ANALYSIS
# ==========================================

def summarize(transcript):

    if not transcript or len(transcript.split()) < 30:
        return "Insufficient transcript for reliable analysis."

    content_type = detect_content_type(transcript)

    section_map = {

        "Meeting": """
        1. Meeting Overview
        2. Main Discussion
        3. Key Decisions
        4. Explicit Action Items
        5. Open Questions
        """,

        "Educational Lecture": """
        1. Lecture Overview
        2. Main Topics
        3. Important Concepts
        4. Examples and Explanations
        5. Key Learnings
        6. Conclusion
        """,

        "Tutorial": """
        1. Tutorial Overview
        2. Required Tools
        3. Step-by-Step Instructions
        4. Important Notes
        5. Expected Result
        """,

        "Interview": """
        1. Interview Overview
        2. Main Questions
        3. Important Answers
        4. Key Insights
        5. Conclusion
        """,

        "Podcast": """
        1. Podcast Overview
        2. Main Discussion Topics
        3. Speaker Insights
        4. Important Opinions
        5. Key Takeaways
        """,

        "News": """
        1. News Overview
        2. Main Events
        3. Important Details
        4. People and Organizations Mentioned
        5. Conclusion
        """,

        "Recipe": """
        1. Recipe Overview
        2. Ingredients
        3. Cooking Instructions
        4. Cooking Tips
        5. Final Result
        """,

        "Entertainment": """
        1. Video Overview
        2. Main Events
        3. Important Moments
        4. Key Takeaways
        """,

        "Other": """
        1. Overview
        2. Main Topics
        3. Important Information
        4. Conclusion
        """
    }

    sections = section_map[content_type]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3500,
        chunk_overlap=250
    )

    chunks = splitter.split_text(transcript)

    print("Total summary chunks:", len(chunks))

    summaries = []

    for i, chunk in enumerate(chunks, 1):

        print(f"Analyzing section {i}/{len(chunks)}...")

        result = invoke_llm(
            """
            You are an accurate video transcript analyst.

            Rules:
            - Use only transcript evidence.
            - Never invent facts, numbers or examples.
            - Preserve important technical details.
            - Do not turn explanations into action items.
            - Clearly mark uncertain information.
            - Respond in the same language as the transcript.
            """,

            """
            Content type: {content_type}

            Analyze this transcript section:

            {text}
            """,

            {
                "content_type": content_type,
                "text": chunk
            }
        )

        summaries.append(result)

    combined = "\n\n".join(summaries)

    final_summary = invoke_llm(
        """
        You are an expert editor.

        Combine the supplied section analyses into
        one accurate and non-repetitive report.

        Do not add facts that are not present.
        Do not invent conclusions.
        """,

        """
        Content type: {content_type}

        Required output sections:
        {sections}

        Section analyses:
        {text}
        """,

        {
            "content_type": content_type,
            "sections": sections,
            "text": combined[:12000]
        }
    )

    return final_summary


# ==========================================
# TITLE GENERATION
# ==========================================

def generate_title(transcript):

    if not transcript or len(transcript.split()) < 10:
        return "Title unavailable - Insufficient transcript"

    return invoke_llm(
        """
        Generate one concise, meaningful video title.

        Rules:
        - Use only transcript information.
        - Do not invent names or topics.
        - Preserve technical terminology.
        - Return only the title.
        """,

        "Generate a title for this transcript:\n\n{text}",

        {"text": transcript[:4000]}
    )


# ==========================================
# ACTION ITEMS
# ==========================================

def extract_action_items(transcript):

    return invoke_llm(
        """
        Extract explicitly assigned tasks only.

        Rules:
        - Only extract actual commitments or assignments.
        - Do not convert lecture examples into tasks.
        - Do not invent owners or deadlines.
        - If no tasks exist, return:
          No explicit action items found.
        """,

        "Transcript:\n\n{text}",

        {"text": transcript[:6000]}
    )


# ==========================================
# KEY DECISIONS
# ==========================================

def extract_decisions(transcript):

    return invoke_llm(
        """
        Extract only explicit decisions or conclusions.

        Rules:
        - Do not convert mathematical examples into decisions.
        - Do not invent decisions.
        - Preserve important technical findings.
        - If none exist, say so clearly.
        """,

        "Transcript:\n\n{text}",

        {"text": transcript[:6000]}
    )


# ==========================================
# OPEN QUESTIONS
# ==========================================

def extract_questions(transcript):

    return invoke_llm(
        """
        Extract explicitly unanswered questions only.

        Rules:
        - Do not invent questions.
        - Do not convert every topic into an open question.
        - If no unanswered questions exist, say so.
        """,

        "Transcript:\n\n{text}",

        {"text": transcript[:6000]}
    )
