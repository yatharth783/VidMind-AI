
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.vector_store import (
    build_vector_store,
    load_vector_store,
    get_retriever
)


# Initialize Ollama LLM
def get_llm():
    return ChatOllama(
        model="llama3.2",
        temperature=0.3,
        num_predict=2048
    )


# Format retrieved documents
def format_docs(docs):
    return "\n\n".join(
        [doc.page_content for doc in docs]
    )


# Build RAG chain for a new meeting transcript
def build_rag_chain(transcript: str):

    vector_store = build_vector_store(transcript)

    retriever = get_retriever(vector_store, k=4)

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant.

Answer the user's question based ONLY on the meeting
transcript context provided below.

If the answer is not found in the context, say:
"I could not find this information in the meeting transcript."

Always be concise and precise.
If quoting someone, mention it clearly.

Context from meeting transcript:
{context}"""
        ),
        ("human", "{question}")
    ])

    # Full LCEL RAG Pipeline
    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# Load existing RAG vector store
def load_rag_chain():

    vector_store = load_vector_store()

    retriever = get_retriever(vector_store)

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant.

Answer the user's question based ONLY on the meeting
transcript context provided below.

If the answer is not found in the context, say:
"I could not find this information in the meeting transcript."

Always be concise and precise.
If quoting someone, mention it clearly.

Context from meeting transcript:
{context}"""
        ),
        ("human", "{question}")
    ])

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# Ask questions from meeting transcript
def ask_question(rag_chain, question: str) -> str:

    print(f"\nQuestion: {question}")

    answer = rag_chain.invoke(question)

    print(f"\nAnswer: {answer}")

    return answer
