"""Shared RAG logic: config, retriever loading, and answer generation."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ---------------------------------------------------------------- config
DATA_DIR = Path("data")
INDEX_DIR = Path("index")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"  # Llama 3.3 on Groq

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4

SYSTEM_PROMPT = """You are the assistant for a Faculty Development Programme (FDP) on AI.
Answer the question using ONLY the context below.
If the answer is not in the context, say you don't have that information
instead of guessing. Keep answers clear and concise.

Context:
{context}"""

prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "{question}")]
)


# ---------------------------------------------------------------- helpers
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_retriever():
    """Load the saved FAISS index and return a retriever."""
    if not (INDEX_DIR / "index.faiss").exists():
        raise FileNotFoundError(
            "FAISS index not found. Run `python ingest.py` first."
        )
    # allow_dangerous_deserialization is safe here: we only load an index
    # that we built ourselves with ingest.py.
    store = FAISS.load_local(
        str(INDEX_DIR), get_embeddings(), allow_dangerous_deserialization=True
    )
    return store.as_retriever(search_kwargs={"k": TOP_K})


def get_llm() -> ChatGroq:
    if not os.getenv("GROQ_API_KEY"):
        raise EnvironmentError("GROQ_API_KEY is not set. Add it to your .env file.")
    return ChatGroq(model=LLM_MODEL, temperature=0)


def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


def answer(question: str, retriever, llm):
    """Retrieve relevant chunks, then generate a grounded answer.

    Returns (answer_text, source_documents).
    """
    docs = retriever.invoke(question)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": format_docs(docs), "question": question})
    return response, docs
