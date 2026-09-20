# FDP RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built for the two-day **Faculty Development Programme (FDP) on AI** at **Sri Guru Gobind Singh College of Commerce (SGGSCC), University of Delhi**, organised in collaboration with **Rabbitt AI**.

The chatbot answers questions grounded in the FDP's own documents (schedule, modules, notices, brochure content), and was demoed live to attendees during the programme.

---

## Why RAG?

A plain LLM doesn't know anything about a specific event. RAG fixes this by retrieving relevant chunks from your documents at query time and handing them to the model as context, so answers stay grounded in the source material instead of being guessed.

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | Llama 3.3 (served via Groq) |
| Orchestration | LangChain |
| Vector store | FAISS |
| Embeddings | `<your embedding model>` |
| Interface | `<Streamlit / CLI / notebook>` |
| Language | Python 3.10+ |

## How It Works

```
FDP documents ──► Load & chunk ──► Embed ──► FAISS index
                                                 │
User question ──► Embed ──► Similarity search ◄──┘
                                 │
                     Top-k relevant chunks
                                 │
                   Prompt = question + context
                                 │
                        Llama 3.3 (Groq)
                                 │
                         Grounded answer
```

1. **Ingest**: FDP documents are loaded and split into overlapping chunks.
2. **Index**: Each chunk is embedded and stored in a FAISS index.
3. **Retrieve**: The user's question is embedded and the most similar chunks are fetched.
4. **Generate**: The chunks and the question are sent to Llama 3.3 on Groq, which produces the answer.

## Project Structure

```
.
├── data/              # FDP source documents
├── app.py             # Chatbot entry point
├── ingest.py          # Chunking + FAISS index building
├── requirements.txt
├── .env.example
└── README.md
```

> Adjust file names to match your repo.

## Setup

```bash
git clone https://github.com/sehajsinghjohar-cpu/<repo-name>.git
cd <repo-name>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

```bash
# 1. Build the vector index from documents in data/
python ingest.py

# 2. Start the chatbot
python app.py            # or: streamlit run app.py
```

Example questions:

- "What topics are covered on Day 1?"
- "Who are the resource persons for the AI modules?"
- "How do I submit feedback for the programme?"

## Limitations

- Answers are only as good as the documents indexed; anything outside them may get a weak or "I don't know" response.
- Retrieval quality depends on chunk size, overlap, and `k`; tune these for your corpus.
- Uses a hosted LLM API, so it needs an internet connection and a valid Groq key.

## Possible Extensions

- Source citations with each answer
- Conversation memory across turns
- Re-ranking retrieved chunks
- Reuse as a live demo for the SGGSCC admin staff AI training

## Acknowledgements

- **Rabbitt AI** (Harneet Singh) for the FDP collaboration
- **Dr. Bimaldeep Kaur**, T&P Cell, SGGSCC
- Groq, Meta (Llama), LangChain, and FAISS

## Author

**Sehaj**: BS in Data Science and Applied AI, IIT Jodhpur
GitHub: [sehajsinghjohar-cpu](https://github.com/sehajsinghjohar-cpu)
