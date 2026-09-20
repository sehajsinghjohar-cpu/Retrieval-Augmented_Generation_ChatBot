"""Load FDP documents from data/, chunk them, embed them, and save a FAISS index."""

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    INDEX_DIR,
    get_embeddings,
)

LOADERS = {
    ".pdf": lambda p: PyPDFLoader(str(p)),
    ".docx": lambda p: Docx2txtLoader(str(p)),
    ".txt": lambda p: TextLoader(str(p), encoding="utf-8"),
    ".md": lambda p: TextLoader(str(p), encoding="utf-8"),
}


def load_documents():
    docs = []
    for path in sorted(DATA_DIR.rglob("*")):
        loader_factory = LOADERS.get(path.suffix.lower())
        if loader_factory is None:
            continue
        loaded = loader_factory(path).load()
        for d in loaded:
            d.metadata["source"] = path.name  # clean filename for citations
        docs.extend(loaded)
        print(f"Loaded {path.name} ({len(loaded)} part(s))")
    return docs


def main():
    docs = load_documents()
    if not docs:
        raise SystemExit(
            f"No supported files found in '{DATA_DIR}/'. "
            "Add .pdf, .docx, .txt or .md files and try again."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks. Building index...")

    store = FAISS.from_documents(chunks, get_embeddings())
    INDEX_DIR.mkdir(exist_ok=True)
    store.save_local(str(INDEX_DIR))
    print(f"Index saved to '{INDEX_DIR}/'.")


if __name__ == "__main__":
    main()
