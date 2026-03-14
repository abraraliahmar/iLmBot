import os
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
    FAISS_INDEX_PATH,
    PDF_PATHS,
)
from models.embeddings import get_embeddings
from models.llm import get_llm, get_rag_prompt

# ── Module-level singletons ─────────────────────────────────────────
_vectorstore = None
_retriever = None


# ── PDF loading ──────────────────────────────────────────────────────

def _load_pdf(path: str) -> list[Document]:
    """Extract text from a PDF page-by-page using PyMuPDF.

    Args:
        path: Filesystem path to the PDF.

    Returns:
        List of LangChain Document objects (one per page).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF not found: {path}")
    docs: list[Document] = []
    try:
        pdf = fitz.open(path)
        for page_num in range(len(pdf)):
            text = pdf[page_num].get_text()
            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": os.path.basename(path), "page": page_num + 1},
                    )
                )
        pdf.close()
    except Exception as exc:
        raise RuntimeError(f"Error reading PDF '{path}': {exc}") from exc
    return docs


def _load_all_pdfs() -> list[Document]:
    """Load all available PDF volumes from PDF_PATHS."""
    all_docs: list[Document] = []
    for path in PDF_PATHS:
        if os.path.exists(path):
            all_docs.extend(_load_pdf(path))
    if not all_docs:
        raise FileNotFoundError(
            f"No PDFs found. Expected at least one of: {PDF_PATHS}"
        )
    return all_docs


# ── Chunking ─────────────────────────────────────────────────────────

def _chunk_documents(docs: list[Document]) -> list[Document]:
    """Split documents into smaller overlapping chunks.

    Args:
        docs: Full-page documents from PDF extraction.

    Returns:
        Chunked Document list suitable for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


# ── Index build / load ───────────────────────────────────────────────

def build_index() -> FAISS:
    """Build a FAISS index from scratch and persist it to disk.

    Call this once (or whenever the PDF changes) to create the index.
    """
    global _vectorstore, _retriever
    print("Loading PDF(s)…")
    docs = _load_all_pdfs()
    print(f"  → {len(docs)} pages extracted.")

    print("Chunking…")
    chunks = _chunk_documents(docs)
    print(f"  → {len(chunks)} chunks created.")

    print("Building FAISS index…")
    embeddings = get_embeddings()
    _vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(os.path.dirname(FAISS_INDEX_PATH) or "data", exist_ok=True)
    _vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"  → Index saved to {FAISS_INDEX_PATH}/")

    _retriever = _vectorstore.as_retriever(search_kwargs={"k": TOP_K_RESULTS})
    return _vectorstore


def load_index() -> FAISS:
    """Load a pre-built FAISS index from disk, or build one if missing."""
    global _vectorstore, _retriever
    if _vectorstore is not None:
        return _vectorstore

    embeddings = get_embeddings()
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            _vectorstore = FAISS.load_local(
                FAISS_INDEX_PATH,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            _retriever = _vectorstore.as_retriever(
                search_kwargs={"k": TOP_K_RESULTS}
            )
            return _vectorstore
        except Exception as exc:
            print(f"Warning: could not load index, rebuilding. ({exc})")

    # Fallback: build from PDF
    return build_index()


# ── Retriever accessor ───────────────────────────────────────────────

def get_retriever():
    """Return the FAISS retriever, loading the index if needed."""
    global _retriever
    if _retriever is None:
        load_index()
    return _retriever


# ── Helper: format retrieved docs into a context string ──────────────

def format_docs(docs: list[Document]) -> str:
    """Concatenate retrieved chunks into a single context string."""
    parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Chunk {i} — {src}, p.{page}]\n{doc.page_content}")
    return "\n\n".join(parts)


# ── RAG chain (LCEL) ────────────────────────────────────────────────

def get_rag_chain(model_name: str | None = None, mode: str = "concise",
                  chat_history: list | None = None):
    """Build and return an LCEL RAG chain.

    Args:
        model_name: Groq model identifier.
        mode: 'concise' or 'detailed'.
        chat_history: List of LangChain message objects for multi-turn.

    Returns:
        An LCEL chain that accepts a question string and returns an answer string.
    """
    retriever = get_retriever()
    prompt = get_rag_prompt(mode)
    llm = get_llm(model_name)

    history = chat_history or []

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "chat_history": lambda _: history,
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def retrieve_chunks(query: str) -> list[Document]:
    """Run the retriever and return raw Document chunks (for the Sources panel)."""
    retriever = get_retriever()
    try:
        return retriever.invoke(query)
    except Exception:
        return []
