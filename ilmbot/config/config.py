"""
config/config.py
────────────────
Central configuration for IlmBot.
All secrets are loaded from environment variables (never hardcoded).
"""

import os
from dotenv import load_dotenv

# ── Load .env (local dev) or Streamlit secrets (cloud) ──────────────
load_dotenv()


def _get_key(name: str) -> str:
    """Return an env-var value, falling back to Streamlit secrets."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st
        value = st.secrets.get(name)
    except Exception:
        pass
    return value or ""


# ── API keys ────────────────────────────────────────────────────────
GROQ_API_KEY: str = _get_key("GROQ_API_KEY")
TAVILY_API_KEY: str = _get_key("TAVILY_API_KEY")

# ── Model settings ──────────────────────────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL: str = "llama-3.1-8b-instant"
AVAILABLE_MODELS: list[str] = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
]

# ── RAG settings ────────────────────────────────────────────────────
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 100
TOP_K_RESULTS: int = 5

# ── Paths ───────────────────────────────────────────────────────────
FAISS_INDEX_PATH: str = "data/faiss_index"
PDF_PATHS: list[str] = [
    r"data/V1_Quran.pdf",
    r"data/V2_Quran.pdf",
    r"data/V3_Quran.pdf",
]
