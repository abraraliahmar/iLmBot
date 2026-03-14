from langchain_huggingface import HuggingFaceEmbeddings
from config.config import EMBEDDING_MODEL

_embeddings_instance = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a singleton HuggingFaceEmbeddings instance.

    The model is downloaded once on first call and reused thereafter.
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        try:
            _embeddings_instance = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load embedding model '{EMBEDDING_MODEL}': {exc}"
            ) from exc
    return _embeddings_instance
