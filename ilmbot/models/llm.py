"""
models/llm.py
─────────────
ChatGroq LLM initialisation and prompt construction.
Supports concise / detailed response modes.
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from config.config import GROQ_API_KEY, DEFAULT_LLM_MODEL

# ── Base system instruction ─────────────────────────────────────────
_BASE_SYSTEM = (
    "You are IlmBot, an Islamic knowledge assistant grounded in the Quran as "
    "translated by Maulana Abul Kalam Azad. You help people understand what the "
    "Quran actually says — without political bias, without extremist interpretation, "
    "and without dismissing sincere questions. Always cite the Surah and Ayah number "
    "when referencing Quranic content. Be respectful to all users regardless of faith."
)

_CONCISE_ADDENDUM = (
    "\n\nRespond in 2-3 concise sentences. Include one Ayah reference "
    "(Surah name and number : Ayah number) that best supports your answer."
)

_DETAILED_ADDENDUM = (
    "\n\nProvide a full, detailed explanation that includes: "
    "historical context, Maulana Azad's commentary and perspective, "
    "multiple Ayah references (Surah:Ayah), and nuanced discussion of "
    "different scholarly viewpoints where relevant."
)


def get_llm(model_name: str | None = None) -> ChatGroq:
    """Return a ChatGroq instance for the given model.

    Args:
        model_name: Groq model identifier. Falls back to DEFAULT_LLM_MODEL.

    Returns:
        A ready-to-use ChatGroq object.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")
    try:
        return ChatGroq(
            model=model_name or DEFAULT_LLM_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0.3,
            max_tokens=2048,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to initialise ChatGroq: {exc}") from exc


def build_system_prompt(mode: str = "concise") -> str:
    """Compose the full system prompt for the selected response mode.

    Args:
        mode: 'concise' or 'detailed'.

    Returns:
        Complete system prompt string.
    """
    addendum = _DETAILED_ADDENDUM if mode == "detailed" else _CONCISE_ADDENDUM
    return _BASE_SYSTEM + addendum


def get_rag_prompt(mode: str = "concise") -> ChatPromptTemplate:
    """Return a ChatPromptTemplate wired for RAG (context + question).

    The prompt includes a placeholder for chat history so multi-turn
    context is preserved.
    """
    system = build_system_prompt(mode)
    system += (
        "\n\nUse the following Quranic passages retrieved from Maulana Abul Kalam "
        "Azad's translation to answer the question. If the passages do not contain "
        "enough information, say so honestly rather than guessing.\n\n"
        "Context:\n{context}"
    )
    return ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{question}"),
        ]
    )


def get_direct_prompt(mode: str = "concise") -> ChatPromptTemplate:
    """Prompt for direct (non-RAG) responses — greetings, meta questions."""
    system = build_system_prompt(mode)
    return ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{question}"),
        ]
    )


def format_history(history: list[dict]) -> list:
    """Convert session-state dicts to LangChain message objects.

    Args:
        history: List of {'role': 'user'|'assistant', 'content': str}.

    Returns:
        List of HumanMessage / AIMessage objects.
    """
    msgs = []
    for msg in history:
        if msg["role"] == "user":
            msgs.append(HumanMessage(content=msg["content"]))
        else:
            msgs.append(AIMessage(content=msg["content"]))
    return msgs
