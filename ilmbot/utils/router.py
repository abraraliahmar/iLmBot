from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.config import GROQ_API_KEY, DEFAULT_LLM_MODEL

_ROUTER_SYSTEM = """\
You are a query classifier for an Islamic knowledge chatbot called iLmBot.
iLmBot has access to two sources:
1. A FAISS vector index of the Quran (Maulana Abul Kalam Azad translation).
2. A live web search engine.

Given a user question, respond with EXACTLY one word — the routing strategy:
• rag   — The question can be answered from the Quran / Islamic theology.
• web   — The question is about current events, a specific person, or needs live data.
• both  — The question mixes Quranic content with real-world context (e.g. verifying
           an extremist claim, comparing a modern fatwa to scripture).
• direct — The message is a greeting, a meta question about the bot, or small talk.

Reply with ONLY the single lowercase word. No punctuation, no explanation.\
"""

_router_prompt = ChatPromptTemplate.from_messages(
    [("system", _ROUTER_SYSTEM), ("human", "{question}")]
)


def route_query(question: str, model_name: str | None = None) -> str:
    """Classify *question* into a routing strategy.

    Args:
        question: The user's raw query.
        model_name: Optional Groq model override.

    Returns:
        One of 'rag', 'web', 'both', 'direct'.
    """
    if not GROQ_API_KEY:
        # Cannot call the router without an API key — default to rag
        return "rag"

    try:
        llm = ChatGroq(
            model=model_name or DEFAULT_LLM_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0,
            max_tokens=10,
        )
        chain = _router_prompt | llm | StrOutputParser()
        result = chain.invoke({"question": question}).strip().lower()

        # Sanitise: accept only valid strategies
        if result in {"rag", "web", "both", "direct"}:
            return result
        # Fuzzy fallback
        if "rag" in result:
            return "rag"
        if "web" in result:
            return "web"
        if "both" in result:
            return "both"
        return "rag"  # safe default
    except Exception:
        return "rag"
