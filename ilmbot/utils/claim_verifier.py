from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from models.llm import get_llm
from utils.rag import get_retriever, format_docs


# ── Prompt templates ─────────────────────────────────────────────────

_SUPPORT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Quranic scholar analysing whether a claim is supported by "
     "the Quran as translated by Maulana Abul Kalam Azad.\n\n"
     "Quranic passages:\n{context}\n\n"
     "Identify ANY verses from the passages above that could SUPPORT the "
     "following claim. Quote the Surah and Ayah numbers. "
     "If nothing supports the claim, say 'No supporting evidence found.'"),
    ("human", "Claim: {claim}"),
])

_CONTRADICT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Quranic scholar analysing whether a claim is contradicted or "
     "nuanced by the Quran as translated by Maulana Abul Kalam Azad.\n\n"
     "Quranic passages:\n{context}\n\n"
     "Identify ANY verses from the passages above that CONTRADICT, QUALIFY, "
     "or add important NUANCE to the following claim. Quote the Surah and "
     "Ayah numbers. If nothing contradicts the claim, say "
     "'No contradicting evidence found.'"),
    ("human", "Claim: {claim}"),
])

_VERDICT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are IlmBot's claim-verification engine. Based on the supporting "
     "and contradicting analyses below, produce a final verdict.\n\n"
     "Supporting analysis:\n{support}\n\n"
     "Contradicting analysis:\n{contradict}\n\n"
     "Choose EXACTLY one verdict from:\n"
     "• Supported\n"
     "• Contradicted\n"
     "• Partially supported — context matters\n"
     "• Not addressed in the Quran\n\n"
     "Then write 2-3 sentences explaining your reasoning. "
     "Start your response with the verdict on its own line."),
    ("human", "Claim: {claim}"),
])


# ── Main verify function ────────────────────────────────────────────

def verify_claim(
    claim: str,
    model_name: str | None = None,
) -> dict:
    """Run the full claim-verification pipeline.

    Args:
        claim: The statement to verify against the Quran.
        model_name: Optional Groq model override.

    Returns:
        Dict with keys: support, contradict, verdict, chunks.
    """
    # 1. Retrieve relevant chunks
    try:
        retriever = get_retriever()
        chunks = retriever.invoke(claim)
        context = format_docs(chunks)
    except Exception as exc:
        return {
            "support": "",
            "contradict": "",
            "verdict": f"Error retrieving Quranic passages: {exc}",
            "chunks": [],
        }

    llm = get_llm(model_name)

    # 2. Support chain
    try:
        support_chain = _SUPPORT_PROMPT | llm | StrOutputParser()
        support_text = support_chain.invoke({"context": context, "claim": claim})
    except Exception as exc:
        support_text = f"Error: {exc}"

    # 3. Contradiction chain
    try:
        contradict_chain = _CONTRADICT_PROMPT | llm | StrOutputParser()
        contradict_text = contradict_chain.invoke({"context": context, "claim": claim})
    except Exception as exc:
        contradict_text = f"Error: {exc}"

    # 4. Verdict chain
    try:
        verdict_chain = _VERDICT_PROMPT | llm | StrOutputParser()
        verdict_text = verdict_chain.invoke({
            "support": support_text,
            "contradict": contradict_text,
            "claim": claim,
        })
    except Exception as exc:
        verdict_text = f"Error generating verdict: {exc}"

    return {
        "support": support_text,
        "contradict": contradict_text,
        "verdict": verdict_text,
        "chunks": chunks,
    }
