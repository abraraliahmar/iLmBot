import streamlit as st
from config.config import GROQ_API_KEY, AVAILABLE_MODELS, DEFAULT_LLM_MODEL
from models.llm import get_llm, get_direct_prompt, format_history
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── Page config 
st.set_page_config(
    page_title="iLmBot — Quranic Knowledge Assistant",
    page_icon="📖",
    layout="wide",
)

# ── Custom CSS 
st.markdown("""
<style>
    .stApp { }
    .block-container { max-width: 900px; }
    .verdict-box { padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar 
with st.sidebar:
    st.image(r"ilmbot\data\ilmpath.png", width=64)
    st.title("iLmBot")
    st.caption("Quranic knowledge · grounded in Maulana Azad's translation")
    st.divider()

    response_mode = st.radio(
        "Response mode",
        ["Concise", "Detailed"],
        index=0,
        help="Concise: 2-3 sentences + one Ayah. Detailed: full tafsir-style.",
    )
    mode = response_mode.lower()

    selected_model = st.selectbox(
        "LLM model",
        AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(DEFAULT_LLM_MODEL),
    )

    st.divider()
    if st.button("🗑️  Clear conversation", use_container_width=True):
        st.session_state.pop("messages", None)
        st.session_state.pop("sources", None)
        st.rerun()

    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "iLmBot retrieves relevant ayahs from a FAISS index of the Quran "
        "(Maulana Abul Kalam Azad's translation), and optionally searches "
        "the web, then answers your question with cited references."
    )

# ── Session state defaults 
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources" not in st.session_state:
    st.session_state.sources = {}  


# ── Cached loaders 
@st.cache_resource(show_spinner="Loading Quranic index…")
def _load_retriever():
    """Load (or build) the FAISS retriever once."""
    from utils.rag import load_index, get_retriever
    load_index()
    return get_retriever()


# ── Tabs 
tab_ask, tab_verify = st.tabs(["📖  Ask a Question", "✅  Verify a Claim"])

# TAB 1 — Ask a Question
with tab_ask:
    # Display existing chat
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Show sources expander if available
            src = st.session_state.sources.get(idx)
            if src:
                with st.expander("📚 Sources"):
                    if src.get("route"):
                        st.caption(f"**Routing:** {src['route']}")
                    if src.get("chunks"):
                        st.markdown("**Quranic passages retrieved:**")
                        for i, c in enumerate(src["chunks"], 1):
                            page = c.metadata.get("page", "?")
                            source_file = c.metadata.get("source", "")
                            st.markdown(
                                f"**[{i}]** *{source_file}, p.{page}*\n\n"
                                f"{c.page_content[:300]}{'…' if len(c.page_content) > 300 else ''}"
                            )
                    if src.get("web"):
                        st.markdown("**Web results:**")
                        for w in src["web"]:
                            st.markdown(f"- [{w['title']}]({w['url']}): {w['snippet'][:150]}")
    # Welcome section — shown only when no messages exist
    if not st.session_state.messages:
        st.markdown("""
        <div style="
            max-width: 680px;
            margin: 2rem auto;
            padding: 2rem;
            text-align: center;
            opacity: 0.7;
        ">
            <h2 style="font-weight: 400; margin-bottom: 0.5rem;">Welcome to iLmBot</h2>
            <p style="font-size: 0.95rem; line-height: 1.7; color: gray;">
                In a world where religious misinformation spreads quickly — from extremist 
                interpretations to casual Islamophobia — it's hard to know what the Quran 
                <em>actually</em> says.<br><br>
                iLmBot was built to fix that. It lets you ask questions, challenge claims, 
                and explore Quranic teachings grounded in the scholarly, progressive 
                translation of <strong>Maulana Abul Kalam Azad</strong> — one of the most 
                nuanced and non-extremist translations ever produced.<br><br>
                Whether you're a Muslim seeking clarity, a non-Muslim with genuine curiosity, 
                or someone who just heard a claim and wants to verify it — ask away.
            </p>
            <div style="
                display: flex;
                justify-content: center;
                gap: 0.8rem;
                margin-top: 1.5rem;
                flex-wrap: wrap;
            ">
                <span style="
                    background: rgba(128,128,128,0.1);
                    padding: 0.4rem 0.9rem;
                    border-radius: 1rem;
                    font-size: 0.82rem;
                    color: gray;
                ">💬 "What does the Quran say about patience?"</span>
                <span style="
                    background: rgba(128,128,128,0.1);
                    padding: 0.4rem 0.9rem;
                    border-radius: 1rem;
                    font-size: 0.82rem;
                    color: gray;
                ">🔍 "Does Islam oppress women?"</span>
                <span style="
                    background: rgba(128,128,128,0.1);
                    padding: 0.4rem 0.9rem;
                    border-radius: 1rem;
                    font-size: 0.82rem;
                    color: gray;
                ">✅ Try the "Verify a Claim" tab</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Chat input
    if user_input := st.chat_input("Ask about the Quran…"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Process
        with st.chat_message("assistant"):
            with st.spinner("Retrieving from Quran…"):
                try:
                    # ── Route 
                    from utils.router import route_query
                    strategy = route_query(user_input, model_name=selected_model)

                    source_info: dict = {"route": strategy, "chunks": [], "web": []}
                    answer = ""

                    chat_history = format_history(st.session_state.messages[:-1])

                    # ── RAG path 
                    if strategy in ("rag", "both"):
                        from utils.rag import get_rag_chain, retrieve_chunks
                        chunks = retrieve_chunks(user_input)
                        source_info["chunks"] = chunks

                        rag_chain = get_rag_chain(
                            model_name=selected_model,
                            mode=mode,
                            chat_history=chat_history,
                        )
                        answer = rag_chain.invoke(user_input)

                    # ── Web path 
                    if strategy in ("web", "both"):
                        from utils.search import web_search
                        web_results = web_search(user_input)
                        source_info["web"] = web_results

                        if strategy == "web":
                            # Build answer from web results via LLM
                            web_context = "\n".join(
                                f"- {r['title']}: {r['snippet']}" for r in web_results
                            )
                            from models.llm import get_rag_prompt
                            prompt = get_rag_prompt(mode)
                            llm = get_llm(selected_model)
                            web_chain = (
                                {
                                    "context": lambda _: web_context,
                                    "question": RunnablePassthrough(),
                                    "chat_history": lambda _: chat_history,
                                }
                                | prompt
                                | llm
                                | StrOutputParser()
                            )
                            answer = web_chain.invoke(user_input)
                        elif strategy == "both" and answer:
                            # Supplement existing RAG answer with web context
                            web_summary = "\n".join(
                                f"- {r['title']}: {r['snippet'][:120]}" for r in web_results
                            )
                            answer += (
                                f"\n\n---\n**Additional context from the web:**\n{web_summary}"
                            )

                    if strategy == "direct":
                        prompt = get_direct_prompt(mode)
                        llm = get_llm(selected_model)
                        direct_chain = (
                            {
                                "question": RunnablePassthrough(),
                                "chat_history": lambda _: chat_history,
                            }
                            | prompt
                            | llm
                            | StrOutputParser()
                        )
                        answer = direct_chain.invoke(user_input)

                    if not answer:
                        answer = "I wasn't able to generate a response. Please try rephrasing your question."

                    st.markdown(answer)

                    msg_idx = len(st.session_state.messages)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.sources[msg_idx] = source_info
                    with st.expander("📚 Sources"):
                        st.caption(f"**Routing:** {strategy}")
                        if source_info["chunks"]:
                            for i, c in enumerate(source_info["chunks"], 1):
                                page = c.metadata.get("page", "?")
                                sf = c.metadata.get("source", "")
                                st.markdown(
                                    f"**[{i}]** *{sf}, p.{page}*\n\n"
                                    f"{c.page_content[:300]}{'…' if len(c.page_content) > 300 else ''}"
                                )
                        if source_info["web"]:
                            st.markdown("**Web results:**")
                            for w in source_info["web"]:
                                st.markdown(f"- [{w['title']}]({w['url']})")

                except Exception as exc:
                    st.error(f"Something went wrong: {exc}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"⚠️ Error: {exc}"}
                    )


# TAB 2 — Verify a Claim
with tab_verify:
    st.subheader("Verify a Claim Against the Quran")
    st.markdown(
        "Paste any statement you've heard — from a religious leader, social media, "
        "or anywhere else — and iLmBot will check it against Maulana Azad's Quran "
        "translation."
    )

    claim_input = st.text_area(
        "Enter a claim to verify",
        placeholder="e.g. 'Islam says women cannot work'",
        height=100,
    )

    if st.button("🔍  Verify Claim", type="primary", use_container_width=True):
        if not claim_input.strip():
            st.warning("Please enter a claim first.")
        elif not GROQ_API_KEY:
            st.error("GROQ_API_KEY is not set. Add it to your .env or Streamlit secrets.")
        else:
            with st.spinner("Analysing claim against the Quran…"):
                try:
                    from utils.claim_verifier import verify_claim
                    result = verify_claim(claim_input.strip(), model_name=selected_model)

                    verdict_text = result["verdict"]
                    verdict_lower = verdict_text.lower()

                    if "supported" in verdict_lower and "partially" not in verdict_lower and "not" not in verdict_lower:
                        st.success(f"**Verdict:** {verdict_text}")
                    elif "contradicted" in verdict_lower:
                        st.error(f"**Verdict:** {verdict_text}")
                    elif "partially" in verdict_lower:
                        st.warning(f"**Verdict:** {verdict_text}")
                    else:
                        st.info(f"**Verdict:** {verdict_text}")

                    # ── Side-by-side evidence 
                    col_sup, col_con = st.columns(2)
                    with col_sup:
                        st.markdown("### 🟢 Quranic Support")
                        st.markdown(result["support"] or "*No supporting evidence found.*")
                    with col_con:
                        st.markdown("### 🔴 Quranic Contradiction")
                        st.markdown(result["contradict"] or "*No contradicting evidence found.*")

                    # ── Retrieved chunks 
                    if result["chunks"]:
                        with st.expander("📚 Retrieved Quranic passages"):
                            for i, c in enumerate(result["chunks"], 1):
                                page = c.metadata.get("page", "?")
                                sf = c.metadata.get("source", "")
                                st.markdown(
                                    f"**[{i}]** *{sf}, p.{page}*\n\n"
                                    f"{c.page_content[:400]}{'…' if len(c.page_content) > 400 else ''}"
                                )

                except Exception as exc:
                    st.error(f"Verification failed: {exc}")


# ── Footer 
st.divider()
st.caption(
    "iLmBot is an educational tool. It is not a substitute for scholarly study. "
    "All Quranic references are from Maulana Abul Kalam Azad's translation."
)
