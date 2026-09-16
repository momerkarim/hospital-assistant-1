import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

# ── Config ────────────────────────────────────────────────────────────────
FAISS_INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 4  # number of chunks retrieved per question

st.set_page_config(
    page_title="Hospital Policy Assistant",
    page_icon="🏥",
    layout="centered",
)

# ── Minimal, clean styling ───────────────────────────────────────────────
st.markdown("""
    <style>
        .main { padding-top: 1.5rem; }
        .source-tag {
            display: inline-block;
            background-color: #eef2f7;
            color: #1f2937;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.8rem;
            margin: 3px 6px 3px 0;
            border: 1px solid #d1d9e6;
        }
        .answer-box {
            background-color: #f8fafc;
            border-left: 4px solid #2563eb;
            padding: 1rem 1.25rem;
            border-radius: 6px;
            margin-top: 0.5rem;
        }
        .stTextInput > div > div > input {
            padding: 0.6rem;
        }
    </style>
""", unsafe_allow_html=True)


# ── Load resources (cached so they only load once per session) ──────────
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


@st.cache_resource(show_spinner=False)
def load_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
    if not api_key:
        st.error(
            "GROQ_API_KEY not found. Add it to your Streamlit secrets "
            "(Settings → Secrets) as: GROQ_API_KEY = \"your-key-here\""
        )
        st.stop()
    return Groq(api_key=api_key)


def build_context(retrieved_docs):
    """Combine retrieved chunks into a single context block for the LLM prompt."""
    context_parts = []
    for i, doc in enumerate(retrieved_docs, start=1):
        source = doc.metadata.get("source", "unknown")
        context_parts.append(f"[Excerpt {i} — Source: {source}]\n{doc.page_content}")
    return "\n\n".join(context_parts)


def generate_answer(client, question, context):
    system_prompt = (
        "You are a knowledgeable assistant for hospital staff, answering questions "
        "strictly based on the hospital's official policy and guideline documents "
        "provided in the context below. "
        "Answer clearly and professionally. "
        "If the context does not contain enough information to answer confidently, "
        "say so explicitly rather than guessing or using outside knowledge."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ── UI ────────────────────────────────────────────────────────────────────
st.title("🏥 Hospital Policy Assistant")
st.caption("Ask a question about hospital guidelines, policies, or procedures.")

vectorstore = load_vectorstore()
client = load_groq_client()

question = st.text_input(
    "Your question",
    placeholder="e.g., What is the protocol for reporting a needlestick injury?",
    label_visibility="collapsed",
)

ask_clicked = st.button("Ask", type="primary", use_container_width=False)

if ask_clicked and question.strip():
    with st.spinner("Searching documents and generating answer..."):
        retrieved_docs = vectorstore.similarity_search(question, k=TOP_K)

        if not retrieved_docs:
            st.warning("No relevant information found in the knowledge base.")
        else:
            context = build_context(retrieved_docs)
            answer = generate_answer(client, question, context)

            st.markdown("### Answer")
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

            st.markdown("### Sources")
            seen_sources = set()
            for doc in retrieved_docs:
                source = doc.metadata.get("source", "unknown")
                department = doc.metadata.get("department", "")
                tag_label = f"{source}" + (f" · {department}" if department else "")
                if tag_label not in seen_sources:
                    st.markdown(
                        f'<span class="source-tag">📄 {tag_label}</span>',
                        unsafe_allow_html=True,
                    )
                    seen_sources.add(tag_label)

            with st.expander("View retrieved excerpts"):
                for i, doc in enumerate(retrieved_docs, start=1):
                    st.markdown(f"**Excerpt {i} — {doc.metadata.get('source', 'unknown')}**")
                    st.write(doc.page_content)
                    st.divider()

elif ask_clicked and not question.strip():
    st.warning("Please enter a question.")
