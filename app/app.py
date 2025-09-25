import os
import streamlit as st
from dotenv import load_dotenv

try:
    from app.rag_pipeline import RAGPipeline  # when running from project root
except ModuleNotFoundError:
    from rag_pipeline import RAGPipeline  # when running from within app/

load_dotenv()

st.set_page_config(page_title="RAG over PDFs", page_icon="📄", layout="wide")

if not os.getenv("GOOGLE_API_KEY"):
    st.warning("Set GOOGLE_API_KEY in your environment or .env file to use Google.")

st.title("📄 RAG over PDFs (Google)")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top K", 1, 10, 3, key="top_k_slider")
    embed_model = st.text_input(
        "Embedding model", value="models/gemini-embedding-001", key="embed_model_input"
    )
    chat_model = st.text_input(
        "Chat model", value="gemini-2.5-flash", key="chat_model_input"
    )

    st.divider()
    st.subheader("Upload PDFs")
    upload_dir = st.text_input(
        "Upload save folder", value="data/uploads", key="upload_dir_input"
    )
    uploaded_files = st.file_uploader(
        "Select one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )
    upload_index_btn = st.button("Add files", key="upload_index_btn")

# Keep a single pipeline instance across interactions via session state
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline(
        model_name=chat_model, embedding_model=embed_model
    )

# Recreate pipeline when settings change
if st.session_state.get("_last_models") != (embed_model, chat_model):
    st.session_state.pipeline = RAGPipeline(
        model_name=chat_model, embedding_model=embed_model
    )
    st.session_state["_last_models"] = (embed_model, chat_model)

if upload_index_btn:
    if uploaded_files:
        os.makedirs(upload_dir, exist_ok=True)
        saved = 0
        for f in uploaded_files:
            dest_path = os.path.join(upload_dir, f.name)
            with open(dest_path, "wb") as out:
                out.write(f.getbuffer())
            saved += 1
        with st.spinner(f"Saved {saved} file(s). Indexing..."):
            result = st.session_state.pipeline.index_data(upload_dir)
            st.success(
                f"Indexed {result['added']} chunks. Collection now has {result['collection_count']} documents."
            )
    else:
        st.info("No files selected.")

st.subheader("Ask a question")
query = st.text_input("Your question", key="qa_question")
ask = st.button("Ask", key="ask_btn")

if ask and query:
    with st.spinner("Retrieving and generating answer..."):
        output = st.session_state.pipeline.answer(query, top_k=top_k)
        st.markdown("**Answer**")
        st.write(output["answer"]) if isinstance(output, dict) else st.write(output)
        if isinstance(output, dict) and output.get("sources"):
            st.markdown("**Sources**")
            for s in output["sources"]:
                st.markdown(
                    f"- {s.get('source', 'unknown')} (score: {s.get('similarity', 0):.3f})"
                )
                # Show content preview if available
                if s.get("content"):
                    st.markdown(f"  *Preview: {s['content'][:100]}...*")
