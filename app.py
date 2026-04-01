"""
Medical RAG Bot - Streamlit Interface
Interactive web UI for asking questions about medical documents.

Run with: streamlit run app.py
"""

import streamlit as st
import sys
import os
import re
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rag.pipeline_factory import load_rag_pipeline
from src.pdf_loader import load_medical_pdfs
from src.chunking import chunk_documents
from src.loaders import PDFUploadHandler
from src.utils.config import VECTOR_STORE_PATH, MEDICAL_PDFS_DIR

# Constants
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = ['pdf']

# Page config
st.set_page_config(
    page_title="Medical RAG Bot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- STYLING ----------
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 25px;
    background: linear-gradient(135deg, #2563eb, #1e3a8a);
    color: white;
    border-radius: 10px;
    margin-bottom: 20px;
}
.answer-box {
    background-color: #e8f5e9;
    padding: 15px;
    border-radius: 5px;
    border-left: 4px solid #4caf50;
}
.source-box {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 5px;
    border-left: 4px solid #667eea;
}
</style>
""", unsafe_allow_html=True)


# ---------- HELPERS ----------
def vector_store_exists():
    index_file = os.path.join(VECTOR_STORE_PATH.replace("/faiss_index", ""), "faiss_index.faiss")
    pkl_file = os.path.join(VECTOR_STORE_PATH.replace("/faiss_index", ""), "faiss_index.pkl")
    return os.path.exists(index_file) and os.path.exists(pkl_file)


@st.cache_resource
def load_pipeline():
    if not vector_store_exists():
        return None
    try:
        return load_rag_pipeline(VECTOR_STORE_PATH)
    except Exception as e:
        st.error(f"Error loading pipeline: {e}")
        return None


def validate_uploaded_file(uploaded_file):
    if uploaded_file.size > MAX_FILE_SIZE:
        return False, "File too large"
    if not uploaded_file.name.lower().endswith('.pdf'):
        return False, "Only PDF allowed"
    safe_name = re.sub(r'[^\w\.\-\s]', '', uploaded_file.name)
    return True, safe_name


def sanitize_query(query):
    query = re.sub(r'[^\w\s\.\?\!\-\'\"]', '', query)
    return query.strip()[:500]


def get_document_stats():
    handler = PDFUploadHandler()
    pdfs = handler.list_uploaded_pdfs()
    return {
        "count": len(pdfs),
        "total_size_mb": handler.get_total_size_mb(),
        "pdfs": pdfs
    }


# ---------- MAIN ----------
def main():
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Medical RAG Bot</h1>
        <p>Ask questions from your medical PDFs</p>
    </div>
    """, unsafe_allow_html=True)

    if not vector_store_exists():
        st.error("❌ Vector store not found. Run indexing first.")
        st.code("python scripts/index_medical_documents.py")
        return

    pipeline = load_pipeline()
    if pipeline is None:
        st.stop()

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to:",
        ["Ask Questions", "Documents", "Upload PDF", "About"]
    )

    # ---------- ASK ----------
    if page == "Ask Questions":
        st.header("Ask Your Medical Question")

        query = st.text_area("Enter question")
        include_sources = st.checkbox("Show sources", True)
        num_sources = st.slider("Sources", 1, 10, 5)

        if st.button("Get Answer"):
            query = sanitize_query(query)

            if not query:
                st.warning("Enter valid query")
                return

            with st.spinner("Thinking..."):
                if include_sources:
                    answer, sources = pipeline.answer_query_with_context(query, top_k=num_sources)

                    st.markdown("### Answer")
                    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

                    st.markdown("### Sources")
                    for i, doc in enumerate(sources, 1):
                        with st.expander(f"Source {i}"):
                            st.markdown(f'<div class="source-box">{doc.page_content}</div>', unsafe_allow_html=True)
                else:
                    answer = pipeline.answer_query(query)
                    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

    # ---------- DOCUMENTS ----------
    elif page == "Documents":
        st.header("Documents")

        stats = get_document_stats()
        st.metric("Total PDFs", stats["count"])
        st.metric("Total Size", f"{stats['total_size_mb']} MB")

        if stats["pdfs"]:
            for i, pdf in enumerate(stats["pdfs"], 1):
                col1, col2, col3 = st.columns([3, 1, 1])
                col1.write(pdf["filename"])
                col2.write(f"{pdf['size_mb']} MB")

                if col3.button("Delete", key=i):
                    handler = PDFUploadHandler()
                    handler.delete_pdf(pdf["filename"])
                    st.rerun()
        else:
            st.info("No documents uploaded")

    # ---------- UPLOAD ----------
    elif page == "Upload PDF":
        st.header("Upload PDF")

        uploaded_file = st.file_uploader("Upload PDF", type="pdf")

        if uploaded_file:
            valid, name = validate_uploaded_file(uploaded_file)

            if not valid:
                st.error(name)
                return

            if st.button("Upload"):
                path = os.path.join(MEDICAL_PDFS_DIR, name)
                with open(path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.success("Uploaded successfully!")
                st.info("Run indexing script next.")

    # ---------- ABOUT ----------
    elif page == "About":
        st.header("About")

        st.markdown("""
        **Medical RAG Bot** lets you:
        - Upload medical PDFs
        - Ask questions
        - Get answers with sources
        
        **Steps:**
        1. Upload PDFs  
        2. Run indexing  
        3. Ask questions  
        """)


if __name__ == "__main__":
    main()