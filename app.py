"""
Medical RAG Bot - Streamlit Interface
Interactive web UI for asking questions about medical documents.

Run with: streamlit run app.py
"""

import streamlit as st
import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rag.pipeline_factory import load_rag_pipeline, create_rag_pipeline
from src.pdf_loader import load_medical_pdfs
from src.chunking import chunk_documents
from src.loaders import PDFUploadHandler
from src.utils.config import VECTOR_STORE_PATH, MEDICAL_PDFS_DIR


# Page configuration
st.set_page_config(
    page_title="Medical RAG Bot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with improved styling
st.markdown("""
    <style>
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main-header {
        text-align: center;
        padding: 30px 20px;
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 50%, #1e3a8a 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5em;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        margin: 10px 0 0 0;
        font-size: 1.2em;
        opacity: 0.95;
    }
    
    .stMarkdown {
        text-align: justify;
    }
    
    .source-box {
        background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
        padding: 16px;
        border-radius: 8px;
        border-left: 5px solid #3b82f6;
        margin: 12px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .answer-box {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        padding: 16px;
        border-radius: 8px;
        border-left: 5px solid #10b981;
        margin: 12px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .query-box {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        padding: 16px;
        border-radius: 8px;
        border-left: 5px solid #0ea5e9;
        margin: 12px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #f0f1f3 100%);
        padding: 16px;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        text-align: center;
    }
    
    .file-item {
        background-color: #f8f9fa;
        padding: 12px;
        border-radius: 6px;
        margin: 8px 0;
        border-left: 3px solid #3b82f6;
    }
    
    .section-header {
        padding-bottom: 10px;
        border-bottom: 2px solid #3b82f6;
        margin-bottom: 20px;
    }
    
    .stButton > button {
        border-radius: 6px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    .stTextArea > label {
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)


def vector_store_exists():
    """Check if vector store exists."""
    # Check for the FAISS index file (without extension) and metadata
    index_file = os.path.join(VECTOR_STORE_PATH.replace("/faiss_index", ""), "faiss_index.faiss")
    pkl_file = os.path.join(VECTOR_STORE_PATH.replace("/faiss_index", ""), "faiss_index.pkl")
    
    return os.path.exists(index_file) and os.path.exists(pkl_file)


@st.cache_resource
def load_pipeline():
    """Load RAG pipeline (cached only if vector store exists)."""
    if not vector_store_exists():
        return None
    try:
        return load_rag_pipeline(VECTOR_STORE_PATH)
    except (FileNotFoundError, RuntimeError, Exception) as e:
        st.error(f"Error loading pipeline: {str(e)}")
        return None


def get_document_stats():
    """Get statistics about uploaded documents."""
    handler = PDFUploadHandler()
    pdfs = handler.list_uploaded_pdfs()
    return {
        "count": len(pdfs),
        "total_size_mb": handler.get_total_size_mb(),
        "pdfs": pdfs
    }


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🏥 Medical Information RAG Bot</h1>
            <p>Intelligent Q&A System for Medical Documents</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Check if vector store exists - don't use cache for this check
    has_vector_store = vector_store_exists()
    
    if not has_vector_store:
        # Show setup page
        st.warning("⚠️ Vector store not initialized", icon="⚠️")
        st.markdown("""
            ### 🚀 Getting Started
            
            Before you can ask questions, you need to index your medical documents.
            
            **Follow these steps:**
            
            #### 1️⃣ Add Medical PDFs
            Upload your medical PDF files to the `data/medical_pdfs/` directory or use the CLI:
            ```bash
            cp your_medical_documents.pdf data/medical_pdfs/
            ```
            
            #### 2️⃣ Index Documents
            Run the indexing script in your terminal:
            ```bash
            python scripts/index_medical_documents.py
            ```
            
            #### 3️⃣ Refresh the App
            Once indexing is complete, refresh this page in your browser (F5 or Cmd+R). You'll then be able to ask questions!
            
            ---
            
            ### 📊 About Vector Store
            - **Location:** `data/vector_store/faiss_index.faiss`
            - **Purpose:** Stores document embeddings for fast semantic search
            - **Status:** ❌ Not initialized
            
            ### 💡 What's Next?
            After indexing:
            - 🔍 Use the **Ask Questions** page to query your documents
            - 📚 View indexed documents in the **Documents** page
            - 📤 Upload new PDFs in the **Upload PDF** page
        """)
        
        # Add a refresh button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Check for Vector Store", use_container_width=True):
                st.rerun()
        return
    
    # Sidebar navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["🔍 Ask Questions", "📚 Documents", "📤 Upload PDF", "ℹ️ About"],
        help="Navigate between different features"
    )
    
    # Add sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 System Info")
    stats = get_document_stats()
    st.sidebar.metric("Documents", stats["count"])
    st.sidebar.metric("Total Size", f"{stats['total_size_mb']} MB")
    st.sidebar.markdown("### ✅ Vector Store")
    st.sidebar.success("Initialized & Ready", icon="✅")
    
    # Load pipeline
    pipeline = load_pipeline()
    
    if pipeline is None:
        st.error("❌ Error loading pipeline. Please check your vector store or re-index documents.")
        st.stop()
    
    # PAGE 1: Ask Questions
    if page == "🔍 Ask Questions":
        st.markdown('<div class="section-header"><h2>🔍 Ask Your Medical Questions</h2></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_area(
                "💬 Enter your medical question:",
                placeholder="e.g., What is diabetes? How is chlamydia transmitted?",
                height=120,
                help="Type your medical question here for an intelligent answer"
            )
        
        with col2:
            st.markdown("### ⚙️ Options")
            include_sources = st.checkbox("📄 Show sources", value=True, help="Display source documents")
            num_sources = st.slider("# Sources:", 1, 10, 5, help="Number of relevant documents to retrieve")
            confidence = st.slider("Confidence:", 0.0, 1.0, 0.7, step=0.1)
        
        if st.button("🚀 Get Answer", use_container_width=True, type="primary"):
            if not query.strip():
                st.warning("⚠️ Please enter a question.", icon="⚠️")
            else:
                with st.spinner("⏳ Processing your question..."):
                    try:
                        if include_sources:
                            answer, sources = pipeline.answer_query_with_context(
                                query, 
                                top_k=num_sources
                            )
                            
                            # Display answer with better formatting
                            st.markdown("### 📖 Answer")
                            st.markdown(
                                f'<div class="answer-box"><p>{answer}</p></div>',
                                unsafe_allow_html=True
                            )
                            
                            # Display sources
                            st.markdown("### 📚 Source Documents")
                            st.info(f"✓ Retrieved {len(sources)} relevant document(s)", icon="ℹ️")
                            
                            for i, doc in enumerate(sources, 1):
                                with st.expander(f"📄 Source {i} - {doc.metadata.get('source', 'Unknown') if doc.metadata else 'Unknown'}"):
                                    col_meta, col_icon = st.columns([0.9, 0.1])
                                    
                                    with col_meta:
                                        if doc.metadata:
                                            st.caption(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                                            if 'page' in doc.metadata:
                                                st.caption(f"**Page:** {doc.metadata.get('page')}")
                                    
                                    st.markdown(
                                        f'<div class="source-box"><p>{doc.page_content}</p></div>',
                                        unsafe_allow_html=True
                                    )
                        else:
                            answer = pipeline.answer_query(query)
                            st.markdown("### 📖 Answer")
                            st.markdown(
                                f'<div class="answer-box"><p>{answer}</p></div>',
                                unsafe_allow_html=True
                            )
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}", icon="❌")
        
        # Example queries
        st.markdown("---")
        st.markdown("### 💡 Example Queries")
        st.markdown("Click any example below to quickly search for common medical questions:")
        
        example_queries = [
            "What is skin disease?",
            "How is chlamydia transmitted?",
            "What are common treatments for skin conditions?",
            "What is PCOS?",
            "What are epidemiological definitions?",
            "How to prevent fungal infections?"
        ]
        
        cols = st.columns(3)
        for i, example in enumerate(example_queries):
            col = cols[i % 3]
            if col.button(f"🔹 {example}", use_container_width=True, key=f"example_{i}"):
                st.session_state.query = example
                st.rerun()
    
    # PAGE 2: Documents
    elif page == "📚 Documents":
        st.markdown('<div class="section-header"><h2>📚 Uploaded Documents</h2></div>', unsafe_allow_html=True)
        
        stats = get_document_stats()
        
        # Stats cards with better layout
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📄 Total PDFs", stats["count"], help="Number of uploaded documents")
        with col2:
            st.metric("💾 Total Size", f"{stats['total_size_mb']} MB", help="Combined file size")
        with col3:
            st.metric("📑 Document Status", "Ready", help="Vector store status")
        with col4:
            st.metric("🔄 Last Updated", "Today", help="Last indexing time")
        
        st.markdown("---")
        
        if stats["count"] > 0:
            st.markdown("### 📋 Uploaded Files")
            col1, col2, col3, col4 = st.columns([3, 1, 1, 0.5])
            col1.markdown("**Filename**")
            col2.markdown("**Size**")
            col3.markdown("**Indexed**")
            col4.markdown("**Action**")
            
            st.divider()
            
            for i, pdf in enumerate(stats["pdfs"], 1):
                col1, col2, col3, col4 = st.columns([3, 1, 1, 0.5])
                
                with col1:
                    st.markdown(f"**{i}. {pdf['filename']}**")
                
                with col2:
                    st.caption(f"{pdf['size_mb']} MB")
                
                with col3:
                    st.caption("✓ Yes")
                
                with col4:
                    if st.button("🗑️", key=f"delete_{i}", help="Delete document"):
                        handler = PDFUploadHandler()
                        result = handler.delete_pdf(pdf['filename'])
                        if result['success']:
                            st.success(f"✓ Deleted {pdf['filename']}")
                            st.rerun()
                        else:
                            st.error(result['message'])
        else:
            st.info("📭 No documents uploaded yet. Upload your first PDF to get started!", icon="ℹ️")
        
        st.markdown("---")
        st.markdown("### 💡 Tips")
        col1, col2 = st.columns(2)
        with col1:
            st.info("📌 **After uploading new PDFs:** Run `python scripts/index_medical_documents.py` to re-index", icon="💡")
        with col2:
            st.success("✅ **Your documents are secure:** All data stored locally", icon="✅")
    
    # PAGE 3: Upload PDF
    elif page == "📤 Upload PDF":
        st.markdown('<div class="section-header"><h2>📤 Upload Medical Documents</h2></div>', unsafe_allow_html=True)
        
        st.markdown("""
            Upload PDF files to expand your medical document collection.
            After uploading, re-index the documents to make them searchable via RAG.
        """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📁 Upload New File")
            uploaded_file = st.file_uploader(
                "Choose a PDF file",
                type="pdf",
                help="Select a medical PDF document (single file)"
            )
            
            if uploaded_file is not None:
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.write(f"📄 **File:** {uploaded_file.name}")
                with col_info2:
                    file_size_mb = uploaded_file.size / (1024*1024)
                    st.write(f"📊 **Size:** {file_size_mb:.2f} MB")
                
                if st.button("✅ Upload", type="primary", use_container_width=True):
                    # Save uploaded file
                    save_path = os.path.join(MEDICAL_PDFS_DIR, uploaded_file.name)
                    
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.success(f"✅ Successfully uploaded **{uploaded_file.name}**", icon="✅")
                    
                    st.markdown("### 🔄 Next Steps")
                    st.info(
                        "Run this command in your terminal to re-index documents:\n\n"
                        "`python scripts/index_medical_documents.py`\n\n"
                        "Then refresh this page to see the updated document list.",
                        icon="ℹ️"
                    )
        
        with col2:
            st.markdown("### 📊 Current Library")
            stats = get_document_stats()
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Total Files", stats["count"])
            with col_m2:
                st.metric("Total Size", f"{stats['total_size_mb']} MB")
            
            if stats["pdfs"]:
                st.markdown("**📋 Current Files:**")
                for pdf in stats["pdfs"]:
                    st.markdown(f'<div class="file-item">📄 {pdf["filename"]} ({pdf["size_mb"]} MB)</div>', unsafe_allow_html=True)
            else:
                st.info("No documents yet. Upload your first PDF!", icon="ℹ️")
    
    # PAGE 4: About
    elif page == "ℹ️ About":
        st.markdown('<div class="section-header"><h2>ℹ️ About Medical RAG Bot</h2></div>', unsafe_allow_html=True)
        
        st.markdown("""
            ## 🏥 What is Medical RAG Bot?
            
            Medical RAG Bot is a **Retrieval-Augmented Generation (RAG)** system specifically designed 
            for answering medical questions using your own document collection with high accuracy.
            
            ### 🔧 How It Works
            
            1. **Document Processing** 📄
               - PDFs are processed and split into chunks
               - Text is embedded using sentence transformers
               - Embeddings are stored in FAISS vector database
            
            2. **Question Processing** 🔍
               - Your question is converted to an embedding
               - Similar document chunks are retrieved
               - Context is passed to the language model
            
            3. **Answer Generation** 📝
               - LLM generates answers based on retrieved context
               - Source documents are provided for verification
               - Confidence scores show relevance
            
            ### ✨ Key Features
            
            - 🔍 **Smart Search** - Semantic search across your documents
            - 📚 **Document Management** - Easy upload and organization
            - 📊 **Source Attribution** - Know where answers come from
            - 🏥 **Medical Focus** - Optimized for medical content
            - 🔒 **Privacy** - All data stored locally
            - ⚡ **Fast** - Instant answers from your documents
            
            ### 📋 Technical Stack
            
            - **Backend:** LangChain, FAISS, Transformers
            - **Frontend:** Streamlit
            - **Embeddings:** Sentence Transformers
            - **LLM:** FLAN-T5 (Hugging Face)
            - **Language:** Python 3.14+
            
            ### 🚀 Getting Started
            
            1. Upload your medical PDFs (📤 Upload PDF section)
            2. Index documents: `python scripts/index_medical_documents.py`
            3. Ask questions in the 🔍 Ask Questions section
            4. View uploaded documents in the 📚 Documents section
            
            ### 💡 Tips for Best Results
            
            - Use clear, specific medical questions
            - Include relevant medical terminology
            - Ask one question at a time
            - Review source documents for context
            - Upload comprehensive medical documents
            
            ---
            
            **Version:** 1.0.0 | **Last Updated:** April 2026
        """)


if __name__ == "__main__":
    main()
