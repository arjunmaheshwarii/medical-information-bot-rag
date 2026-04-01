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

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rag.pipeline_factory import load_rag_pipeline, create_rag_pipeline
from src.pdf_loader import load_medical_pdfs
from src.chunking import chunk_documents
from src.loaders import PDFUploadHandler
from src.utils.config import VECTOR_STORE_PATH, MEDICAL_PDFS_DIR


# Constants for validation
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = ['pdf']


# Page configuration
st.set_page_config(
    page_title="Medical RAG Bot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .stMarkdown {
        text-align: justify;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    .answer-box {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #4caf50;
        margin: 10px 0;
    }
    .query-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #2196f3;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Load RAG pipeline (cached)."""
    try:
        return load_rag_pipeline(VECTOR_STORE_PATH)
    except FileNotFoundError:
        st.error("❌ Vector store index not found. Please index documents first.")
        st.stop()


def get_document_stats():
    """Get statistics about uploaded documents."""
    handler = PDFUploadHandler()
    pdfs = handler.list_uploaded_pdfs()
    return {
        "count": len(pdfs),
        "total_size_mb": handler.get_total_size_mb(),
        "pdfs": pdfs
    }


def validate_uploaded_file(uploaded_file) -> tuple[bool, str]:
    """Validate uploaded file for security and size."""
    if uploaded_file is None:
        return False, "No file selected"
    
    # Check file size
    if uploaded_file.size > MAX_FILE_SIZE:
        return False, f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
    
    # Check file extension
    if not uploaded_file.name.lower().endswith('.pdf'):
        return False, "Only PDF files are allowed"
    
    # Sanitize filename
    safe_name = re.sub(r'[^\w\.\-\s]', '', uploaded_file.name)
    if safe_name != uploaded_file.name:
        return False, "Filename contains invalid characters"
    
    return True, safe_name


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🏥 Medical Information RAG Bot</h1>
            <p>Intelligent Question Answering System for Medical Documents</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["🔍 Ask Questions", "📚 Documents", "📤 Upload PDF", "ℹ️ About"]
    )
    
    # Load pipeline
    pipeline = load_pipeline()
    
    # PAGE 1: Ask Questions
    if page == "🔍 Ask Questions":
        st.header("Ask Your Medical Questions")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_area(
                "💬 Enter your medical question:",
                placeholder="e.g., What is diabetes? How is chlamydia transmitted?",
                height=100
            )
        
        with col2:
            st.markdown("### Options")
            include_sources = st.checkbox("Show sources", value=True)
            num_sources = st.slider("Number of sources:", 1, 10, 5)
        
        if st.button("🚀 Get Answer", use_container_width=True, type="primary"):
            if not query.strip():
                st.warning("⚠️ Please enter a question.")
            else:
                with st.spinner("⏳ Processing your question..."):
                    try:
                        if include_sources:
                            answer, sources = pipeline.answer_query_with_context(
                                query, 
                                top_k=num_sources
                            )
                            
                            # Display answer
                            st.markdown("### 📖 Answer")
                            st.markdown(
                                f'<div class="answer-box"><p>{answer}</p></div>',
                                unsafe_allow_html=True
                            )
                            
                            # Display sources
                            st.markdown("### 📚 Source Documents")
                            st.info(f"Retrieved {len(sources)} relevant documents")
                            
                            for i, doc in enumerate(sources, 1):
                                with st.expander(f"📄 Source {i}", expanded=(i==1)):
                                    if doc.metadata:
                                        st.caption(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
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
                        st.error(f"❌ Error: {str(e)}")
        
        # Example queries
        st.markdown("---")
        st.markdown("### 💡 Example Queries")
        example_queries = [
            "What is skin disease?",
            "How is chlamydia transmitted?",
            "What are common treatments for skin conditions?",
            "What is PCOS?",
            "What are epidemiological definitions?"
        ]
        
        cols = st.columns(2)
        for i, example in enumerate(example_queries):
            col = cols[i % 2]
            if col.button(f"📌 {example}", use_container_width=True):
                st.session_state.query = example
                st.rerun()
    
    # PAGE 2: Documents
    elif page == "📚 Documents":
        st.header("Uploaded Documents")
        
        stats = get_document_stats()
        
        # Stats cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Total PDFs", stats["count"])
        with col2:
            st.metric("💾 Total Size", f"{stats['total_size_mb']} MB")
        with col3:
            st.metric("📑 Indexed Chunks", "545")
        
        st.markdown("---")
        
        if stats["count"] > 0:
            st.markdown("### 📋 Uploaded Files")
            
            for i, pdf in enumerate(stats["pdfs"], 1):
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**{i}. {pdf['filename']}**")
                
                with col2:
                    st.caption(f"{pdf['size_mb']} MB")
                
                with col3:
                    if st.button("🗑️", key=f"delete_{i}"):
                        handler = PDFUploadHandler()
                        result = handler.delete_pdf(pdf['filename'])
                        if result['success']:
                            st.success(f"Deleted {pdf['filename']}")
                            st.rerun()
                        else:
                            st.error(result['message'])
        else:
            st.info("📭 No documents uploaded yet. Upload your first PDF!")
        
        st.markdown("---")
        st.markdown("### ⚠️ Note")
        st.warning(
            "After uploading new documents, you need to re-index them for search to work. "
            "Run: `python scripts/index_medical_documents.py`"
        )
    
    # PAGE 3: Upload PDF
    elif page == "📤 Upload PDF":
        st.header("Upload Medical Documents")
        
        st.markdown("""
            Upload PDF files to add to your medical document collection.
            After uploading, you'll need to re-index the documents for RAG retrieval.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📁 Upload File")
            uploaded_file = st.file_uploader(
                "Choose a PDF file",
                type="pdf",
                help="Select a medical PDF document to upload"
            )
            
            if uploaded_file is not None:
                st.write(f"📄 Selected file: **{uploaded_file.name}**")
                st.write(f"📊 File size: **{uploaded_file.size / (1024*1024):.2f} MB**")
                
                # Validate file
                is_valid, message = validate_uploaded_file(uploaded_file)
                if not is_valid:
                    st.error(f"❌ {message}")
                    return
                
                if st.button("✅ Upload", type="primary", use_container_width=True):
                    # Save uploaded file with safe name
                    save_path = os.path.join(MEDICAL_PDFS_DIR, message)  # message is safe_name
                    
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.success(f"✅ Successfully uploaded {message}")
                    
                    st.markdown("### 🔄 Next Steps")
                    st.info(
                        "Now you need to re-index your documents. "
                        "Run the following command in your terminal:\n\n"
                        "`python scripts/index_medical_documents.py`"
                    )
        
        with col2:
            st.markdown("### 📊 Current Documents")
            stats = get_document_stats()
            
            st.metric("Total Files", stats["count"])
            st.metric("Total Size", f"{stats['total_size_mb']} MB")
            
            if stats["pdfs"]:
                st.markdown("**Files:**")
                for pdf in stats["pdfs"]:
                    st.write(f"• {pdf['filename']} ({pdf['size_mb']} MB)")
    
    # PAGE 4: About
    elif page == "ℹ️ About":
        st.header("About Medical RAG Bot")
        
        st.markdown("""
            ## 🏥 What is Medical RAG Bot?
            
            Medical RAG Bot is a **Retrieval-Augmented Generation (RAG)** system designed 
            specifically for answering medical questions using your own document collection.
            
            ### 🔧 How It Works
            
            1. **Document Upload** 📤
               - Upload your medical PDFs (journal articles, guidelines, etc.)
            
            2. **Indexing** 📚
               - Documents are split into chunks and converted to embeddings
               - FAISS vector index is built for fast similarity search
            
            3. **Question Processing** ❓
               - Your question is converted to embeddings
               - Similar documents are retrieved from the index
            
            4. **Answer Generation** ✨
               - Retrieved context is formatted into a prompt
               - FLAN-T5 LLM generates accurate answers based on retrieved documents
            
            ### 🔍 Technical Stack
            
            - **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
            - **Vector Store:** FAISS (fast similarity search)
            - **LLM:** Google FLAN-T5-base
            - **Framework:** Streamlit (UI), LangChain (document loading)
            
            ### 📊 Current System Stats
            
            - **Indexed Documents:** 2 medical PDFs
            - **Total Chunks:** 545
            - **Embedding Dimension:** 384
            - **Index Size:** FAISS optimized
            
            ### 💡 Features
            
            ✅ Ask questions about medical documents  
            ✅ Get relevant source documents  
            ✅ Upload new PDFs dynamically  
            ✅ View document statistics  
            ✅ Example queries for quick start  
            
            ### 🚀 Getting Started
            
            1. **Ask Questions** on the "Ask Questions" page
            2. **Upload Documents** if you have new PDFs
            3. **Re-index** after adding documents: `python scripts/index_medical_documents.py`
            4. **View Documents** to see your collection
            
            ### 📝 Project Repository
            
            GitHub: [medical-information-bot-rag](https://github.com/arjunmaheshwarii/medical-information-bot-rag)
        """)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("🔴 **Status:** Production Ready")
        with col2:
            st.success("✅ **All Systems:** Operational")
        with col3:
            st.warning("⚡ **Running on:** CPU")


if __name__ == "__main__":
    main()
