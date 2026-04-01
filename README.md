# Medical Information Bot RAG

A Retrieval-Augmented Generation (RAG) system for medical information queries using local models only.

## Features

- Local PDF document ingestion
- Semantic search using FAISS vector store
- Answer generation with FLAN-T5 model
- Confidence scoring for answers
- Secure file upload with validation
- Input sanitization for safety

## Architecture

The system consists of:
- **Frontend**: Streamlit web interface
- **Embeddings**: Sentence-Transformers for text encoding
- **Vector Store**: FAISS for efficient similarity search
- **LLM**: FLAN-T5 for answer generation
- **Chunking**: Recursive text splitting for document processing

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## Usage

1. Upload PDF documents via the sidebar
2. Index the documents
3. Ask questions in natural language
4. View answers with source citations and confidence scores

## Security

- No external API dependencies
- Input validation and sanitization
- File upload restrictions (PDF only, size limits)
- Local model execution only

## Troubleshooting

- Ensure CUDA is available for GPU acceleration
- Check memory usage for large document collections
- Verify PDF files are not corrupted
