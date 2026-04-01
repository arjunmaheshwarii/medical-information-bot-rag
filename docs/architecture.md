# Architecture Documentation

## System Overview

The Medical Information Bot is a RAG system that processes medical PDF documents to provide accurate answers to user queries.

## Components

### 1. Document Processing Pipeline
- PDF loading with PyPDF2
- Text chunking with LangChain RecursiveCharacterTextSplitter
- Embedding generation with Sentence-Transformers

### 2. Vector Storage
- FAISS index for efficient similarity search
- Persistence to disk for reuse

### 3. Retrieval System
- Semantic search with confidence scoring
- Re-ranking based on similarity thresholds

### 4. Answer Generation
- FLAN-T5 model for natural language generation
- Prompt engineering for medical context

### 5. User Interface
- Streamlit for web interface
- File upload with validation
- Query interface with results display

## Data Flow

1. User uploads PDFs → Validation → Storage
2. Documents indexed → Chunking → Embedding → FAISS storage
3. User query → Sanitization → Embedding → Retrieval → Re-ranking
4. Retrieved chunks + query → LLM → Answer generation
5. Answer + sources + confidence → UI display

## Security Considerations

- Input sanitization prevents injection attacks
- File validation prevents malicious uploads
- No external network calls
- Local model execution ensures data privacy