"""
Standalone script to build vector index from medical PDFs.
Run this to prepare documents for RAG queries.

Usage:
    python scripts/index_medical_documents.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pdf_loader import load_medical_pdfs
from src.chunking import chunk_documents
from src.rag.pipeline_factory import create_rag_pipeline
from src.utils.config import MEDICAL_PDFS_DIR, VECTOR_STORE_PATH


def main():
    """Build index from medical PDFs."""
    
    print("\n" + "="*70)
    print("MEDICAL DOCUMENT INDEXING")
    print("="*70)
    
    # Step 1: Load PDFs
    print("\n[1/4] Loading medical PDFs from:", MEDICAL_PDFS_DIR)
    docs = load_medical_pdfs(MEDICAL_PDFS_DIR)
    
    if not docs:
        print("\n❌ Error: No PDFs found!")
        print("Please add PDF files to", MEDICAL_PDFS_DIR)
        return
    
    print(f"✓ Loaded {len(docs)} pages from {len(set(d.metadata.get('source') for d in docs))} documents")
    
    # Step 2: Chunk documents
    print("\n[2/4] Chunking documents...")
    chunks = chunk_documents(docs, chunk_size=400, chunk_overlap=50)
    print(f"✓ Created {len(chunks)} chunks")
    
    # Step 3: Create RAG pipeline
    print("\n[3/4] Creating RAG pipeline...")
    pipeline = create_rag_pipeline(load_existing=False)
    pipeline.vector_store.create_from_documents(chunks)
    
    stats = pipeline.vector_store.get_index_stats()
    print(f"✓ Vector store created:")
    print(f"   - Vectors: {stats['num_vectors']}")
    print(f"   - Dimension: {stats['dimension']}")
    print(f"   - Documents: {stats['num_documents']}")
    
    # Step 4: Save index
    print("\n[4/4] Saving index to disk...")
    pipeline.vector_store.save(VECTOR_STORE_PATH)
    print(f"✓ Index saved to: {VECTOR_STORE_PATH}")
    
    # Summary
    print("\n" + "="*70)
    print("INDEXING COMPLETE!")
    print("="*70)
    print("\nYou can now run RAG queries using:")
    print("  - examples/basic_rag_example.py")
    print("  - Run from Python: from src.rag import load_rag_pipeline")
    print("\n")


if __name__ == "__main__":
    main()
