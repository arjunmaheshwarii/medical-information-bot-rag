"""
Basic RAG example showing end-to-end usage.
Run this to test the entire system.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pdf_loader import load_medical_pdfs
from src.chunking import chunk_documents
from src.rag.pipeline_factory import create_rag_pipeline, load_rag_pipeline
from src.loaders import PDFUploadHandler
from src.utils.config import MEDICAL_PDFS_DIR, VECTOR_STORE_PATH


def example_1_basic_query():
    """
    Example 1: Basic question-answering.
    Build index from PDFs and answer a query.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic RAG Query")
    print("="*60)
    
    # Load and chunk documents
    print("\n1. Loading medical PDFs...")
    docs = load_medical_pdfs(MEDICAL_PDFS_DIR)
    if not docs:
        print("⚠ No PDFs found. Please add PDFs to data/medical_pdfs/")
        return
    
    print(f"✓ Loaded {len(docs)} pages")
    
    print("\n2. Chunking documents...")
    chunks = chunk_documents(docs, chunk_size=400, chunk_overlap=50)
    print(f"✓ Created {len(chunks)} chunks")
    
    print("\n3. Creating RAG pipeline...")
    pipeline = create_rag_pipeline(load_existing=False)
    pipeline.vector_store.create_from_documents(chunks)
    print(f"✓ Vector store ready: {pipeline.vector_store.get_index_stats()}")
    
    print("\n4. Saving index...")
    pipeline.vector_store.save(VECTOR_STORE_PATH)
    print(f"✓ Index saved to {VECTOR_STORE_PATH}")
    
    print("\n5. Answering query...")
    query = "What is the main topic of the medical documents?"
    answer = pipeline.answer_query(query)
    
    print(f"\nQuery: {query}")
    print(f"\nAnswer:\n{answer}")


def example_2_with_sources():
    """
    Example 2: Get answer with source documents.
    Shows which documents were used to generate the answer.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Query with Source Documents")
    print("="*60)
    
    try:
        print("\n1. Loading saved index...")
        pipeline = load_rag_pipeline(VECTOR_STORE_PATH)
        print("✓ Pipeline loaded")
    except FileNotFoundError:
        print("⚠ Index not found. Run Example 1 first.")
        return
    
    print("\n2. Answering query with sources...")
    query = "What is a common medical condition?"
    answer, sources = pipeline.answer_query_with_context(query)
    
    print(f"\nQuery: {query}")
    print(f"\nAnswer:\n{answer}")
    
    print(f"\nSource documents ({len(sources)}):")
    for i, doc in enumerate(sources, 1):
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"  {i}. ...{preview}...")


def example_3_upload_pdf():
    """
    Example 3: Upload and manage PDFs.
    Demonstrates the upload handler functionality.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: PDF Upload & Management")
    print("="*60)
    
    handler = PDFUploadHandler()
    
    print(f"\n1. Current PDFs: {handler.get_pdf_count()} files, {handler.get_total_size_mb()} MB")
    
    # List PDFs
    pdfs = handler.list_uploaded_pdfs()
    if pdfs:
        print("\nUploaded PDFs:")
        for pdf in pdfs:
            print(f"  - {pdf['filename']} ({pdf['size_mb']} MB)")
    else:
        print("\nNo PDFs uploaded yet.")
    
    # Example: How to upload (commented out)
    print("\n2. To upload a PDF, use:")
    print("   result = handler.upload_pdf('/path/to/file.pdf')")
    print("   if result['success']:")
    print("       print('Upload successful')")
    
    # Example: How to delete (commented out)
    print("\n3. To delete a PDF, use:")
    print("   result = handler.delete_pdf('filename.pdf')")
    print("   if result['success']:")
    print("       print('Deletion successful')")


def example_4_batch_queries():
    """
    Example 4: Batch query processing.
    Answer multiple questions at once.
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Batch Query Processing")
    print("="*60)
    
    try:
        print("\n1. Loading pipeline...")
        pipeline = load_rag_pipeline(VECTOR_STORE_PATH)
    except FileNotFoundError:
        print("⚠ Index not found. Run Example 1 first.")
        return
    
    queries = [
        "What is the main topic?",
        "What treatments are mentioned?",
        "What is a medical condition?",
    ]
    
    print(f"\n2. Processing {len(queries)} queries...\n")
    
    for i, query in enumerate(queries, 1):
        print(f"{i}. Query: {query}")
        try:
            answer = pipeline.answer_query(query)
            print(f"   Answer: {answer[:100]}...\n")
        except Exception as e:
            print(f"   Error: {str(e)}\n")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("RAG PIPELINE EXAMPLES")
    print("="*60)
    
    # Check if data directory exists
    if not os.path.exists(MEDICAL_PDFS_DIR):
        print(f"\n⚠ Medical PDFs directory not found: {MEDICAL_PDFS_DIR}")
        print("Please create this directory and add medical PDF files.")
        return
    
    # Run examples
    example_1_basic_query()
    example_2_with_sources()
    example_3_upload_pdf()
    example_4_batch_queries()
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
