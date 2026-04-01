"""
Convenience wrapper around RAG pipeline.
For backward compatibility and easy use.
"""

from src.pdf_loader import load_medical_pdfs
from src.chunking import chunk_documents
from src.rag.pipeline_factory import create_rag_pipeline, load_rag_pipeline
from src.utils.config import MEDICAL_PDFS_DIR, VECTOR_STORE_PATH


def run_rag_query(query: str, load_from_disk: bool = False) -> str:
    """
    Run a complete RAG query from scratch or from saved index.

    Args:
        query (str): User question
        load_from_disk (bool): If True, load existing index.
                              If False, build from PDFs.

    Returns:
        str: Generated answer
    """

    if load_from_disk:
        # Load existing pipeline
        pipeline = load_rag_pipeline(VECTOR_STORE_PATH)

    else:
        # Build pipeline from scratch
        print("Loading medical PDFs...")
        docs = load_medical_pdfs(MEDICAL_PDFS_DIR)

        print(f"Chunking {len(docs)} documents...")
        chunks = chunk_documents(docs)

        print(f"Creating vector store from {len(chunks)} chunks...")
        pipeline = create_rag_pipeline(load_existing=False)
        pipeline.vector_store.create_from_documents(chunks)

        print(f"Saving index to {VECTOR_STORE_PATH}...")
        pipeline.vector_store.save(VECTOR_STORE_PATH)

    # Run query
    print(f"\nAnswering: {query}\n")
    answer = pipeline.answer_query(query)

    return answer


if __name__ == "__main__":
    query = "What is a common infection?"
    answer = run_rag_query(query, load_from_disk=False)

    print("Question:", query)
    print("\nAnswer:")
    print(answer)