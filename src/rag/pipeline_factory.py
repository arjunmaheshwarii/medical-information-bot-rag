"""
Factory functions for easily creating RAG pipelines.
Simplifies initialization with sensible defaults.
"""

from src.embeddings import EmbeddingModel, get_embedding_model
from src.vectorstore import FAISSVectorStore
from src.retriever import Retriever
from src.llm import FlanT5Model
from src.utils.config import DEFAULT_TOP_K, VECTOR_STORE_PATH

from .pipeline import RAGPipeline


def create_rag_pipeline(
    vector_store_path: str = VECTOR_STORE_PATH,
    load_existing: bool = False,
    top_k: int = DEFAULT_TOP_K,
) -> RAGPipeline:
    """
    Create a complete RAG pipeline with default components.
    
    Args:
        vector_store_path (str): Path to vector store files
        load_existing (bool): If True, load existing index.
                             If False, create empty for new documents
        top_k (int): Default retrieval count
    
    Returns:
        RAGPipeline: Initialized RAG pipeline
    
    Example:
        >>> # Create new pipeline
        >>> pipeline = create_rag_pipeline(load_existing=False)
        >>> 
        >>> # Load existing pipeline
        >>> pipeline = create_rag_pipeline(load_existing=True)
        >>> answer = pipeline.answer_query("What is diabetes?")
    """
    # Initialize components
    embedding_model = get_embedding_model()
    vector_store = FAISSVectorStore(embedding_model)
    
    # Load or leave empty
    if load_existing:
        vector_store.load(vector_store_path)
    
    retriever = Retriever(vector_store, default_k=top_k)
    llm = FlanT5Model()
    
    # Compose pipeline
    pipeline = RAGPipeline(
        embedding_model=embedding_model,
        vector_store=vector_store,
        retriever=retriever,
        llm=llm,
        top_k=top_k,
    )
    
    return pipeline


def create_empty_rag_pipeline(top_k: int = DEFAULT_TOP_K) -> RAGPipeline:
    """
    Create an empty RAG pipeline (no indexed documents).
    Useful for testing or dynamic indexing.
    
    Args:
        top_k (int): Default retrieval count
    
    Returns:
        RAGPipeline: Empty RAG pipeline
    """
    return create_rag_pipeline(load_existing=False, top_k=top_k)


def load_rag_pipeline(
    vector_store_path: str = VECTOR_STORE_PATH,
    top_k: int = DEFAULT_TOP_K,
) -> RAGPipeline:
    """
    Load an existing RAG pipeline from disk.
    
    Args:
        vector_store_path (str): Path to saved vector store
        top_k (int): Default retrieval count
    
    Returns:
        RAGPipeline: Loaded RAG pipeline
    
    Example:
        >>> pipeline = load_rag_pipeline()
        >>> answer = pipeline.answer_query("medical question")
    """
    return create_rag_pipeline(vector_store_path=vector_store_path, load_existing=True, top_k=top_k)
