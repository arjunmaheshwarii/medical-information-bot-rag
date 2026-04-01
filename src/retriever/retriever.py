"""
Retriever module for document retrieval from vector store.
Handles query interpretation and result filtering.
"""

from typing import List, Tuple
from langchain_core.documents import Document

from src.vectorstore import FAISSVectorStore
from src.utils.config import DEFAULT_TOP_K


class Retriever:
    """
    Document retriever using vector store for semantic search.
    
    Wraps the vector store and provides a clean interface for
    requesting relevant documents given a query.
    
    Features:
    - Retrieve top-k documents
    - Retrieve with relevance scores
    - Configurable retrieval parameters
    """
    
    def __init__(self, vector_store: FAISSVectorStore, default_k: int = DEFAULT_TOP_K):
        """
        Initialize retriever with a vector store.
        
        Args:
            vector_store (FAISSVectorStore): Vector store instance (can be uninitialized)
            default_k (int): Default number of documents to retrieve
        """
        self.vector_store = vector_store
        self.default_k = default_k
    
    def retrieve(self, query: str, k: int = None) -> List[Document]:
        """
        Retrieve top-k documents most similar to query.
        
        Args:
            query (str): Query text
            k (int): Number of documents to retrieve.
                    If None, uses default_k
        
        Returns:
            List[Document]: List of retrieved documents
            
        Example:
            >>> retriever = Retriever(vector_store, default_k=5)
            >>> docs = retriever.retrieve("What is diabetes?", k=3)
            >>> print(f"Retrieved {len(docs)} documents")
        """
        if self.vector_store.index is None:
            raise RuntimeError("Vector store index not initialized. Create index before retrieving.")
        
        if k is None:
            k = self.default_k
        
        return self.vector_store.similarity_search(query, k=k)
    
    def retrieve_with_scores(
        self, 
        query: str, 
        k: int = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with their relevance scores.
        
        Args:
            query (str): Query text
            k (int): Number of documents to retrieve.
                    If None, uses default_k
        
        Returns:
            List[Tuple[Document, float]]: List of (document, score) pairs
                                         Score is between 0-1, higher = more relevant
        
        Example:
            >>> results = retriever.retrieve_with_scores("treatment options", k=5)
            >>> for doc, score in results:
            ...     print(f"Score: {score:.2f} - {doc.page_content[:100]}")
        """
        if k is None:
            k = self.default_k
        
        return self.vector_store.similarity_search_with_scores(query, k=k)
    
    def set_default_k(self, k: int) -> None:
        """
        Update default number of documents to retrieve.
        
        Args:
            k (int): New default k value
        """
        self.default_k = k
    
    def __repr__(self) -> str:
        stats = self.vector_store.get_index_stats()
        return f"Retriever(k={self.default_k}, {stats})"
