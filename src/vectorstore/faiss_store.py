"""
FAISS-based vector store for document embeddings.
Handles indexing, searching, and persistence of document vectors.
"""

import os
import pickle
from typing import List, Tuple, Dict, Any
import numpy as np
import faiss
from langchain_core.documents import Document

from src.embeddings import EmbeddingModel, get_embedding_model
from src.utils.config import VECTOR_STORE_PATH, EMBEDDINGS_DIMENSION


class FAISSVectorStore:
    """
    Vector store implementation using Facebook's FAISS library.
    
    Provides fast similarity search over document embeddings.
    Supports persistence (save/load) for production use.
    
    Features:
    - Create index from documents
    - Similarity search with configurable k
    - Similarity search with scores
    - Save/load index to disk
    - Metadata management for document retrieval
    """
    
    def __init__(self, embedding_model: EmbeddingModel = None):
        """
        Initialize FAISS vector store.
        
        Args:
            embedding_model (EmbeddingModel): Embedding model instance.
                                             If None, uses default.
        """
        self.embedding_model = embedding_model or get_embedding_model()
        self.index = None
        self.documents_metadata = []  # Store original docs for retrieval
        self.id_to_doc = {}  # Mapping from vector ID to original document
        
    def create_from_documents(self, documents: List[Document]) -> "FAISSVectorStore":
        """
        Create FAISS index from LangChain Document objects.
        
        Args:
            documents (List[Document]): List of LangChain Document objects
                                       Must have .page_content attribute
                                       
        Returns:
            FAISSVectorStore: Self for method chaining
        """
        if not documents:
            raise ValueError("Cannot create index from empty document list")
        
        # Extract texts and embed
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_model.embed_batch(texts)
        
        # Convert to numpy array (FAISS requirement)
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        dimension = embeddings_array.shape[1]
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance metric
        self.index.add(embeddings_array)
        
        # Store metadata
        self.documents_metadata = documents
        self.id_to_doc = {i: doc for i, doc in enumerate(documents)}
        
        return self
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Search for top-k most similar documents.
        
        Args:
            query (str): Query text
            k (int): Number of results to return (default: 5)
            
        Returns:
            List[Document]: List of k most similar documents
        """
        if self.index is None:
            raise RuntimeError("Index not initialized. Call create_from_documents() first.")
        
        # Embed query
        query_embedding = np.array([self.embedding_model.embed(query)]).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Return documents (indices[0] because batch size is 1)
        return [self.id_to_doc[idx] for idx in indices[0]]
    
    def similarity_search_with_scores(
        self, 
        query: str, 
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Search with similarity scores.
        
        Args:
            query (str): Query text
            k (int): Number of results to return
            
        Returns:
            List[Tuple[Document, float]]: List of (document, score) pairs.
                                         Score is inverse of L2 distance.
        """
        if self.index is None:
            raise RuntimeError("Index not initialized. Call create_from_documents() first.")
        
        query_embedding = np.array([self.embedding_model.embed(query)]).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        
        # Convert L2 distances to similarity scores (inverse)
        # Lower distance = higher similarity
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            doc = self.id_to_doc[idx]
            # Convert L2 distance to similarity score (0-1 range, higher = better)
            similarity_score = 1 / (1 + distance)
            results.append((doc, similarity_score))
        
        return results
    
    def save(self, save_path: str = VECTOR_STORE_PATH) -> None:
        """
        Save FAISS index and metadata to disk.
        
        Args:
            save_path (str): Path to save index (without extension)
                           Will create .faiss and .pkl files
        """
        if self.index is None:
            raise RuntimeError("No index to save. Create index first.")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        
        # Save FAISS index
        index_file = f"{save_path}.faiss"
        faiss.write_index(self.index, index_file)
        
        # Save metadata
        metadata_file = f"{save_path}.pkl"
        with open(metadata_file, "wb") as f:
            pickle.dump({
                "documents_metadata": self.documents_metadata,
                "id_to_doc": self.id_to_doc,
            }, f)
        
        print(f"[OK] Vector store saved: {index_file}, {metadata_file}")
    
    def load(self, load_path: str = VECTOR_STORE_PATH) -> "FAISSVectorStore":
        """
        Load FAISS index and metadata from disk.
        
        Args:
            load_path (str): Path to load index (without extension)
                           Must match save() path
                           
        Returns:
            FAISSVectorStore: Self for method chaining
        """
        # Load FAISS index
        index_file = f"{load_path}.faiss"
        self.index = faiss.read_index(index_file)
        
        # Load metadata
        metadata_file = f"{load_path}.pkl"
        with open(metadata_file, "rb") as f:
            data = pickle.load(f)
            self.documents_metadata = data["documents_metadata"]
            self.id_to_doc = data["id_to_doc"]
        
        print(f"[OK] Vector store loaded: {index_file}, {metadata_file}")
        return self
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Returns:
            Dict: Index statistics (num_vectors, dimension, etc.)
        """
        if self.index is None:
            return {"status": "not_initialized"}
        
        return {
            "num_vectors": self.index.ntotal,
            "dimension": self.index.d,
            "num_documents": len(self.documents_metadata),
            "status": "ready"
        }
    
    def __repr__(self) -> str:
        stats = self.get_index_stats()
        return f"FAISSVectorStore({stats})"
