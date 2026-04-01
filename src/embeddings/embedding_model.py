"""
Embedding model wrapper using sentence-transformers.
Converts text to numerical embeddings for semantic search.
"""

from typing import List, Union
from sentence_transformers import SentenceTransformer
from src.utils.config import EMBEDDINGS_MODEL_NAME, EMBEDDINGS_DIMENSION
import torch
from functools import lru_cache
import hashlib


class EmbeddingModel:
    """
    Wrapper for sentence-transformers embeddings.
    
    Uses all-MiniLM-L6-v2 by default, which is optimized for:
    - Speed (efficient inference)
    - Quality (trained on semantic similarity)
    - Size (lightweight, ~384D embeddings)
    
    Perfect for medical document retrieval tasks.
    """
    
    def __init__(self, model_name: str = EMBEDDINGS_MODEL_NAME):
        """
        Initialize embedding model.
        
        Args:
            model_name (str): HuggingFace model identifier
                             Default: 'all-MiniLM-L6-v2'
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = EMBEDDINGS_DIMENSION
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    @lru_cache(maxsize=1000)
    def _cached_encode(self, text_hash: str, text: str) -> List[float]:
        """Cache embeddings by text hash."""
        return self.model.encode([text], convert_to_tensor=False, show_progress_bar=False)[0].tolist()
    
    def _get_optimal_batch_size(self) -> int:
        """Auto-detect optimal batch size based on available memory."""
        if self.device == "cuda":
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            return min(128, max(8, gpu_mem // (1024**3)))  # Rough heuristic
        return 32  # CPU default
        
    def embed(self, text: str) -> List[float]:
        """
        Embed a single text string.
        
        Args:
            text (str): Text to embed
            
        Returns:
            List[float]: Embedding vector (384-dimensional for MiniLM)
        """
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return self._cached_encode(text_hash, text)
    
    def embed_batch(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """
        Embed multiple texts efficiently.
        
        Args:
            texts (List[str]): List of texts to embed
            batch_size (int): Batch size for processing (auto-detected if None)
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        if batch_size is None:
            batch_size = self._get_optimal_batch_size()
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=False,
            show_progress_bar=False
        )
        return embeddings.tolist()
    
    def get_embedding_dim(self) -> int:
        """
        Get dimension of embeddings.
        
        Returns:
            int: Embedding dimension (384 for MiniLM)
        """
        return self.embedding_dim
    
    def __repr__(self) -> str:
        return f"EmbeddingModel(model='{self.model_name}', dim={self.embedding_dim})"


# Convenient singleton pattern for lazy initialization
_embedding_model = None


def get_embedding_model(model_name: str = EMBEDDINGS_MODEL_NAME) -> EmbeddingModel:
    """
    Get or create embedding model instance (singleton-like).
    
    Args:
        model_name (str): HuggingFace model identifier
        
    Returns:
        EmbeddingModel: Embedding model instance
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel(model_name)
    return _embedding_model
