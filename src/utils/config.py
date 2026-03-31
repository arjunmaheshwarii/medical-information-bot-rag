"""
Configuration constants and settings for the RAG pipeline.
Centralized place for model names, paths, and hyperparameters.
"""

# Embeddings configuration
EMBEDDINGS_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_DIMENSION = 384  # Dimension of all-MiniLM-L6-v2 embeddings

# Vector store configuration
VECTOR_STORE_PATH = "data/vector_store/faiss_index"
VECTOR_STORE_METADATA_PATH = "data/vector_store/metadata.pkl"

# Data paths
MEDICAL_PDFS_DIR = "data/medical_pdfs"
DATA_DIR = "data"

# LLM configuration
LLM_MODEL_NAME = "google/flan-t5-base"
LLM_MAX_TOKENS = 200
LLM_TEMPERATURE = 0.7

# RAG configuration
DEFAULT_TOP_K = 5  # Number of documents to retrieve
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# Device configuration (will auto-detect GPU if available)
DEVICE = "cuda"  # Will be set to "auto" in code for auto-detection
