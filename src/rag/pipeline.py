"""
Main RAG (Retrieval-Augmented Generation) pipeline.
Orchestrates all components: embeddings, vector store, retriever, and LLM.
"""

from typing import List, Tuple
from langchain_core.documents import Document

from src.embeddings import EmbeddingModel, get_embedding_model
from src.vectorstore import FAISSVectorStore
from src.retriever import Retriever
from src.llm import FlanT5Model
from src.utils.config import DEFAULT_TOP_K


class RAGPipeline:
    """
    Complete Retrieval-Augmented Generation pipeline.
    
    Workflow:
    1. User asks a question
    2. Retriever fetches relevant documents from vector store
    3. Context is formatted using prompt template
    4. LLM generates answer based on retrieved context
    5. Return answer to user
    
    Features:
    - Dependency injection for all components
    - Easy to test individual parts
    - Pluggable: can swap embedding model, LLM, etc.
    - Clean data flow: Question → Retrieved Docs → Prompt → LLM → Answer
    """
    
    def __init__(
        self,
        embedding_model: EmbeddingModel = None,
        vector_store: FAISSVectorStore = None,
        retriever: Retriever = None,
        llm: FlanT5Model = None,
        top_k: int = DEFAULT_TOP_K,
    ):
        """
        Initialize RAG pipeline with components.
        
        All components are optional - if not provided, defaults are used.
        This allows flexibility for testing and customization.
        
        Args:
            embedding_model (EmbeddingModel): Embedding model.
                                             If None, creates default
            vector_store (FAISSVectorStore): Vector store with indexed docs.
                                           If None, creates empty instance
            retriever (Retriever): Document retriever.
                                  If None, creates from vector_store
            llm (FlanT5Model): Language model.
                              If None, creates default FLAN-T5
            top_k (int): Number of documents to retrieve
        """
        self.embedding_model = embedding_model or get_embedding_model()
        self.vector_store = vector_store or FAISSVectorStore(self.embedding_model)
        self.retriever = retriever or Retriever(self.vector_store, default_k=top_k)
        self.llm = llm or FlanT5Model()
        self.top_k = top_k
    
    def answer_query(
        self,
        query: str,
        template_name: str = "qa",
        top_k: int = None,
    ) -> str:
        """
        Answer a user query using RAG pipeline.
        
        Complete workflow:
        1. Retrieve relevant documents
        2. Extract context
        3. Format prompt with context
        4. Generate answer using LLM
        
        Args:
            query (str): User question
            template_name (str): Prompt template to use
                                ('qa', 'summary', 'explain')
            top_k (int): Number of documents to retrieve.
                        If None, uses default
        
        Returns:
            str: Generated answer based on context
        
        Example:
            >>> pipeline = RAGPipeline()
            >>> answer = pipeline.answer_query("What causes diabetes?")
            >>> print(answer)
        """
        if top_k is None:
            top_k = self.top_k
        
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(query, k=top_k)
        
        # Step 2: Extract and concatenate context
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        
        # Step 3: Generate answer
        answer = self.llm.generate(
            context=context,
            question=query,
            template_name=template_name,
        )
        
        return answer
    
    def answer_query_with_context(
        self,
        query: str,
        template_name: str = "qa",
        top_k: int = None,
    ) -> Tuple[str, List[Dict]]:
        """
        Answer query and return both answer and retrieved documents with confidence.
        
        Useful for debugging or displaying source documents to users.
        
        Args:
            query (str): User question
            template_name (str): Prompt template to use
            top_k (int): Number of documents to retrieve
        
        Returns:
            Tuple[str, List[Dict]]: (answer, sources_with_confidence)
                                   sources format: [{"content": str, "source": str, "confidence": float}]
        
        Example:
            >>> pipeline = RAGPipeline()
            >>> answer, sources = pipeline.answer_query_with_context(
            ...     "What is hypertension?"
            ... )
            >>> print(f"Answer: {answer}")
            >>> print(f"Sources: {len(sources)} documents")
        """
        if top_k is None:
            top_k = self.top_k
        
        # Retrieve documents with scores
        docs, scores = self.retriever.retrieve(query, k=top_k)
        
        if not docs:
            return "I don't have sufficient information to answer this question.", []
        
        # Build context with scores
        context_parts = []
        sources = []
        for i, (doc, score) in enumerate(zip(docs, scores)):
            context_parts.append(f"Document {i+1} (confidence: {score:.3f}):\n{doc.page_content}")
            sources.append({
                "content": doc.page_content[:200] + "...",
                "source": doc.metadata.get("source", "Unknown"),
                "confidence": score
            })
        
        context = "\n\n".join(context_parts)
        
        # Generate answer
        answer = self.llm.generate(
            context=context,
            question=query,
            template_name=template_name,
        )
        
        return answer, sources
    
    def update_config(self, **kwargs) -> None:
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Configuration to update
                     - top_k: Default retrieval count
                     - temperature: LLM temperature
                     - max_tokens: LLM max tokens
        """
        if "top_k" in kwargs:
            self.top_k = kwargs["top_k"]
            self.retriever.set_default_k(kwargs["top_k"])
    
    def get_stats(self) -> dict:
        """
        Get statistics about the pipeline.
        
        Returns:
            dict: Pipeline statistics
        """
        return {
            "embedding_model": str(self.embedding_model),
            "vector_store": str(self.vector_store),
            "retriever": str(self.retriever),
            "llm": str(self.llm),
            "default_top_k": self.top_k,
        }
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"RAGPipeline(top_k={self.top_k}, components={len(stats)})"
