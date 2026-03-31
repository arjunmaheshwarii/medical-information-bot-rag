"""
Integration tests for RAG pipeline.
Tests the full workflow end-to-end.
"""

import unittest
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embeddings import EmbeddingModel
from src.vectorstore import FAISSVectorStore
from src.retriever import Retriever
from src.llm import FlanT5Model
from src.rag import RAGPipeline, create_empty_rag_pipeline
from src.loaders import PDFUploadHandler
from langchain_core.documents import Document


class TestEmbeddings(unittest.TestCase):
    """Test embedding model."""
    
    def setUp(self):
        self.model = EmbeddingModel()
    
    def test_single_embedding(self):
        """Test single text embedding."""
        text = "What is diabetes?"
        embedding = self.model.embed(text)
        
        self.assertIsInstance(embedding, list)
        self.assertEqual(len(embedding), 384)  # MiniLM dimension
    
    def test_batch_embedding(self):
        """Test batch text embedding."""
        texts = ["diabetes", "hypertension", "infection"]
        embeddings = self.model.embed_batch(texts)
        
        self.assertEqual(len(embeddings), 3)
        self.assertEqual(len(embeddings[0]), 384)


class TestVectorStore(unittest.TestCase):
    """Test FAISS vector store."""
    
    def setUp(self):
        self.model = EmbeddingModel()
        self.store = FAISSVectorStore(self.model)
        
        # Create sample documents
        self.docs = [
            Document(page_content="Diabetes is a metabolic disorder"),
            Document(page_content="Hypertension affects millions"),
            Document(page_content="COVID-19 is a viral infection"),
        ]
    
    def test_create_index(self):
        """Test index creation."""
        self.store.create_from_documents(self.docs)
        stats = self.store.get_index_stats()
        
        self.assertEqual(stats["num_vectors"], 3)
        self.assertEqual(stats["num_documents"], 3)
        self.assertEqual(stats["status"], "ready")
    
    def test_similarity_search(self):
        """Test similarity search."""
        self.store.create_from_documents(self.docs)
        results = self.store.similarity_search("What is diabetes?", k=2)
        
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], Document)
    
    def test_save_load(self):
        """Test saving and loading vector store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_index")
            
            # Save
            self.store.create_from_documents(self.docs)
            self.store.save(save_path)
            
            # Load
            store2 = FAISSVectorStore(self.model)
            store2.load(save_path)
            
            # Verify
            stats1 = self.store.get_index_stats()
            stats2 = store2.get_index_stats()
            
            self.assertEqual(stats1["num_vectors"], stats2["num_vectors"])


class TestRetriever(unittest.TestCase):
    """Test document retriever."""
    
    def setUp(self):
        self.model = EmbeddingModel()
        self.store = FAISSVectorStore(self.model)
        
        docs = [
            Document(page_content="Diabetes is a metabolic disorder"),
            Document(page_content="Hypertension affects blood pressure"),
            Document(page_content="COVID-19 pandemic information"),
        ]
        
        self.store.create_from_documents(docs)
        self.retriever = Retriever(self.store, default_k=2)
    
    def test_retrieve(self):
        """Test document retrieval."""
        results = self.retriever.retrieve("diabetes disorder")
        
        self.assertEqual(len(results), 2)
        self.assertIn("Diabetes", results[0].page_content)
    
    def test_retrieve_with_scores(self):
        """Test retrieval with scores."""
        results = self.retriever.retrieve_with_scores("medical condition", k=2)
        
        self.assertEqual(len(results), 2)
        doc, score = results[0]
        self.assertIsInstance(doc, Document)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)


class TestRAGPipeline(unittest.TestCase):
    """Test RAG pipeline."""
    
    def setUp(self):
        self.pipeline = create_empty_rag_pipeline(top_k=2)
        
        # Add sample documents
        docs = [
            Document(page_content="Diabetes is characterized by high blood sugar levels"),
            Document(page_content="Hypertension is elevated blood pressure"),
            Document(page_content="Heart disease is a cardiovascular condition"),
        ]
        
        self.pipeline.vector_store.create_from_documents(docs)
        # Recreate retriever with new vector store
        self.pipeline.retriever = Retriever(self.pipeline.vector_store, default_k=2)
    
    def test_pipeline_creation(self):
        """Test pipeline initialization."""
        self.assertIsNotNone(self.pipeline.embedding_model)
        self.assertIsNotNone(self.pipeline.vector_store)
        self.assertIsNotNone(self.pipeline.retriever)
        self.assertIsNotNone(self.pipeline.llm)
    
    def test_answer_query(self):
        """Test query answering."""
        query = "What is diabetes?"
        answer = self.pipeline.answer_query(query)
        
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)
    
    def test_answer_with_context(self):
        """Test query answering with source documents."""
        query = "What is a medical condition?"
        answer, sources = self.pipeline.answer_query_with_context(query)
        
        self.assertIsInstance(answer, str)
        self.assertIsInstance(sources, list)
        self.assertGreater(len(sources), 0)


class TestPDFUploadHandler(unittest.TestCase):
    """Test PDF upload handler."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.handler = PDFUploadHandler(self.temp_dir)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_handler_creation(self):
        """Test handler initialization."""
        self.assertTrue(os.path.exists(self.temp_dir))
        self.assertEqual(self.handler.get_pdf_count(), 0)
    
    def test_list_uploaded_pdfs(self):
        """Test listing PDFs."""
        pdfs = self.handler.list_uploaded_pdfs()
        self.assertIsInstance(pdfs, list)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
