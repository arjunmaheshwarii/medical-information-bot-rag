"""
PDF loader module for reading medical PDFs.
Supports loading from directories and individual files.
"""

import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from src.utils.config import MEDICAL_PDFS_DIR


def load_medical_pdfs(pdf_dir: str = MEDICAL_PDFS_DIR) -> List[Document]:
    """
    Load all PDF documents from a directory.
    
    Args:
        pdf_dir (str): Directory containing PDF files
                      Default: data/medical_pdfs/
    
    Returns:
        List[Document]: List of loaded documents
    
    Example:
        >>> docs = load_medical_pdfs("data/medical_pdfs")
        >>> print(f"Loaded {len(docs)} pages")
    """
    documents = []
    
    if not os.path.exists(pdf_dir):
        print(f"PDF directory not found: {pdf_dir}")
        return documents
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return documents
    
    for pdf_file in pdf_files:
        try:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            documents.extend(docs)
            print(f"[OK] Loaded {pdf_file} ({len(docs)} pages)")
        except Exception as e:
            print(f"[ERROR] Error loading {pdf_file}: {str(e)}")
    
    return documents


def load_single_pdf(file_path: str) -> List[Document]:
    """
    Load a single PDF file.
    
    Args:
        file_path (str): Path to PDF file
    
    Returns:
        List[Document]: List of page documents
    
    Example:
        >>> docs = load_single_pdf("data/medical_pdfs/cardiology.pdf")
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    
    loader = PyPDFLoader(file_path)
    return loader.load()


if __name__ == "__main__":
    docs = load_medical_pdfs(MEDICAL_PDFS_DIR)
    print(f"[OK] Loaded {len(docs)} pages from medical PDFs")
