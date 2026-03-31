"""
PDF upload handler for dynamic document management.
Allows users to upload medical PDFs at runtime.
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from langchain_community.document_loaders import PyPDFLoader

from src.utils.config import MEDICAL_PDFS_DIR


class PDFUploadHandler:
    """
    Handles PDF file uploads, validation, and management.
    
    Features:
    - Validate PDF files before storing
    - Upload single or multiple files
    - List uploaded PDFs
    - Delete PDFs
    - Get file metadata
    
    All PDFs are stored in data/medical_pdfs/
    """
    
    def __init__(self, upload_dir: str = MEDICAL_PDFS_DIR):
        """
        Initialize PDF upload handler.
        
        Args:
            upload_dir (str): Directory to store uploaded PDFs
                             Default: data/medical_pdfs/
        """
        self.upload_dir = upload_dir
        
        # Create directory if it doesn't exist
        os.makedirs(self.upload_dir, exist_ok=True)
    
    def upload_pdf(self, file_path: str, destination_name: Optional[str] = None) -> Dict:
        """
        Upload a PDF file to the upload directory.
        
        Args:
            file_path (str): Path to PDF file to upload
            destination_name (str): Name to save as. If None, uses original filename
        
        Returns:
            Dict: Upload result with metadata
                 {
                     "success": bool,
                     "filename": str,
                     "filepath": str,
                     "size_mb": float,
                     "message": str
                 }
        
        Example:
            >>> handler = PDFUploadHandler()
            >>> result = handler.upload_pdf("/path/to/medical_paper.pdf")
            >>> if result["success"]:
            ...     print(f"Uploaded: {result['filename']}")
        """
        # Validate file exists
        if not os.path.exists(file_path):
            return {
                "success": False,
                "filename": None,
                "filepath": None,
                "size_mb": 0,
                "message": f"File not found: {file_path}"
            }
        
        # Validate PDF
        validation = self.validate_pdf(file_path)
        if not validation["is_valid"]:
            return {
                "success": False,
                "filename": None,
                "filepath": None,
                "size_mb": 0,
                "message": f"Invalid PDF: {validation['error']}"
            }
        
        # Determine destination filename
        if destination_name is None:
            destination_name = os.path.basename(file_path)
        
        # Ensure .pdf extension
        if not destination_name.lower().endswith(".pdf"):
            destination_name += ".pdf"
        
        destination_path = os.path.join(self.upload_dir, destination_name)
        
        # Check if file already exists
        if os.path.exists(destination_path):
            return {
                "success": False,
                "filename": destination_name,
                "filepath": destination_path,
                "size_mb": 0,
                "message": f"File already exists: {destination_name}"
            }
        
        # Copy file
        try:
            shutil.copy2(file_path, destination_path)
            
            # Get file size
            size_bytes = os.path.getsize(destination_path)
            size_mb = round(size_bytes / (1024 * 1024), 2)
            
            return {
                "success": True,
                "filename": destination_name,
                "filepath": destination_path,
                "size_mb": size_mb,
                "message": f"Successfully uploaded {destination_name} ({size_mb} MB)"
            }
        except Exception as e:
            return {
                "success": False,
                "filename": destination_name,
                "filepath": destination_path,
                "size_mb": 0,
                "message": f"Upload error: {str(e)}"
            }
    
    def upload_batch(self, file_paths: List[str]) -> List[Dict]:
        """
        Upload multiple PDF files at once.
        
        Args:
            file_paths (List[str]): List of file paths to upload
        
        Returns:
            List[Dict]: List of upload results for each file
        
        Example:
            >>> handler = PDFUploadHandler()
            >>> results = handler.upload_batch([
            ...     "/path/file1.pdf",
            ...     "/path/file2.pdf"
            ... ])
            >>> for result in results:
            ...     print(f"{result['filename']}: {result['success']}")
        """
        results = []
        for file_path in file_paths:
            result = self.upload_pdf(file_path)
            results.append(result)
        
        return results
    
    @staticmethod
    def validate_pdf(file_path: str) -> Dict:
        """
        Validate that a file is a valid PDF.
        
        Args:
            file_path (str): Path to file to validate
        
        Returns:
            Dict: Validation result
                 {
                     "is_valid": bool,
                     "error": str (if invalid)
                 }
        """
        if not os.path.exists(file_path):
            return {"is_valid": False, "error": "File does not exist"}
        
        if not file_path.lower().endswith(".pdf"):
            return {"is_valid": False, "error": "File is not a PDF"}
        
        try:
            # Try to load with PyPDFLoader to validate
            loader = PyPDFLoader(file_path)
            loader.load()
            return {"is_valid": True, "error": None}
        except Exception as e:
            return {"is_valid": False, "error": str(e)}
    
    def list_uploaded_pdfs(self) -> List[Dict]:
        """
        List all uploaded PDF files.
        
        Returns:
            List[Dict]: List of PDF metadata
                       [{
                           "filename": str,
                           "filepath": str,
                           "size_mb": float,
                           "upload_time": str (creation time)
                       }]
        
        Example:
            >>> handler = PDFUploadHandler()
            >>> pdfs = handler.list_uploaded_pdfs()
            >>> for pdf in pdfs:
            ...     print(f"{pdf['filename']}: {pdf['size_mb']} MB")
        """
        pdfs = []
        
        if not os.path.exists(self.upload_dir):
            return pdfs
        
        for filename in os.listdir(self.upload_dir):
            if filename.lower().endswith(".pdf"):
                filepath = os.path.join(self.upload_dir, filename)
                size_bytes = os.path.getsize(filepath)
                size_mb = round(size_bytes / (1024 * 1024), 2)
                creation_time = os.path.getctime(filepath)
                
                pdfs.append({
                    "filename": filename,
                    "filepath": filepath,
                    "size_mb": size_mb,
                    "upload_time": creation_time
                })
        
        return pdfs
    
    def delete_pdf(self, filename: str) -> Dict:
        """
        Delete an uploaded PDF file.
        
        Args:
            filename (str): Name of file to delete
        
        Returns:
            Dict: Deletion result
                 {"success": bool, "message": str}
        
        Example:
            >>> handler = PDFUploadHandler()
            >>> result = handler.delete_pdf("old_document.pdf")
            >>> print(result["message"])
        """
        filepath = os.path.join(self.upload_dir, filename)
        
        if not os.path.exists(filepath):
            return {
                "success": False,
                "message": f"File not found: {filename}"
            }
        
        try:
            os.remove(filepath)
            return {
                "success": True,
                "message": f"Successfully deleted {filename}"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Deletion error: {str(e)}"
            }
    
    def get_pdf_count(self) -> int:
        """
        Get number of uploaded PDFs.
        
        Returns:
            int: Number of PDF files in upload directory
        """
        pdfs = self.list_uploaded_pdfs()
        return len(pdfs)
    
    def get_total_size_mb(self) -> float:
        """
        Get total size of all uploaded PDFs in MB.
        
        Returns:
            float: Total size in MB
        """
        pdfs = self.list_uploaded_pdfs()
        total = sum(pdf["size_mb"] for pdf in pdfs)
        return round(total, 2)
    
    def __repr__(self) -> str:
        count = self.get_pdf_count()
        total_size = self.get_total_size_mb()
        return f"PDFUploadHandler(dir='{self.upload_dir}', files={count}, size={total_size}MB)"
