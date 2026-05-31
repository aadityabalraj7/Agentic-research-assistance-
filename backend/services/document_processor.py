"""
Document processing service for handling file uploads and text extraction.
"""

import os
import io
import tempfile
from typing import List, Dict, Any, Tuple
from pathlib import Path

# PDF processing
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# DOCX processing
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# For unstructured processing (optional)
try:
    from unstructured.partition.auto import partition
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False


class DocumentProcessor:
    """Handles document processing for various file types."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_document(self, file_content: bytes, filename: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Process a document and return chunks with metadata.
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            
        Returns:
            Tuple of (text_chunks, metadata_list)
        """
        # Extract text based on file type
        if filename.lower().endswith('.pdf'):
            text = self._extract_pdf_text(file_content)
        elif filename.lower().endswith('.docx'):
            text = self._extract_docx_text(file_content)
        elif filename.lower().endswith('.txt'):
            text = self._extract_txt_text(file_content)
        else:
            raise ValueError(f"Unsupported file type: {filename}")
        
        # Chunk the text
        chunks = self._chunk_text(text)
        
        # Create metadata for each chunk
        metadata_list = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "filename": filename,
                "chunk_index": i,
                "chunk_id": f"{filename}_chunk_{i}",
                "total_chunks": len(chunks)
            }
            
            # Try to extract page number for PDFs
            if filename.lower().endswith('.pdf'):
                # This would require more sophisticated PDF parsing
                # For now, we'll estimate page number
                metadata["estimated_page"] = max(1, (i * self.chunk_size) // 2000 + 1)  # Rough estimate
            
            metadata_list.append(metadata)
        
        return chunks, metadata_list
    
    def _extract_pdf_text(self, file_content: bytes) -> str:
        """Extract text from PDF file."""
        if not PDF_AVAILABLE:
            raise ImportError("PyPDF2 is not installed. Install with: pip install PyPDF2")
        
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        return text
    
    def _extract_docx_text(self, file_content: bytes) -> str:
        """Extract text from DOCX file."""
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx is not installed. Install with: pip install python-docx")
        
        docx_file = io.BytesIO(file_content)
        doc = docx.Document(docx_file)
        text = ""
        
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        
        return text
    
    def _extract_txt_text(self, file_content: bytes) -> str:
        """Extract text from TXT file."""
        try:
            return file_content.decode('utf-8')
        except UnicodeDecodeError:
            # Try other encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    return file_content.decode(encoding)
                except UnicodeDecodeError:
                    continue
            raise ValueError("Unable to decode text file with common encodings")
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return [""]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at a sentence or word boundary
            if end < len(text):
                # Look for sentence boundary
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + self.chunk_size // 2:
                    end = sentence_end + 1
                else:
                    # Look for word boundary
                    word_end = text.rfind(' ', start, end)
                    if word_end > start + self.chunk_size // 2:
                        end = word_end
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= len(text):  # Prevent infinite loop
                break
        
        return chunks if chunks else [text]


# Global instance
document_processor = DocumentProcessor()