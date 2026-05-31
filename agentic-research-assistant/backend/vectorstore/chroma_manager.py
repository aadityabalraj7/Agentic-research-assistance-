"""
ChromaDB vector store manager for document storage and retrieval.
"""

import os
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from backend.models.schemas import DocumentChunk, DocumentSource


class ChromaManager:
    """Manages ChromaDB collections for document storage and retrieval."""
    
    def __init__(self, db_path: str = "./chroma_db"):
        """
        Initialize ChromaDB manager.
        
        Args:
            db_path: Path to ChromaDB persistence directory
        """
        self.db_path = db_path
        self.client = None
        self.collection = None
        self.embedding_function = None
        self._initialize()
    
    def _initialize(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Ensure directory exists
            Path(self.db_path).mkdir(parents=True, exist_ok=True)
            
            # Initialize client
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Initialize embedding function (using sentence transformers)
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="research_documents",
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            raise
    
    def add_documents(self, chunks: List[DocumentChunk]) -> List[str]:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of document chunks to add
            
        Returns:
            List of chunk IDs that were added
        """
        try:
            ids = []
            texts = []
            metadatas = []
            
            for chunk in chunks:
                chunk_id = chunk.id or str(uuid.uuid4())
                ids.append(chunk_id)
                texts.append(chunk.content)
                metadatas.append(chunk.metadata)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
            
            return ids
            
        except Exception as e:
            print(f"Error adding documents to ChromaDB: {e}")
            raise
    
    def search_documents(self, query: str, n_results: int = 5) -> List[DocumentSource]:
        """
        Search for relevant document chunks.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of document sources with content and metadata
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            sources = []
            if results["ids"] and len(results["ids"][0]) > 0:
                for i, chunk_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    document = results["documents"][0][i] if results["documents"] else ""
                    distance = results["distances"][0][i] if results["distances"] else 0.0
                    
                    # Convert distance to similarity score (chromadb uses cosine distance)
                    similarity_score = 1.0 - distance
                    
                    source = DocumentSource(
                        filename=metadata.get("filename", "Unknown"),
                        page_number=metadata.get("page_number"),
                        chunk_id=chunk_id,
                        content_preview=document[:200] + "..." if len(document) > 200 else document
                    )
                    
                    sources.append(source)
            
            return sources
            
        except Exception as e:
            print(f"Error searching documents in ChromaDB: {e}")
            return []
    
    def delete_document(self, filename: str) -> bool:
        """
        Delete all chunks associated with a filename.
        
        Args:
            filename: Name of the file to delete
            
        Returns:
            True if deletion was successful
        """
        try:
            # Get all chunks for this filename
            results = self.collection.get(
                where={"filename": filename},
                include=["metadatas"]
            )
            
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                return True
            
            return False
            
        except Exception as e:
            print(f"Error deleting document from ChromaDB: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the document collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.collection.name,
                "db_path": self.db_path
            }
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {"total_chunks": 0, "error": str(e)}
    
    def reset_collection(self) -> bool:
        """
        Reset the entire document collection.
        
        Returns:
            True if reset was successful
        """
        try:
            self.client.delete_collection(name="research_documents")
            self.collection = self.client.get_or_create_collection(
                name="research_documents",
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            return True
        except Exception as e:
            print(f"Error resetting collection: {e}")
            return False


# Global instance
chroma_manager = ChromaManager()