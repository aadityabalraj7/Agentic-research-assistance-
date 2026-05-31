"""
LangChain tool for Retrieval-Augmented Generation (RAG) using ChromaDB.
"""

from typing import List, Optional
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from backend.vectorstore.chroma_manager import chroma_manager
from backend.models.schemas import DocumentSource


class RagToolInput(BaseModel):
    """Input for the RAG tool."""
    query: str = Field(description="The search query to find relevant documents")
    max_results: int = Field(default=5, description="Maximum number of results to return")


class RagTool(BaseTool):
    """Tool for retrieving relevant documents from the vector store."""
    
    name: str = "document_retrieval"
    description: str = """
    Useful for searching and retrieving relevant information from uploaded documents.
    Use this when the user's question might be answered by information in the uploaded PDF, DOCX, or TXT files.
    """
    args_schema: type[BaseModel] = RagToolInput
    
    def _run(
        self, 
        query: str, 
        max_results: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[DocumentSource]:
        """Use the tool to search for relevant documents."""
        try:
            # Search ChromaDB for relevant documents
            sources = chroma_manager.search_documents(query, n_results=max_results)
            return sources
        except Exception as e:
            return [DocumentSource(
                filename="Error",
                chunk_id="error",
                content_preview=f"Error searching documents: {str(e)}"
            )]
    
    async def _arun(
        self, 
        query: str, 
        max_results: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[DocumentSource]:
        """Async version of the tool (not implemented)."""
        raise NotImplementedError("Async version not implemented")


# Global instance
rag_tool = RagTool()