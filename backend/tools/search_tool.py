"""
LangChain tool for web search using Tavily API.
"""

import os
from typing import List, Optional
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from pydantic import BaseModel, Field
import requests

from backend.models.schemas import DocumentSource


class WebSearchToolInput(BaseModel):
    """Input for the web search tool."""
    query: str = Field(description="The search query to find information on the web")
    max_results: int = Field(default=5, description="Maximum number of results to return")


class WebSearchTool(BaseTool):
    """Tool for searching the web using Tavily API."""
    
    name: str = "web_search"
    description: str = """
    Useful for searching the internet for current information, news, or general knowledge.
    Use this when the user's question requires up-to-date information not likely to be in uploaded documents.
    """
    args_schema: type[BaseModel] = WebSearchToolInput
    
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY environment variable is required")
        self.base_url = "https://api.tavily.com/search"
    
    def _run(
        self, 
        query: str, 
        max_results: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[DocumentSource]:
        """Use the tool to search the web."""
        try:
            if not self.api_key:
                return [DocumentSource(
                    filename="Error",
                    chunk_id="error",
                    content_preview="Tavily API key not configured"
                )]
            
            payload = {
                "api_key": self.api_key,
                "query": query,
                "max_results": max_results,
                "include_domains": [],
                "exclude_domains": [],
                "search_depth": "basic",
                "include_answer": False,
                "include_raw_content": False,
                "include_images": False
            }
            
            response = requests.post(self.base_url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            sources = []
            
            for result in data.get("results", []):
                source = DocumentSource(
                    filename=result.get("title", "Web Result"),
                    chunk_id=result.get("url", ""),
                    content_preview=result.get("content", "")[:200] + "..." if len(result.get("content", "")) > 200 else result.get("content", ""),
                )
                sources.append(source)
            
            return sources
            
        except Exception as e:
            return [DocumentSource(
                filename="Error",
                chunk_id="error",
                content_preview=f"Error searching web: {str(e)}"
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
web_search_tool = WebSearchTool()