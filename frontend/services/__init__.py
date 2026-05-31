"""
Service clients for communicating with the Agentic Research Assistant API.
"""

import requests
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import streamlit as st


class APIClient:
    """Client for communicating with the backend API."""
    
    def __init__(self, base_url: str = "http://localhost:8000/api"):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an HTTP request to the API."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.request(method, url, timeout=30, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse API response: {str(e)}"}
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        # Health check is at root level, not under /api
        try:
            response = requests.get(f"{self.base_url.replace('/api', '')}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"Health check failed: {str(e)}"}
    
    def chat(self, message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Send a chat message."""
        payload = {"message": message}
        if session_id:
            payload["session_id"] = session_id
        return self._make_request("POST", "/chat", json=payload)
    
    def research(self, query: str, use_documents: bool = True, use_web: bool = True, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Conduct deep research."""
        payload = {
            "query": query,
            "use_documents": use_documents,
            "use_web": use_web
        }
        if session_id:
            payload["session_id"] = session_id
        return self._make_request("POST", "/research", json=payload)
    
    def upload_file(self, file_data: bytes, filename: str, content_type: str) -> Dict[str, Any]:
        """Upload a file."""
        files = {"file": (filename, file_data, content_type)}
        # For file uploads, we need to use requests directly
        try:
            response = requests.post(f"{self.base_url}/upload", files=files, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"File upload failed: {str(e)}"}
    
    def get_history(self, session_id: str) -> Dict[str, Any]:
        """Get chat history."""
        return self._make_request("GET", f"/history/{session_id}")
    
    def get_memory(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return self._make_request("GET", "/memory")
    
    def export_pdf(self, content: str, title: str = "Research Report") -> Dict[str, Any]:
        """Export report as PDF."""
        payload = {"content": content, "title": title, "format": "pdf"}
        return self._make_request("POST", "/export/pdf", json=payload)
    
    def export_docx(self, content: str, title: str = "Research Report") -> Dict[str, Any]:
        """Export report as DOCX."""
        payload = {"content": content, "title": title, "format": "docx"}
        return self._make_request("POST", "/export/docx", json=payload)


# Global API client instance
api_client = APIClient()


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{int(datetime.now().timestamp())}"
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "research_in_progress" not in st.session_state:
        st.session_state.research_in_progress = False
    
    if "api_client" not in st.session_state:
        st.session_state.api_client = api_client


def get_api_client() -> APIClient:
    """Get the API client from session state."""
    if "api_client" not in st.session_state:
        st.session_state.api_client = APIClient()
    return st.session_state.api_client