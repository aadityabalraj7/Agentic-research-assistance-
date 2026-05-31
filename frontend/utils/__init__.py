"""
Utility functions for the Agentic Research Assistant frontend.
"""

import re
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import streamlit as st


def format_citation(source: Dict[str, Any]) -> str:
    """
    Format a source into a citation string.
    
    Args:
        source: Source dictionary with filename, page_number, etc.
        
    Returns:
        Formatted citation string
    """
    filename = source.get("filename", "Unknown Source")
    
    if source.get("page_number") is not None:
        return f"[{filename} | Page {source['page_number']}]"
    elif source.get("section"):
        return f"[{filename} | Section {source['section']}]"
    else:
        return f"[{filename}]"


def extract_citations_from_text(text: str) -> List[str]:
    """
    Extract citation patterns from text.
    
    Args:
        text: Text to search for citations
        
    Returns:
        List of found citations
    """
    # Pattern for [Filename.pdf | Page X] or [Source 1]
    patterns = [
        r'\[[^\]]+\|[^\]]+\]',  # [Filename | Page X] format
        r'\[[^\]]+\]'           # [Source] format
    ]
    
    citations = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        citations.extend(matches)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_citations = []
    for citation in citations:
        if citation not in seen:
            seen.add(citation)
            unique_citations.append(citation)
    
    return unique_citations


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to a maximum length with ellipsis.
    
    Args:
        text: Text to truncate
        max_length: Maximum length before truncation
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def calculate_file_hash(file_content: bytes) -> str:
    """
    Calculate SHA256 hash of file content.
    
    Args:
        file_content: File content as bytes
        
    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(file_content).hexdigest()


def format_timestamp(timestamp: str) -> str:
    """
    Format ISO timestamp for display.
    
    Args:
        timestamp: ISO format timestamp string
        
    Returns:
        Formatted timestamp string
    """
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime("%b %d, %Y at %I:%M %p")
    except:
        return timestamp


def time_ago(timestamp: str) -> str:
    """
    Calculate time ago from timestamp.
    
    Args:
        timestamp: ISO format timestamp string
        
    Returns:
        Human-readable time ago string
    """
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        now = datetime.now()
        diff = now - dt
        
        if diff.days > 0:
            return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        else:
            return "just now"
    except:
        return "unknown time"


def validate_file_type(filename: str) -> bool:
    """
    Validate if file type is supported.
    
    Args:
        filename: Name of the file to validate
        
    Returns:
        True if file type is supported
    """
    allowed_extensions = {'.pdf', '.docx', '.txt'}
    return any(filename.lower().endswith(ext) for ext in allowed_extensions)


def get_file_type_icon(filename: str) -> str:
    """
    Get appropriate icon for file type.
    
    Args:
        filename: Name of the file
        
    Returns:
        Emoji icon for the file type
    """
    if filename.lower().endswith('.pdf'):
        return "📄"
    elif filename.lower().endswith('.docx'):
        return "📝"
    elif filename.lower().endswith('.txt'):
        return "📓"
    else:
        return "📎"


def safe_json_loads(json_string: str, default: Any = None) -> Any:
    """
    Safely load JSON string with fallback.
    
    Args:
        json_string: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON object or default value
    """
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return default


def create_download_link(content: bytes, filename: str, mime_type: str = "application/octet-stream"):
    """
    Create a download link for content.
    
    Args:
        content: Content as bytes
        filename: Download filename
        mime_type: MIME type of content
        
    Returns:
        HTML download link
    """
    import base64
    b64 = base64.b64encode(content).decode()
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}">Download {filename}</a>'
    return href


# Export utility functions
__all__ = [
    "format_citation",
    "extract_citations_from_text",
    "truncate_text",
    "calculate_file_hash",
    "format_timestamp",
    "time_ago",
    "validate_file_type",
    "get_file_type_icon",
    "safe_json_loads",
    "create_download_link"
]