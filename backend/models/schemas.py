"""
Pydantic schemas for the Agentic Research Assistant.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# Document schemas
class DocumentChunk(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any] = {}
    score: Optional[float] = None

class DocumentSource(BaseModel):
    filename: str
    page_number: Optional[int] = None
    chunk_id: str
    content_preview: str

# Chat schemas
class ChatMessage(BaseModel):
    id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    citations: List[DocumentSource] = []

class ChatSession(BaseModel):
    id: str
    messages: List[ChatMessage] = []
    created_at: datetime
    updated_at: datetime

# Research schemas
class ResearchPlan(BaseModel):
    objective: str
    subtasks: List[str] = []
    tools_needed: List[str] = []
    estimated_time: Optional[int] = None  # in minutes

class ResearchResult(BaseModel):
    query: str
    plan: ResearchPlan
    findings: List[Dict[str, Any]] = []
    citations: List[DocumentSource] = []
    report: str
    confidence_score: Optional[float] = None

# Memory schemas
class MemoryEntry(BaseModel):
    id: str
    content: str
    type: str  # "conversation", "research", "preference"
    timestamp: datetime
    metadata: Dict[str, Any] = {}

class UserPreference(BaseModel):
    key: str
    value: Any
    updated_at: datetime

# Agent trace schemas
class AgentStep(BaseModel):
    agent_name: str
    action: str
    timestamp: datetime
    duration_ms: Optional[int] = None
    inputs: Dict[str, Any] = {}
    outputs: Dict[str, Any] = {}
    status: str  # "pending", "running", "completed", "failed"

class AgentTrace(BaseModel):
    session_id: str
    steps: List[AgentStep] = []
    start_time: datetime
    end_time: Optional[datetime] = None

# Export schemas
class ExportRequest(BaseModel):
    content: str
    title: str
    format: str  # "pdf" or "docx"
    include_citations: bool = True

class ExportResponse(BaseModel):
    download_url: str
    filename: str
    expires_at: datetime