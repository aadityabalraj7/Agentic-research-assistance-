"""
API routes for the Agentic Research Assistant.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
from pathlib import Path

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    citations: List[dict] = []
    session_id: str
    agent_trace: List[dict] = []


class ResearchRequest(BaseModel):
    query: str
    use_documents: bool = True
    use_web: bool = True
    session_id: Optional[str] = None


class ResearchResponse(BaseModel):
    report: str
    citations: List[dict] = []
    session_id: str
    sources: List[dict] = []


class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_created: int


# Document upload endpoint
@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document for RAG."""
    try:
        # Validate file type
        allowed_extensions = {".pdf", ".docx", ".txt"}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Read file content
        file_content = await file.read()
        
        # Import here to avoid circular imports
        from backend.services.document_processor import document_processor
        from backend.vectorstore.chroma_manager import chroma_manager
        from backend.models.schemas import DocumentChunk
        
        # Process document
        chunks, metadata_list = document_processor.process_document(file_content, file.filename)
        
        # Add chunks to vector store
        document_chunks = []
        for i, (chunk, metadata) in enumerate(zip(chunks, metadata_list)):
            doc_chunk = DocumentChunk(
                id=metadata["chunk_id"],
                content=chunk,
                metadata=metadata
            )
            document_chunks.append(doc_chunk)
        
        chunk_ids = chroma_manager.add_documents(document_chunks)
        
        return UploadResponse(
            message=f"Document '{file.filename}' uploaded and processed successfully",
            filename=file.filename,
            chunks_created=len(chunks)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Chat endpoint
@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat requests with the research assistant."""
    try:
        from backend.services.chat_service import ChatService
        
        chat_service = ChatService()
        result = chat_service.process_message(
            message=request.message,
            session_id=request.session_id,
            use_documents=True,
            use_web=True
        )
        
        return ChatResponse(
            response=result["response"],
            citations=result.get("citations", []),
            session_id=result["session_id"],
            agent_trace=result.get("agent_trace", [])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Research endpoint
@router.post("/research", response_model=ResearchResponse)
async def conduct_research(request: ResearchRequest):
    """Conduct deep research on a topic."""
    try:
        from backend.services.chat_service import ChatService
        
        chat_service = ChatService()
        result = chat_service.process_message(
            message=request.query,
            session_id=request.session_id,
            use_documents=request.use_documents,
            use_web=request.use_web
        )
        
        return ResearchResponse(
            report=result.get("research_report", result.get("response", "")),
            citations=result.get("citations", []),
            session_id=result["session_id"],
            sources=result.get("sources", [])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# History endpoint
@router.get("/history/{session_id}")
async def get_history(session_id: str):
    """Get chat history for a session."""
    try:
        from backend.memory.memory_store import memory_store
        
        chat_session = memory_store.get_chat_session(session_id)
        if chat_session:
            return {
                "session_id": session_id,
                "history": [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat() if hasattr(msg.timestamp, 'isoformat') else str(msg.timestamp),
                        "citations": []
                    }
                    for msg in chat_session.messages
                ]
            }
        else:
            return {"session_id": session_id, "history": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Memory endpoint
@router.get("/memory")
async def get_memory():
    """Get memory statistics and contents."""
    try:
        from backend.memory.memory_store import memory_store
        
        memory_entries = memory_store.get_memory_entries(limit=1000)
        research_history = memory_store.get_research_history(limit=100)
        
        return {
            "memories": [entry.model_dump() for entry in memory_entries],
            "counts": {
                "memory_entries": len(memory_entries),
                "research_history": len(research_history)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Export endpoints
@router.post("/export/pdf")
async def export_pdf(content: dict):
    """Export research report as PDF."""
    try:
        from backend.exports.export_manager import export_manager
        
        report_content = content.get("content", "")
        title = content.get("title", "Research Report")
        
        pdf_bytes = export_manager.export_to_pdf(report_content, title)
        
        from fastapi.responses import Response
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={title.replace(' ', '_')}.pdf"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export/docx")
async def export_docx(content: dict):
    """Export research report as DOCX."""
    try:
        from backend.exports.export_manager import export_manager
        
        report_content = content.get("content", "")
        title = content.get("title", "Research Report")
        
        docx_bytes = export_manager.export_to_docx(report_content, title)
        
        from fastapi.responses import Response
        return Response(
            content=docx_bytes,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f"attachment; filename={title.replace(' ', '_')}.docx"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))