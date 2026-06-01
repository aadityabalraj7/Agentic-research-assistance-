"""
Chat service that integrates with the research workflow.
"""

from typing import List, Dict, Any, Optional
import time
from datetime import datetime

from backend.memory.memory_store import memory_store
from backend.models.schemas import ChatSession, ChatMessage


class ChatService:
    """Service for handling chat interactions with the research assistant."""
    
    def __init__(self):
        """Initialize the chat service."""
        pass
    
    def process_message(self, 
                       message: str, 
                       session_id: str = None,
                       use_documents: bool = True,
                       use_web: bool = True) -> Dict[str, Any]:
        """
        Process a chat message through the research workflow.
        
        Args:
            message: User's message/question
            session_id: Session identifier
            use_documents: Whether to search uploaded documents
            use_web: Whether to search the web
            
        Returns:
            Dictionary containing response and metadata
        """
        start_time = time.time()
        
        # Generate session ID if not provided
        if not session_id:
            session_id = f"session_{int(time.time())}"
        
        try:
            # Import here to avoid circular imports and eager LLM initialization
            from backend.agents.supervisor_agent import supervisor_agent
            
            # Execute the research workflow through supervisor agent
            result = supervisor_agent.conduct_research(
                query=message,
                use_documents=use_documents,
                use_web=use_web,
                chat_history=[]
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Prepare response
            response = {
                "session_id": session_id,
                "message": message,
                "response": result.get("final_response", "No response generated"),
                "research_report": result.get("research_report", ""),
                "citations": [],
                "sources": [],
                "agent_trace": result.get("steps", []),
                "processing_time": processing_time,
                "success": result.get("success", False),
                "research_plan": result.get("plan", {}),
                "verified_information": result.get("verified_information", ""),
                "error": result.get("error")
            }
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "session_id": session_id,
                "message": message,
                "response": f"I encountered an error while processing your request: {str(e)}",
                "citations": [],
                "sources": [],
                "agent_trace": [],
                "processing_time": processing_time,
                "success": False,
                "error": str(e)
            }


# Global instance - created when first accessed
_chat_service_instance = None

def get_chat_service():
    """Get or create the chat service instance."""
    global _chat_service_instance
    if _chat_service_instance is None:
        _chat_service_instance = ChatService()
    return _chat_service_instance

chat_service = property(lambda self: get_chat_service())