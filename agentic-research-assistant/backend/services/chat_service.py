"""
Chat service that integrates with the LangGraph research workflow.
"""

from typing import List, Dict, Any, Optional
import time
from datetime import datetime

from backend.graph.research_graph import research_graph
from backend.memory.memory_store import memory_store
from backend.models.schemas import ChatSession, ChatMessage, ResearchState


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
        
        # Get chat history for context
        chat_history = self._get_chat_history(session_id)
        
        # Initialize research state
        initial_state: ResearchState = {
            "query": message,
            "use_documents": use_documents,
            "use_web": use_web,
            "chat_history": chat_history,
            "research_plan": {},
            "document_information": "",
            "web_information": "",
            "combined_information": "",
            "document_sources": [],
            "web_sources": [],
            "verified_information": "",
            "verification_result": {},
            "final_response": "",
            "research_report": "",
            "agent_trace": [],
            "current_step": "initialized",
            "success": False,
            "error": None
        }
        
        try:
            # Execute the research workflow
            final_state = research_graph.invoke(initial_state)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Prepare response
            response = {
                "session_id": session_id,
                "message": message,
                "response": final_state.get("final_response", "No response generated"),
                "research_report": final_state.get("research_report", ""),
                "citations": self._extract_citations(final_state),
                "sources": self._get_all_sources(final_state),
                "agent_trace": final_state.get("agent_trace", []),
                "processing_time": processing_time,
                "success": final_state.get("success", False),
                "research_plan": final_state.get("research_plan", {}),
                "verified_information": final_state.get("verified_information", ""),
                "error": final_state.get("error")
            }
            
            # Save to chat history
            self._save_chat_exchange(session_id, message, response)
            
            # Save to memory
            self._save_to_memory(session_id, message, response)
            
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
    
    def _get_chat_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve chat history for a session."""
        try:
            history_result = memory_store.get_chat_session(session_id)
            if history_result and history_result.messages:
                # Convert ChatMessage objects to dictionaries
                return [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat() if hasattr(msg.timestamp, 'isoformat') else str(msg.timestamp)
                    }
                    for msg in history_result.messages[-10:]  # Last 10 messages
                ]
            return []
        except:
            return []
    
    def _save_chat_exchange(self, session_id: str, user_message: str, assistant_response: Dict[str, Any]):
        """Save chat exchange to memory."""
        try:
            # Create chat session if it doesn't exist
            user_msg = ChatMessage(
                id="",
                role="user",
                content=user_message,
                timestamp=datetime.now(),
                citations=[]
            )
            
            assistant_msg = ChatMessage(
                id="",
                role="assistant",
                content=assistant_response.get("response", ""),
                timestamp=datetime.now(),
                citations=assistant_response.get("citations", [])
            )
            
            # Get existing session or create new one
            existing_session = memory_store.get_chat_session(session_id)
            if existing_session:
                # Append to existing session
                updated_messages = existing_session.messages + [user_msg, assistant_msg]
                updated_session = ChatSession(
                    id=session_id,
                    messages=updated_messages,
                    created_at=existing_session.created_at,
                    updated_at=datetime.now()
                )
            else:
                # Create new session
                updated_session = ChatSession(
                    id=session_id,
                    messages=[user_msg, assistant_msg],
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
            
            memory_store.save_chat_session(updated_session)
        except Exception as e:
            print(f"Failed to save chat exchange: {e}")
    
    def _save_to_memory(self, session_id: str, user_message: str, assistant_response: Dict[str, Any]):
        """Save interaction to long-term memory."""
        try:
            # Save as a memory entry for future reference
            memory_entry = {
                "session_id": session_id,
                "user_message": user_message,
                "assistant_response": assistant_response.get("response", ""),
                "timestamp": datetime.now().isoformat(),
                "topic": self._extract_topic(user_message),
                "success": assistant_response.get("success", False)
            }
            
            # This would go into a more sophisticated memory system
            # For now, we'll just note that this happened
            pass
        except Exception as e:
            print(f"Failed to save to memory: {e}")
    
    def _extract_citations(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract citations from the research state."""
        citations = []
        
        # Extract from document sources
        for source in state.get("document_sources", []):
            if isinstance(source, dict):
                citations.append({
                    "type": "document",
                    "filename": source.get("filename", "Unknown"),
                    "page_number": source.get("page_number"),
                    "preview": source.get("content_preview", "")[:100] + "..." if len(source.get("content_preview", "")) > 100 else source.get("content_preview", "")
                })
        
        # Extract from web sources
        for source in state.get("web_sources", []):
            if isinstance(source, dict):
                citations.append({
                    "type": "web",
                    "title": source.get("filename", "Unknown Source"),
                    "url": source.get("chunk_id", ""),
                    "preview": source.get("content_preview", "")[:100] + "..." if len(source.get("content_preview", "")) > 100 else source.get("content_preview", "")
                })
        
        return citations
    
    def _get_all_sources(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get all sources from the research state."""
        sources = []
        
        # Document sources
        for source in state.get("document_sources", []):
            if isinstance(source, dict):
                sources.append({
                    "type": "document",
                    "filename": source.get("filename", "Unknown"),
                    "page_number": source.get("page_number"),
                    "content_preview": source.get("content_preview", ""),
                    "chunk_id": source.get("chunk_id", "")
                })
        
        # Web sources
        for source in state.get("web_sources", []):
            if isinstance(source, dict):
                sources.append({
                    "type": "web",
                    "title": source.get("filename", "Unknown Source"),
                    "url": source.get("chunk_id", ""),
                    "content_preview": source.get("content_preview", ""),
                    "chunk_id": source.get("chunk_id", "")
                })
        
        return sources
    
    def _extract_topic(self, message: str) -> str:
        """Extract a topic from the user message for memory categorization."""
        # Simple topic extraction - in practice, this could use NLP
        words = message.lower().split()
        # Remove common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "shall", "can", "what", "when", "where", "why", "how", "who", "which"}
        meaningful_words = [w for w in words if w not in stop_words and len(w) > 2]
        
        if meaningful_words:
            return " ".join(meaningful_words[:5])  # First 5 meaningful words
        return "general_query"


# Global instance
chat_service = ChatService()