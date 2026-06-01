"""
LangGraph workflow for the Agentic Research Assistant.
Defines the multi-agent research workflow with conditional routing.
"""

from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict

# State structure for our workflow
class ResearchState(TypedDict):
    """State for the research workflow."""
    query: str
    use_documents: bool
    use_web: bool
    chat_history: List[Dict[str, Any]]
    research_plan: Dict[str, Any]
    document_information: str
    web_information: str
    combined_information: str
    document_sources: List[Dict[str, Any]]
    web_sources: List[Dict[str, Any]]
    verified_information: str
    verification_result: Dict[str, Any]
    final_response: str
    research_report: str
    agent_trace: List[Dict[str, Any]]
    current_step: str
    success: bool
    error: Optional[str]


def create_research_graph():
    """Create and compile the LangGraph research workflow."""
    # For now, use supervisor agent directly without complex graph
    # This simplifies and avoids LangGraph compatibility issues
    
    def run_workflow(state: ResearchState) -> ResearchState:
        """Simple workflow using supervisor agent."""
        from backend.agents.supervisor_agent import supervisor_agent
        
        result = supervisor_agent.conduct_research(
            query=state["query"],
            use_documents=state["use_documents"],
            use_web=state["use_web"],
            chat_history=state["chat_history"]
        )
        
        return {
            "query": state["query"],
            "use_documents": state["use_documents"],
            "use_web": state["use_web"],
            "chat_history": state["chat_history"],
            "research_plan": result.get("plan", {}),
            "document_information": result.get("information", ""),
            "web_information": "",
            "combined_information": result.get("verified_information", ""),
            "document_sources": result.get("sources", []),
            "web_sources": [],
            "verified_information": result.get("verified_information", ""),
            "verification_result": {},
            "final_response": result.get("final_response", ""),
            "research_report": result.get("research_report", ""),
            "agent_trace": result.get("steps", []),
            "current_step": "completed",
            "success": result.get("success", False),
            "error": result.get("error")
        }
    
    return run_workflow


# Global instance
research_graph = create_research_graph()