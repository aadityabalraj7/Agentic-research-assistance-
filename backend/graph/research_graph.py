"""
LangGraph workflow for the Agentic Research Assistant.
Defines the multi-agent research workflow with conditional routing.
"""

from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

from backend.agents.planner_agent import planner_agent
from backend.agents.retrieval_agent import retrieval_agent
from backend.agents.web_agent import web_research_agent
from backend.agents.verifier_agent import verifier_agent
from backend.agents.writer_agent import writer_agent
from backend.agents.supervisor_agent import supervisor_agent


# Define the state structure for our graph
class ResearchState(TypedDict):
    """State for the research workflow."""
    query: str
    use_documents: bool
    use_web: bool
    chat_history: List[Dict[str, Any]]
    
    # Planning phase
    research_plan: Dict[str, Any]
    
    # Information gathering phase
    document_information: str
    web_information: str
    combined_information: str
    document_sources: List[Dict[str, Any]]
    web_sources: List[Dict[str, Any]]
    
    # Verification phase
    verified_information: str
    verification_result: Dict[str, Any]
    
    # Response generation phase
    final_response: str
    research_report: str
    
    # Metadata
    agent_trace: List[Dict[str, Any]]
    current_step: str
    success: bool
    error: Optional[str]


def planner_node(state: ResearchState) -> ResearchState:
    """Planner node: Creates research plan."""
    print("Executing Planner Node...")
    
    try:
        plan_result = planner_agent.create_research_plan(
            state["query"], 
            state["chat_history"]
        )
        
        # Update state
        state["research_plan"] = plan_result
        state["current_step"] = "planning_completed"
        
        # Add to agent trace
        state["agent_trace"].append({
            "agent": "Planner",
            "action": "create_research_plan",
            "status": "completed" if "error" not in plan_result else "failed",
            "timestamp": str(None),  # Would use actual timestamp
            "details": plan_result
        })
        
    except Exception as e:
        state["error"] = f"Planner node failed: {str(e)}"
        state["current_step"] = "planning_failed"
        state["agent_trace"].append({
            "agent": "Planner",
            "action": "create_research_plan",
            "status": "failed",
            "timestamp": str(None),
            "details": {"error": str(e)}
        })
    
    return state


def routing_node(state: ResearchState) -> str:
    """Routing node: Determines which information gathering paths to take."""
    print("Executing Routing Node...")
    
    # Determine next steps based on configuration
    next_steps = []
    
    if state["use_documents"]:
        next_steps.append("retrieval")
    
    if state["use_web"]:
        next_steps.append("web_research")
    
    # If neither is selected, go directly to verification
    if not next_steps:
        return "verification"
    
    # For simplicity, we'll go through both sequentially if both are selected
    # In a more complex implementation, this could be parallel
    return next_steps[0]  # Return first step


def retrieval_node(state: ResearchState) -> ResearchState:
    """Retrieval node: Searches documents for information."""
    print("Executing Retrieval Node...")
    
    try:
        retrieval_result = retrieval_agent.retrieve_information(
            state["query"], 
            state["chat_history"]
        )
        
        # Update state
        state["document_information"] = retrieval_result.get("information", "")
        state["document_sources"] = retrieval_result.get("sources", [])
        state["current_step"] = "retrieval_completed"
        
        # Add to agent trace
        state["agent_trace"].append({
            "agent": "Retrieval",
            "action": "retrieve_information",
            "status": "completed" if retrieval_result.get("success") else "failed",
            "timestamp": str(None),
            "details": retrieval_result
        })
        
    except Exception as e:
        state["error"] = f"Retrieval node failed: {str(e)}"
        state["current_step"] = "retrieval_failed"
        state["agent_trace"].append({
            "agent": "Retrieval",
            "action": "retrieve_information",
            "status": "failed",
            "timestamp": str(None),
            "details": {"error": str(e)}
        })
    
    return state


def web_research_node(state: ResearchState) -> ResearchState:
    """Web research node: Searches the web for information."""
    print("Executing Web Research Node...")
    
    try:
        web_result = web_research_agent.research_information(
            state["query"], 
            state["chat_history"]
        )
        
        # Update state
        state["web_information"] = web_result.get("information", "")
        state["web_sources"] = web_result.get("sources", [])
        state["current_step"] = "web_research_completed"
        
        # Add to agent trace
        state["agent_trace"].append({
            "agent": "Web Research",
            "action": "research_information",
            "status": "completed" if web_result.get("success") else "failed",
            "timestamp": str(None),
            "details": web_result
        })
        
    except Exception as e:
        state["error"] = f"Web research node failed: {str(e)}"
        state["current_step"] = "web_research_failed"
        state["agent_trace"].append({
            "agent": "Web Research",
            "action": "research_information",
            "status": "failed",
            "timestamp": str(None),
            "details": {"error": str(e)}
        })
    
    return state


def combination_node(state: ResearchState) -> ResearchState:
    """Combination node: Combines information from different sources."""
    print("Executing Combination Node...")
    
    try:
        # Combine information from documents and web
        doc_info = state.get("document_information", "")
        web_info = state.get("web_information", "")
        
        combined_parts = []
        if doc_info.strip():
            combined_parts.append(f"Information from Documents:\n{doc_info}")
        if web_info.strip():
            combined_parts.append(f"Information from Web:\n{web_info}")
        
        state["combined_information"] = "\n\n".join(combined_parts)
        state["current_step"] = "combination_completed"
        
        # Add to agent trace
        state["agent_trace"].append({
            "agent": "System",
            "action": "combine_information",
            "status": "completed",
            "timestamp": str(None),
            "details": {
                "document_length": len(state.get("document_information", "")),
                "web_length": len(state.get("web_information", "")),
                "combined_length": len(state["combined_information"])
            }
        })
        
    except Exception as e:
        state["error"] = f"Combination node failed: {str(e)}"
        state["current_step"] = "combination_failed"
        state["agent_trace"].append({
            "agent": "System",
            "action": "combine_information",
            "status": "failed",
            "timestamp": str(None),
            "details": {"error": str(e)}
        })
    
    return state


def verification_node(state: ResearchState) -> ResearchState:
    """Verification node: Fact-checks the combined information."""
    print("Executing Verification Node...")
    
    try:
        # Determine what information to verify
        info_to_verify = state.get("combined_information", "")
        if not info_to_verify.strip():
            info_to_verify = "No information was retrieved from the selected sources."
        
        # All sources for verification
        all_sources = state.get("document_sources", []) + state.get("web_sources", [])
        
        verification_result = verifier_agent.verify_information(
            info_to_verify,
            sources=all_sources,
            chat_history=state["chat_history"]
        )
        
        # Update state
        state["verified_information"] = verification_result.get("verification_result", "")
        state["verification_result"] = verification_result
        state["current_step"] = "verification_completed"
        
        # Add to agent trace
        state["agent_trace"].append({
            "agent": "Verifier",
            "action": "verify_information",
            "status": "completed" if verification_result.get("success") else "failed",
            "timestamp": str(None),
            "details": verification_result
        })
        
    except Exception as e:
        state["error"] = f"Verification node failed: {str(e)}"
        state["current_step"] = "verification_failed"
        state["agent_trace"].append({
            "agent": "Verifier",
            "action": "verify_information",
            "status": "failed",
            "timestamp": str(None),
            "details": {"error": str(e)}
        })
    
    return state


def response_generation_node(state: ResearchState) -> ResearchState:
    """Response generation node: Creates the final response."""
    print("Executing Response Generation Node...")
    
    try:
        # Generate final response
        response_result = writer_agent.generate_response(
            state["query"],
            state["verified_information"],
            sources=[],  # Would use actual sources
            chat_history=state["chat_history"]
        )
        
        # Update state
        state["final_response"] = response_result.get("response", "")
        state["current_step"] = "response_generation_completed"
        
        # Add to agent trace
        state["agent_trace"].append({
            "agent": "Writer",
            "action": "generate_response",
            "status": "completed" if response_result.get("success") else "failed",
            "timestamp": str(None),
            "details": response_result
        })
        
    except Exception as e:
        state["error"] = f"Response generation node failed: {str(e)}"
        state["current_step"] = "response_generation_failed"
        state["agent_trace"].append({
            "agent": "Writer",
            "action": "generate_response",
            "status": "failed",
            "timestamp": str(None),
            "details": {"error": str(e)}
        })
    
    return state


def report_generation_node(state: ResearchState) -> ResearchState:
    """Report generation node: Creates a formal research report."""
    print("Executing Report Generation Node...")
    
    try:
        # Generate research report
        report_result = writer_agent.generate_research_report(
            state["query"],
            state["verified_information"],
            sources=[],  # Would use actual sources
            research_plan=state["research_plan"],
            chat_history=state["chat_history"]
        )
        
        # Update state
        state["research_report"] = report_result.get("report", "")
        state["current_step"] = "report_generation_completed"
        
        # Add to agent trace
        state["agent_trace"].append({
            "agent": "Writer",
            "action": "generate_research_report",
            "status": "completed" if report_result.get("success") else "failed",
            "timestamp": str(None),
            "details": report_result
        })
        
    except Exception as e:
        state["error"] = f"Report generation node failed: {str(e)}"
        state["current_step"] = "report_generation_failed"
        state["agent_trace"].append({
            "agent": "Writer",
            "action": "generate_research_report",
            "status": "failed",
            "timestamp": str(None),
            "details": {"error": str(e)}
        })
    
    return state


def success_check_node(state: ResearchState) -> ResearchState:
    """Success check node: Determines if the research was successful."""
    print("Executing Success Check Node...")
    
    # Check if any critical steps failed
    critical_steps_failed = any(
        "failed" in step.get("status", "") 
        for step in state["agent_trace"][-5:]  # Check last few steps
        if step.get("status")
    )
    
    state["success"] = not critical_steps_failed and not bool(state.get("error"))
    state["current_step"] = "success_check_completed"
    
    # Add to agent trace
    state["agent_trace"].append({
        "agent": "System",
        "action": "success_check",
        "status": "completed",
        "timestamp": str(None),
        "details": {"success": state["success"]}
    })
    
    return state


def should_continue(state: ResearchState) -> str:
    """Determine if the workflow should continue or end."""
    if state.get("error"):
        return "error"
    
    current_step = state.get("current_step", "")
    
    # Define the flow
    if current_step == "planning_completed":
        return "routing"
    elif current_step == "retrieval_completed":
        # Check if we also need web research
        if state["use_web"] and not state.get("web_information"):
            return "web_research"
        else:
            return "combination"
    elif current_step == "web_research_completed":
        return "combination"
    elif current_step == "combination_completed":
        return "verification"
    elif current_step == "verification_completed":
        return "response_generation"
    elif current_step == "response_generation_completed":
        return "report_generation"
    elif current_step == "report_generation_completed":
        return "success_check"
    elif current_step == "success_check_completed":
        return END
    else:
        # Default to end if we're unsure
        return END


def create_research_graph() -> StateGraph:
    """Create and compile the LangGraph research workflow."""
    
    # Create the state graph
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("routing", lambda state: state)  # Pass-through for routing
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("web_research", web_research_node)
    workflow.add_node("combination", combination_node)
    workflow.add_node("verification", verification_node)
    workflow.add_node("response_generation", response_generation_node)
    workflow.add_node("report_generation", report_generation_node)
    workflow.add_node("success_check", success_check_node)
    
    # Set entry point
    workflow.set_entry_point("planner")
    
    # Add edges
    workflow.add_edge("planner", "routing")
    workflow.add_conditional_edges(
        "routing",
        lambda state: "retrieval" if state["use_documents"] else ("web_research" if state["use_web"] else "combination"),
        {
            "retrieval": "retrieval",
            "web_research": "web_research",
            "combination": "combination"
        }
    )
    workflow.add_conditional_edges(
        "retrieval",
        lambda state: "web_research" if state["use_web"] and not state.get("web_information") else "combination",
        {
            "web_research": "web_research",
            "combination": "combination"
        }
    )
    workflow.add_edge("web_research", "combination")
    workflow.add_edge("combination", "verification")
    workflow.add_edge("verification", "response_generation")
    workflow.add_edge("response_generation", "report_generation")
    workflow.add_edge("report_generation", "success_check")
    workflow.add_edge("success_check", END)
    
    # Compile the graph
    return workflow.compile()


# Global instance
research_graph = create_research_graph()