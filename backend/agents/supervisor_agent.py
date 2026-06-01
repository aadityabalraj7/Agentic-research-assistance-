"""
Supervisor Agent for the Agentic Research Assistant.
Responsible for routing tasks, managing workflow, and coordinating agents.
"""

from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

from backend.agents.planner_agent import planner_agent
from backend.agents.retrieval_agent import retrieval_agent
from backend.agents.web_agent import web_research_agent
from backend.agents.verifier_agent import verifier_agent
from backend.agents.writer_agent import writer_agent
from backend.tools.rag_tool import rag_tool
from backend.tools.search_tool import web_search_tool
from backend.tools.memory_tool import memory_tool


class SupervisorAgent:
    """Agent responsible for supervising and coordinating the research workflow."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name
        self.tools = [rag_tool, web_search_tool, memory_tool]
        self._llm = None
    
    @property
    def llm(self):
        """Lazy load the LLM to avoid import-time initialization issues."""
        if self._llm is None:
            load_dotenv('.env')
            api_key = os.getenv("OPENROUTER_API_KEY")
            model_name = os.getenv("MODEL_NAME") or "nvidia/nemotron-3-super-120b-a12b:free"
            if not api_key:
                from langchain_core.language_models.chat_models import BaseChatModel
                from langchain_core.messages import AIMessage
                
                class DummyLLM(BaseChatModel):
                    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                        from langchain_core.outputs import ChatResult, ChatGeneration
                        return ChatResult(generations=[
                            ChatGeneration(message=AIMessage(content="LLM not configured - missing API key"))
                        ])
                    
                    @property
                    def _llm_type(self):
                        return "dummy"
                
                self._llm = DummyLLM()
            else:
                self._llm = ChatOpenAI(
                    model=model_name,
                    openai_api_base="https://openrouter.ai/api/v1",
                    openai_api_key=api_key,
                    temperature=0.1,
                    max_tokens=2000
                )
        return self._llm
    
    def conduct_research(self, 
                        query: str, 
                        use_documents: bool = True,
                        use_web: bool = True,
                        chat_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Conduct comprehensive research using the multi-agent system.
        
        Args:
            query: The research question or objective
            use_documents: Whether to search uploaded documents
            use_web: Whether to search the web
            chat_history: Previous conversation history
            
        Returns:
            Dictionary containing the final research results
        """
        try:
            research_session = {
                "query": query,
                "steps": [],
                "sources": [],
                "verified_information": "",
                "final_response": "",
                "research_report": "",
                "success": False
            }
            
            # Step 1: Planning
            print("Step 1: Creating research plan...")
            plan_result = planner_agent.create_research_plan(query, chat_history)
            research_session["steps"].append({
                "agent": "Planner",
                "action": "create_research_plan",
                "status": "completed" if "error" not in plan_result else "failed",
                "details": plan_result
            })
            
            # Step 2: Information Gathering
            all_information = []
            
            if use_documents:
                print("Step 2a: Retrieving information from documents...")
                retrieval_result = retrieval_agent.retrieve_information(query, chat_history)
                research_session["steps"].append({
                    "agent": "Retrieval",
                    "action": "retrieve_information",
                    "status": "completed" if retrieval_result.get("success") else "failed",
                    "details": retrieval_result
                })
                if retrieval_result.get("success"):
                    all_information.append(retrieval_result.get("information", ""))
            
            if use_web:
                print("Step 2b: Researching information from web...")
                web_result = web_research_agent.research_information(query, chat_history)
                research_session["steps"].append({
                    "agent": "Web Research",
                    "action": "research_information",
                    "status": "completed" if web_result.get("success") else "failed",
                    "details": web_result
                })
                if web_result.get("success"):
                    all_information.append(web_result.get("information", ""))
            
            # Combine information
            combined_information = "\n\n".join(all_information) if all_information else "No information retrieved."
            
            # Step 3: Fact Verification
            print("Step 3: Verifying information...")
            verification_result = verifier_agent.verify_information(
                combined_information,
                sources=[],
                chat_history=chat_history
            )
            research_session["steps"].append({
                "agent": "Verifier",
                "action": "verify_information",
                "status": "completed" if verification_result.get("success") else "failed",
                "details": verification_result
            })
            
            research_session["verified_information"] = verification_result.get("verification_result", combined_information)
            
            # Step 4: Response Generation
            print("Step 4: Generating response...")
            response_result = writer_agent.generate_response(
                query, 
                research_session["verified_information"],
                sources=[],
                chat_history=chat_history
            )
            research_session["steps"].append({
                "agent": "Writer",
                "action": "generate_response",
                "status": "completed" if response_result.get("success") else "failed",
                "details": response_result
            })
            
            research_session["final_response"] = response_result.get("response", "")
            
            # Step 5: Research Report Generation
            print("Step 5: Generating research report...")
            report_result = writer_agent.generate_research_report(
                query, 
                research_session["verified_information"],
                sources=[],
                research_plan=plan_result,
                chat_history=chat_history
            )
            research_session["steps"].append({
                "agent": "Writer",
                "action": "generate_research_report",
                "status": "completed" if report_result.get("success") else "failed",
                "details": report_result
            })
            
            research_session["research_report"] = report_result.get("report", "")
            
            # Determine overall success
            research_session["success"] = (
                "error" not in plan_result and
                (not use_documents or retrieval_result.get("success", True)) and
                (not use_web or web_result.get("success", True))
            )
            
            return research_session
            
        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "steps": [],
                "sources": [],
                "verified_information": "",
                "final_response": f"Error conducting research: {str(e)}",
                "research_report": f"Error generating report: {str(e)}",
                "success": False
            }


# Global instance
supervisor_agent = SupervisorAgent()