"""
Planner Agent for the Agentic Research Assistant.
Responsible for understanding research objectives and creating research plans.
"""

from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from backend.tools.rag_tool import rag_tool
from backend.tools.search_tool import web_search_tool
from backend.tools.memory_tool import memory_tool


class PlannerAgent:
    """Agent responsible for planning research tasks."""
    
    def __init__(self):
        self._llm = None
    
    @property
    def llm(self):
        """Lazy load the LLM."""
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
                            ChatGeneration(message=AIMessage(content="LLM not configured"))
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
    
    def create_research_plan(self, query: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        return {
            "objective": query,
            "subtasks": ["Analyze the research question", "Search for relevant information", 
                        "Gather and verify sources", "Synthesize findings", "Generate structured report"],
            "tools_needed": ["document_retrieval", "web_search", "memory"],
            "estimated_time": 5,
            "full_plan": f"Research plan for: {query}"
        }


planner_agent = PlannerAgent()