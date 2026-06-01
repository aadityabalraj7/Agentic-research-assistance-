"""
Writer Agent for the Agentic Research Assistant.
"""

from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from backend.tools.rag_tool import rag_tool
from backend.tools.search_tool import web_search_tool
from backend.models.schemas import DocumentSource


class WriterAgent:
    """Agent responsible for generating final answers and research reports."""
    
    def __init__(self):
        self._llm = None
    
    @property
    def llm(self):
        if self._llm is None:
            load_dotenv('.env')
            api_key = os.getenv("OPENROUTER_API_KEY")
            model_name = os.getenv("MODEL_NAME") or "nvidia/nemotron-3-super-120b-a12b:free"
            
            if not api_key:
                self._llm = None
            else:
                self._llm = ChatOpenAI(
                    model=model_name,
                    openai_api_base="https://openrouter.ai/api/v1",
                    openai_api_key=api_key,
                    temperature=0.3,
                    max_tokens=3000
                )
        return self._llm
    
    def generate_response(self, query: str, verified_information: str, 
                        sources: List[DocumentSource] = None, chat_history: List[Dict] = None) -> Dict[str, Any]:
        try:
            if self.llm and verified_information:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a research assistant. Provide a helpful, well-structured answer based on the provided information."),
                    ("human", f"Question: {query}\n\nInformation: {verified_information}")
                ])
                chain = prompt | self.llm
                result = chain.invoke({})
                response_text = result.content if hasattr(result, 'content') else str(result)
            else:
                response_text = f"Based on my research on '{query}':\n\n{verified_information or 'No information gathered.'}"
            
            return {"query": query, "response": response_text, "sources": sources, "tool_used": "writer", "success": True}
        except Exception as e:
            return {"query": query, "error": str(e), "response": f"Error: {str(e)}", "sources": sources, "tool_used": "writer", "success": False}
    
    def generate_research_report(self, query: str, verified_information: str,
                                sources: List[DocumentSource] = None, research_plan: Dict = None,
                                chat_history: List[Dict] = None) -> Dict[str, Any]:
        try:
            if self.llm and verified_information:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "Write a research report with: Executive Summary, Key Findings, Analysis, Conclusion, References."),
                    ("human", f"Topic: {query}\n\nInformation: {verified_information}")
                ])
                chain = prompt | self.llm
                result = chain.invoke({})
                report_content = result.content if hasattr(result, 'content') else str(result)
            else:
                report_content = f"# Report: {query}\n\n## Summary\n{verified_information or 'No information gathered.'}"
            
            return {"query": query, "report": report_content, "sources": sources, "tool_used": "writer", "success": True}
        except Exception as e:
            return {"query": query, "error": str(e), "report": f"Error: {str(e)}", "sources": sources, "tool_used": "writer", "success": False}


writer_agent = WriterAgent()