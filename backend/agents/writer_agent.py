"""
Writer Agent for the Agentic Research Assistant.
Responsible for generating final answers and research reports with citations.
"""

from typing import List, Dict, Any
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool

from backend.tools.rag_tool import rag_tool
from backend.tools.search_tool import web_search_tool
from backend.tools.memory_tool import memory_tool
from backend.models.schemas import DocumentSource


class WriterAgent:
    """Agent responsible for generating final answers and research reports."""
    
    def __init__(self, model_name: str = "deepseek-chat"):
        """
        Initialize the Writer Agent.
        
        Args:
            model_name: Name of the LLM to use via OpenRouter
        """
        self.model_name = model_name
        self.tools = [rag_tool, web_search_tool, memory_tool]
        self.agent_executor = self._create_agent()
    
    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent executor."""
        # Initialize LLM with OpenRouter
        llm = ChatOpenAI(
            model=self.model_name,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0.3,  # Slightly higher for more natural language
            max_tokens=3000
        )
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a Writer Agent in an Agentic Research Assistant system.
                Your role is to generate comprehensive, well-structured answers and research reports.
                
                When generating a response, you should:
                1. Synthesize information from verified sources into a coherent narrative
                2. Structure the response appropriately (executive summary, key findings, detailed analysis, etc.)
                3. Include proper citations for all factual claims using the format:
                   - Document: [Filename.pdf | Page X]
                   - Web: [Source 1] with URL references in references section
                4. Maintain an objective, analytical tone
                5. Acknowledge limitations or uncertainties in the information
                6. Format the response for readability with clear sections
                
                Your output should be suitable for professional consumption and include:
                - Clear, direct answers to the research question
                - Supporting evidence with citations
                - Logical flow and organization
                - Proper attribution of sources
                """
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create agent
        agent = create_openai_functions_agent(llm, self.tools, prompt)
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,
            early_stopping_method="generate"
        )
        
        return agent_executor
    
    def generate_response(self, 
                         query: str, 
                         verified_information: str, 
                         sources: List[DocumentSource] = None,
                         chat_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Generate a final response based on verified information.
        
        Args:
            query: The original research question
            verified_information: Information that has been fact-checked
            sources: List of sources to cite
            chat_history: Previous conversation history
            
        Returns:
            Dictionary containing the generated response and metadata
        """
        try:
            # Prepare source information for citation
            source_info = ""
            if sources:
                source_info = "\n\nAvailable sources for citation:\n"
                for i, source in enumerate(sources, 1):
                    if hasattr(source, 'filename'):
                        source_info += f"[{i}] {source.filename}"
                        if hasattr(source, 'page_number') and source.page_number:
                            source_info += f" | Page {source.page_number}"
                        source_info += "\n"
                    else:
                        source_info += f"[{i}] {source.get('filename', 'Unknown source')}\n"
            
            # Prepare input for the agent
            agent_input = {
                "input": f"""Based on the verified information below, generate a comprehensive response to the research question.
                
Research Question: {query}

Verified Information:
{verified_information}
{source_info}

Please provide a well-structured answer with proper citations.""",
                "chat_history": chat_history or []
            }
            
            # Execute the agent
            result = self.agent_executor.invoke(agent_input)
            
            return {
                "query": query,
                "response": result.get("output", ""),
                "sources": sources,
                "tool_used": "writer",
                "success": True
            }
            
        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "response": "",
                "sources": sources,
                "tool_used": "writer",
                "success": False
            }
    
    def generate_research_report(self, 
                                query: str, 
                                verified_information: str, 
                                sources: List[DocumentSource] = None,
                                research_plan: Dict = None,
                                chat_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Generate a formal research report.
        
        Args:
            query: The original research question
            verified_information: Information that has been fact-checked
            sources: List of sources to cite
            research_plan: The original research plan
            chat_history: Previous conversation history
            
        Returns:
            Dictionary containing the research report and metadata
        """
        try:
            # Prepare source information for citation
            source_info = ""
            if sources:
                source_info = "\n\nAvailable sources for citation:\n"
                for i, source in enumerate(sources, 1):
                    if hasattr(source, 'filename'):
                        source_info += f"[{i}] {source.filename}"
                        if hasattr(source, 'page_number') and source.page_number:
                            source_info += f" | Page {source.page_number}"
                        source_info += "\n"
                    else:
                        source_info += f"[{i}] {source.get('filename', 'Unknown source')}\n"
            
            # Prepare plan information
            plan_info = ""
            if research_plan:
                plan_info = f"\n\nOriginal Research Plan:\n{research_plan.get('full_plan', '')}"
            
            # Prepare input for the agent
            agent_input = {
                "input": f"""Generate a formal research report based on the verified information.
                
Research Question: {query}

Verified Information:
{verified_information}
{source_info}
{plan_info}

Please structure the report with the following sections:
1. Executive Summary
2. Key Findings
3. Detailed Analysis
4. Opportunities (if applicable)
5. Risks (if applicable)
6. Conclusion
7. References

Include proper citations throughout using the format [Source X] and provide a references section with full source details.""",
                "chat_history": chat_history or []
            }
            
            # Execute the agent
            result = self.agent_executor.invoke(agent_input)
            
            return {
                "query": query,
                "report": result.get("output", ""),
                "sources": sources,
                "tool_used": "writer",
                "success": True
            }
            
        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "report": "",
                "sources": sources,
                "tool_used": "writer",
                "success": False
            }


# Global instance
writer_agent = WriterAgent()