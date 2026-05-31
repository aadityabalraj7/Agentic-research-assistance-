"""
Web Research Agent for the Agentic Research Assistant.
Responsible for searching the internet and gathering online evidence.
"""

from typing import List, Dict, Any
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool

from backend.tools.search_tool import web_search_tool


class WebResearchAgent:
    """Agent responsible for researching information from the web."""
    
    def __init__(self, model_name: str = "deepseek-chat"):
        """
        Initialize the Web Research Agent.
        
        Args:
            model_name: Name of the LLM to use via OpenRouter
        """
        self.model_name = model_name
        self.tools = [web_search_tool]
        self.agent_executor = self._create_agent()
    
    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent executor."""
        # Initialize LLM with OpenRouter
        llm = ChatOpenAI(
            model=self.model_name,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0.1,
            max_tokens=2000
        )
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a Web Research Agent in an Agentic Research Assistant system.
                Your role is to search the internet for current, reliable information.
                
                When given a query, you should:
                1. Use the web_search tool to search for relevant information
                2. Evaluate the credibility and relevance of sources
                3. Synthesize information from multiple sources when possible
                4. Provide clear citations for all information retrieved
                5. Note any limitations or conflicts in the information found
                
                Always include citations in your response using the format:
                [Source 1], [Source 2], etc., with corresponding URLs in references
                
                Focus on finding authoritative, up-to-date sources.
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
            max_iterations=3,
            early_stopping_method="generate"
        )
        
        return agent_executor
    
    def research_information(self, query: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Research information from the web for the given query.
        
        Args:
            query: The research question or objective
            chat_history: Previous conversation history
            
        Returns:
            Dictionary containing researched information and sources
        """
        try:
            # Prepare input for the agent
            agent_input = {
                "input": f"Search the web for current, reliable information to answer: {query}",
                "chat_history": chat_history or []
            }
            
            # Execute the agent
            result = self.agent_executor.invoke(agent_input)
            
            return {
                "query": query,
                "information": result.get("output", ""),
                "sources": [],  # Would be populated from actual tool calls
                "tool_used": "web_search",
                "success": True
            }
            
        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "information": "",
                "sources": [],
                "tool_used": "web_search",
                "success": False
            }


# Global instance
web_research_agent = WebResearchAgent()