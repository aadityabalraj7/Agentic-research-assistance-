"""
Retrieval Agent for the Agentic Research Assistant.
Responsible for searching ChromaDB and retrieving relevant document evidence.
"""

from typing import List, Dict, Any
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool

from backend.tools.rag_tool import rag_tool


class RetrievalAgent:
    """Agent responsible for retrieving information from uploaded documents."""
    
    def __init__(self, model_name: str = "deepseek-chat"):
        """
        Initialize the Retrieval Agent.
        
        Args:
            model_name: Name of the LLM to use via OpenRouter
        """
        self.model_name = model_name
        self.tools = [rag_tool]
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
                """You are a Retrieval Agent in an Agentic Research Assistant system.
                Your role is to search through uploaded documents and extract relevant information.
                
                When given a query, you should:
                1. Use the document_retrieval tool to search ChromaDB for relevant chunks
                2. Analyze the retrieved content for relevance and quality
                3. Extract key information that addresses the query
                4. Provide clear citations for all information retrieved
                5. Indicate if the documents contain sufficient information or if web search is needed
                
                Always include citations in your response using the format:
                [Filename.pdf | Page X] or [Filename.docx | Section Y]
                
                Be thorough but concise in your analysis.
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
    
    def retrieve_information(self, query: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Retrieve information from documents for the given query.
        
        Args:
            query: The research question or objective
            chat_history: Previous conversation history
            
        Returns:
            Dictionary containing retrieved information and sources
        """
        try:
            # Prepare input for the agent
            agent_input = {
                "input": f"Search for and retrieve relevant information from uploaded documents to answer: {query}",
                "chat_history": chat_history or []
            }
            
            # Execute the agent
            result = self.agent_executor.invoke(agent_input)
            
            # Get the raw sources from the tool (this is simplified)
            # In a full implementation, we'd extract sources from the agent's tool usage
            sources = []  # Would be populated from actual tool calls
            
            return {
                "query": query,
                "information": result.get("output", ""),
                "sources": sources,
                "tool_used": "document_retrieval",
                "success": True
            }
            
        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "information": "",
                "sources": [],
                "tool_used": "document_retrieval",
                "success": False
            }


# Global instance
retrieval_agent = RetrievalAgent()