"""
Fact Verification Agent for the Agentic Research Assistant.
Responsible for detecting contradictions and verifying evidence quality.
"""

from typing import List, Dict, Any
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool

from backend.tools.rag_tool import rag_tool
from backend.tools.search_tool import web_search_tool
from backend.tools.memory_tool import memory_tool


class VerifierAgent:
    """Agent responsible for verifying facts and detecting contradictions."""
    
    def __init__(self, model_name: str = "deepseek-chat"):
        """
        Initialize the Verifier Agent.
        
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
            temperature=0.1,  # Low temperature for factual consistency
            max_tokens=2000
        )
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a Fact Verification Agent in an Agentic Research Assistant system.
                Your role is to verify the accuracy of information and detect contradictions.
                
                When given information to verify, you should:
                1. Cross-check facts across multiple sources (documents, web, memory)
                2. Identify any contradictions or inconsistencies in the evidence
                3. Assess the credibility and reliability of sources
                4. Determine confidence levels for different pieces of information
                5. Flag information that requires additional verification
                
                You should be skeptical and thorough in your verification process.
                When contradictions are found, clearly explain them and suggest which sources are more reliable.
                
                Provide a verification report that includes:
                - Verified facts with confidence levels
                - Contradictions or inconsistencies found
                - Source reliability assessments
                - Recommendations for additional verification if needed
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
    
    def verify_information(self, information: str, sources: List[Dict] = None, chat_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Verify information and detect contradictions.
        
        Args:
            information: The information to verify (could be from retrieval or web research)
            sources: List of sources that provided the information
            chat_history: Previous conversation history
            
        Returns:
            Dictionary containing verification results
        """
        try:
            # Prepare input for the agent
            source_info = ""
            if sources:
                source_info = f"\n\nSources to verify against:\n{chr(10).join([f'- {s.get(\"filename\", \"Unknown\")}' for s in sources])}"
            
            agent_input = {
                "input": f"Please verify the following information for accuracy and consistency:{source_info}\n\nInformation to verify:\n{information}",
                "chat_history": chat_history or []
            }
            
            # Execute the agent
            result = self.agent_executor.invoke(agent_input)
            
            return {
                "information": information,
                "verification_result": result.get("output", ""),
                "sources_checked": len(sources) if sources else 0,
                "tool_used": "fact_verification",
                "success": True
            }
            
        except Exception as e:
            return {
                "information": information,
                "error": str(e),
                "verification_result": "",
                "sources_checked": 0,
                "tool_used": "fact_verification",
                "success": False
            }


# Global instance
verifier_agent = VerifierAgent()