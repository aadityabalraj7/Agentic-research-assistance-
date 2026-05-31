"""
Planner Agent for the Agentic Research Assistant.
Responsible for understanding research objectives and creating research plans.
"""

from typing import List, Dict, Any
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool

from backend.tools.rag_tool import rag_tool
from backend.tools.search_tool import web_search_tool
from backend.tools.memory_tool import memory_tool


class PlannerAgent:
    """Agent responsible for planning research tasks."""
    
    def __init__(self, model_name: str = "deepseek-chat"):
        """
        Initialize the Planner Agent.
        
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
            temperature=0.1,
            max_tokens=2000
        )
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a Planner Agent in an Agentic Research Assistant system.
                Your role is to understand research objectives and create detailed research plans.
                
                When given a research query, you should:
                1. Analyze the research goal and break it down into specific, actionable subtasks
                2. Determine which tools (document retrieval, web search, memory) are likely needed for each subtask
                3. Create a structured research plan with clear objectives
                4. Consider what type of information would be most valuable
                
                Available tools:
                - document_retrieval: Search uploaded documents (PDF, DOCX, TXT)
                - web_search: Search the internet for current information
                - memory: Access persistent memory for user preferences and history
                
                Output your plan as a structured list of subtasks with tool recommendations.
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
    
    def create_research_plan(self, query: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Create a research plan for the given query.
        
        Args:
            query: The research question or objective
            chat_history: Previous conversation history
            
        Returns:
            Dictionary containing the research plan and execution results
        """
        try:
            # Prepare input for the agent
            agent_input = {
                "input": f"Create a detailed research plan for the following query: {query}",
                "chat_history": chat_history or []
            }
            
            # Execute the agent
            result = self.agent_executor.invoke(agent_input)
            
            # Parse and structure the result
            plan = {
                "objective": query,
                "subtasks": self._extract_subtasks(result.get("output", "")),
                "tools_needed": self._identify_required_tools(result.get("output", "")),
                "estimated_time": self._estimate_time(result.get("output", "")),
                "full_plan": result.get("output", "")
            }
            
            return plan
            
        except Exception as e:
            return {
                "objective": query,
                "error": str(e),
                "subtasks": [],
                "tools_needed": [],
                "estimated_time": 0,
                "full_plan": f"Error creating research plan: {str(e)}"
            }
    
    def _extract_subtasks(self, plan_text: str) -> List[str]:
        """Extract subtasks from the plan text."""
        # Simple extraction - in practice, this could be more sophisticated
        lines = plan_text.split('\n')
        subtasks = []
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('*') or 
                        any(line.startswith(f"{i}.") for i in range(1, 10))):
                # Clean up the line
                clean_line = line.lstrip('- *0123456789. ')
                if clean_line:
                    subtasks.append(clean_line)
        return subtasks if subtasks else [plan_text]
    
    def _identify_required_tools(self, plan_text: str) -> List[str]:
        """Identify which tools are mentioned in the plan."""
        tools = []
        plan_lower = plan_text.lower()
        if "document" in plan_lower or "retrieval" in plan_lower or "upload" in plan_lower:
            tools.append("document_retrieval")
        if "web" in plan_lower or "search" in plan_lower or "internet" in plan_lower:
            tools.append("web_search")
        if "memory" in plan_lower or "remember" in plan_lower or "recall" in plan_lower:
            tools.append("memory")
        return list(set(tools)) if tools else ["document_retrieval", "web_search"]  # Default
    
    def _estimate_time(self, plan_text: str) -> int:
        """Estimate time required for the research plan in minutes."""
        # Simple heuristic based on number of subtasks
        subtasks = self._extract_subtasks(plan_text)
        base_time = 2  # 2 minutes per subtask
        return max(len(subtasks) * base_time, 5)  # Minimum 5 minutes


# Global instance
planner_agent = PlannerAgent()