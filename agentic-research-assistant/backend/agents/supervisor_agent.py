"""
Supervisor Agent for the Agentic Research Assistant.
Responsible for routing tasks, managing workflow, and coordinating agents.
"""

from typing import List, Dict, Any, Optional
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool

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
    
    def __init__(self, model_name: str = "deepseek-chat"):
        """
        Initialize the Supervisor Agent.
        
        Args:
            model_name: Name of the LLM to use via OpenRouter
        """
        self.model_name = model_name
        # Tools for the supervisor to delegate tasks
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
                """You are a Supervisor Agent in an Agentic Research Assistant system.
                Your role is to orchestrate the research workflow by coordinating specialized agents.
                
                You have access to:
                - Planner Agent: Creates research plans and breaks down objectives
                - Retrieval Agent: Searches uploaded documents for information
                - Web Research Agent: Searches the internet for current information
                - Verifier Agent: Fact-checks information and detects contradictions
                - Writer Agent: Generates final answers and research reports
                
                Your responsibilities include:
                1. Delegating tasks to the appropriate specialized agents
                2. Managing the flow of information between agents
                3. Ensuring the research process follows logical steps
                4. Handling errors and retrying failed operations
                5. Determining when sufficient information has been gathered
                6. Coordinating the final response generation
                
                You should think strategically about which agents to use and in what order
                based on the research query and available information.
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
            max_iterations=10,
            early_stopping_method="generate"
        )
        
        return agent_executor
    
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
                "result": plan_result
            })
            
            # Step 2: Information Gathering
            all_information = []
            all_sources = []
            
            if use_documents:
                print("Step 2a: Retrieving information from documents...")
                retrieval_result = retrieval_agent.retrieve_information(query, chat_history)
                research_session["steps"].append({
                    "agent": "Retrieval",
                    "action": "retrieve_information",
                    "result": retrieval_result
                })
                if retrieval_result.get("success"):
                    all_information.append(("documents", retrieval_result.get("information", "")))
                    # Sources would be extracted from retrieval_result
            
            if use_web:
                print("Step 2b: Researching information from web...")
                web_result = web_research_agent.research_information(query, chat_history)
                research_session["steps"].append({
                    "agent": "Web Research",
                    "action": "research_information",
                    "result": web_result
                })
                if web_result.get("success"):
                    all_information.append(("web", web_result.get("information", "")))
                    # Sources would be extracted from web_result
            
            # Combine information
            combined_information = "\n\n".join([f"Source: {source_type}\n{info}" for source_type, info in all_information])
            
            # Step 3: Fact Verification
            print("Step 3: Verifying information...")
            if combined_information.strip():
                verification_result = verifier_agent.verify_information(
                    combined_information, 
                    sources=[],  # Would be actual sources
                    chat_history=chat_history
                )
                research_session["steps"].append({
                    "agent": "Verifier",
                    "action": "verify_information",
                    "result": verification_result
                })
                
                verified_info = verification_result.get("verification_result", combined_information)
                research_session["verified_information"] = verified_info
            else:
                verified_info = "No information retrieved from available sources."
                research_session["verified_information"] = verified_info
            
            # Step 4: Response Generation
            print("Step 4: Generating response...")
            # For simplicity, we'll use mock sources
            mock_sources = []
            response_result = writer_agent.generate_response(
                query, 
                verified_info, 
                sources=mock_sources,
                chat_history=chat_history
            )
            research_session["steps"].append({
                "agent": "Writer",
                "action": "generate_response",
                "result": response_result
            })
            
            research_session["final_response"] = response_result.get("response", "")
            
            # Step 5: Research Report Generation (optional)
            print("Step 5: Generating research report...")
            report_result = writer_agent.generate_research_report(
                query, 
                verified_info, 
                sources=mock_sources,
                research_plan=plan_result,
                chat_history=chat_history
            )
            research_session["steps"].append({
                "agent": "Writer",
                "action": "generate_research_report",
                "result": report_result
            })
            
            research_session["research_report"] = report_result.get("report", "")
            
            # Determine overall success
            research_session["success"] = (
                plan_result.get("success", True) and
                (not use_documents or retrieval_result.get("success", True)) and
                (not use_web or web_result.get("success", True)) and
                verification_result.get("success", True) and
                response_result.get("success", True) and
                report_result.get("success", True)
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