"""
Main Streamlit application for the Agentic Research Assistant.
"""

import streamlit as st
import requests
import json
from typing import Dict, Any, List
import time
from datetime import datetime

# Configure Streamlit page
st.set_page_config(
    page_title="Agentic Research Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8001/api"

def check_api_health() -> bool:
    """Check if the API is healthy."""
    try:
        # Create session with proxy bypass
        session = requests.Session()
        session.trust_env = False
        response = session.get(f"{API_BASE_URL.replace('/api', '')}/health", timeout=10)
        return response.status_code == 200
    except:
        return False

def send_chat_message(message: str, session_id: str = None) -> Dict[str, Any]:
    """Send a chat message to the API."""
    try:
        # Create session with proxy bypass
        session = requests.Session()
        session.trust_env = False
        payload = {
            "message": message,
            "session_id": session_id
        }
        response = session.post(f"{API_BASE_URL}/chat", json=payload, timeout=120)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}", "details": response.text}
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

def conduct_research(query: str, use_documents: bool = True, use_web: bool = True, session_id: str = None) -> Dict[str, Any]:
    """Conduct deep research via the API."""
    try:
        # Create session with proxy bypass
        session = requests.Session()
        session.trust_env = False
        payload = {
            "query": query,
            "use_documents": use_documents,
            "use_web": use_web,
            "session_id": session_id
        }
        response = session.post(f"{API_BASE_URL}/research", json=payload, timeout=300)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}", "details": response.text}
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

def upload_file(uploaded_file) -> Dict[str, Any]:
    """Upload a file to the API."""
    try:
        # Create session with proxy bypass
        session = requests.Session()
        session.trust_env = False
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = session.post(f"{API_BASE_URL}/upload", files=files, timeout=120)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}", "details": response.text}
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

def get_history(session_id: str) -> Dict[str, Any]:
    """Get chat history for a session."""
    try:
        # Create session with proxy bypass
        session = requests.Session()
        session.trust_env = False
        response = session.get(f"{API_BASE_URL}/history/{session_id}", timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}", "details": response.text}
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

def get_memory() -> Dict[str, Any]:
    """Get memory statistics."""
    try:
        # Create session with proxy bypass
        session = requests.Session()
        session.trust_env = False
        response = session.get(f"{API_BASE_URL}/memory", timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}", "details": response.text}
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{int(time.time())}"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "research_in_progress" not in st.session_state:
    st.session_state.research_in_progress = False

# Main app
def main():
    st.title("🔍 Agentic Research Assistant")
    st.markdown("*An AI-powered autonomous research analyst*")
    
    # Check API health
    api_healthy = check_api_health()
    if not api_healthy:
        st.error("⚠️ Backend API is not available. Please ensure the FastAPI server is running.")
        st.info("Start the backend with: `uvicorn backend.main:app --reload --port 8001`")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("🧭 Navigation")
        
        # Upload Documents
        st.subheader("📄 Upload Documents")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'docx', 'txt'],
            help="Upload PDF, DOCX, or TXT files for research"
        )
        
        if uploaded_file is not None:
            if st.button("📤 Upload File"):
                with st.spinner("Uploading and processing..."):
                    result = upload_file(uploaded_file)
                    if "error" in result:
                        st.error(f"Upload failed: {result['error']}")
                    else:
                        st.success(result.get("message", "File uploaded successfully"))
                        st.info(f"Chunks created: {result.get('chunks_created', 0)}")
        
        st.divider()
        
        # Research History
        st.subheader("📚 Research History")
        if st.button("📜 View History"):
            history_result = get_history(st.session_state.session_id)
            if "error" in history_result:
                st.error(f"Failed to load history: {history_result['error']}")
            else:
                st.json(history_result)
        
        st.divider()
        
        # Agent Trace
        st.subheader("👁️ Agent Trace")
        if st.button("👀 View Trace"):
            # This would show real-time agent execution
            st.info("Agent trace functionality would be implemented here")
        
        st.divider()
        
        # Memory Viewer
        st.subheader("🧠 Memory")
        if st.button("💭 View Memory"):
            memory_result = get_memory()
            if "error" in memory_result:
                st.error(f"Failed to load memory: {memory_result['error']}")
            else:
                st.json(memory_result)
        
        st.divider()
        
        # Settings
        st.subheader("⚙️ Settings")
        use_documents = st.checkbox("Search Documents", value=True, help="Enable searching uploaded documents")
        use_web = st.checkbox("Search Web", value=True, help="Enable web search for current information")
        
        # Model selection would go here in a full implementation
        st.selectbox("Model", ["deepseek-chat", "deepseek-reasoner"], index=0, disabled=True)
    
    # Main chat interface
    chat_container = st.container()
    
    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # Show citations if available
                if message.get("citations"):
                    with st.expander("📖 Sources"):
                        for citation in message["citations"]:
                            st.text(f"• {citation}")
    
    # Chat input
    if prompt := st.chat_input("Ask me to research any topic..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Show thinking indicator
        with st.chat_message("assistant"):
            with st.spinner("🔍 Researching..."):
                # Conduct research
                research_result = conduct_research(
                    prompt,
                    use_documents=use_documents,
                    use_web=use_web,
                    session_id=st.session_state.session_id
                )
                
                if "error" in research_result:
                    st.error(f"Research failed: {research_result['error']}")
                    assistant_response = f"I encountered an error while researching: {research_result['error']}"
                else:
                    # Extract response
                    assistant_response = research_result.get("response", 
                                                            research_result.get("report", 
                                                                            "No response generated"))
                    
                    # Add assistant message to chat
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": assistant_response,
                        "citations": research_result.get("citations", [])
                    })
                    
                    # Display assistant message
                    st.markdown(assistant_response)
                    
                    # Show citations if available
                    if research_result.get("citations"):
                        with st.expander("📖 Sources"):
                            for citation in research_result["citations"]:
                                st.text(f"• {citation}")
    
    # Footer
    st.divider()
    st.caption("Agentic Research Assistant • Powered by LangGraph, LangChain, FastAPI & Streamlit")

if __name__ == "__main__":
    main()