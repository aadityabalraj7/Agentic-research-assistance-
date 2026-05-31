"""
Streamlit components for the Agentic Research Assistant.
"""

import streamlit as st
from typing import List, Dict, Any
import time


def render_agent_trace(trace_data: List[Dict[str, Any]] = None):
    """Render the agent trace visualization."""
    if trace_data is None:
        trace_data = []
    
    st.subheader("👁️ Agent Trace")
    
    if not trace_data:
        st.info("No agent activity yet")
        return
    
    for i, trace in enumerate(trace_data[-10:]):  # Show last 10 steps
        agent_name = trace.get("agent", "Unknown")
        action = trace.get("action", "Unknown")
        status = trace.get("status", "unknown")
        timestamp = trace.get("timestamp", "")
        
        # Status emoji
        status_emoji = {
            "completed": "✅",
            "failed": "❌",
            "running": "🟡",
            "pending": "⚪"
        }.get(status, "⚪")
        
        with st.container():
            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                st.write(f"{status_emoji} {agent_name}")
            with col2:
                st.write(f"*{action}*")
            with col3:
                if timestamp:
                    st.caption(timestamp.split('.')[0] if '.' in str(timestamp) else str(timestamp))
        
        # Show details in expander
        with st.expander(f"Details", expanded=False):
            st.json(trace.get("details", {}))


def render_citations(citations: List[Dict[str, Any]] = None):
    """Render citations in an expandable section."""
    if citations is None:
        citations = []
    
    if not citations:
        return
    
    with st.expander(f"📖 Sources ({len(citations)})", expanded=False):
        for i, citation in enumerate(citations, 1):
            if isinstance(citation, dict):
                # Handle structured citation
                filename = citation.get("filename", "Unknown")
                page = citation.get("page_number")
                if page:
                    st.write(f"{i}. [{filename} | Page {page}]")
                else:
                    st.write(f"{i}. [{filename}]")
                    
                    # Show URL if available
                    url = citation.get("url") or citation.get("chunk_id")
                    if url and url != filename:
                        st.caption(f"   URL: {url}")
            else:
                # Handle simple string citation
                st.write(f"{i}. {citation}")


def render_research_stats(stats: Dict[str, Any] = None):
    """Render research statistics."""
    if stats is None:
        stats = {}
    
    st.subheader("📊 Research Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Documents Uploaded", stats.get("documents_uploaded", 0))
    
    with col2:
        st.metric("Text Chunks", stats.get("text_chunks", 0))
    
    with col3:
        st.metric("Research Sessions", stats.get("research_sessions", 0))
    
    with col4:
        st.metric("Reports Generated", stats.get("reports_generated", 0))


def render_research_plan(plan: Dict[str, Any] = None):
    """Render the research plan."""
    if plan is None:
        plan = {}
    
    st.subheader("📋 Research Plan")
    
    if not plan:
        st.info("No research plan available")
        return
    
    st.write(f"**Objective:** {plan.get('objective', 'Not specified')}")
    
    if plan.get("subtasks"):
        st.write("**Subtasks:**")
        for i, subtask in enumerate(plan["subtasks"], 1):
            st.write(f"{i}. {subtask}")
    
    if plan.get("tools_needed"):
        st.write(f"**Tools Needed:** {', '.join(plan['tools_needed'])}")
    
    if plan.get("estimated_time"):
        st.write(f"**Estimated Time:** {plan['estimated_time']} minutes")


def render_verification_results(verification: Dict[str, Any] = None):
    """Render fact verification results."""
    if verification is None:
        verification = {}
    
    st.subheader("🔍 Fact Verification")
    
    if not verification:
        st.info("No verification results available")
        return
    
    status = verification.get("success", False)
    status_text = "✅ Verified" if status else "❌ Issues Found"
    st.write(f"**Status:** {status_text}")
    
    if verification.get("verification_result"):
        with st.expander("Verification Details", expanded=False):
            st.write(verification["verification_result"])
    
    if not verification.get("success", True) and verification.get("error"):
        st.error(f"Verification error: {verification['error']}")


def render_typing_indicator():
    """Render a typing indicator."""
    with st.container():
        st.write("🤔 Assistant is thinking...")
        # Simple animation effect
        placeholder = st.empty()
        for i in range(3):
            placeholder.write("🤔 Assistant is thinking" + "." * (i + 1))
            time.sleep(0.5)
        placeholder.empty()


# Export components
__all__ = [
    "render_agent_trace",
    "render_citations", 
    "render_research_stats",
    "render_research_plan",
    "render_verification_results",
    "render_typing_indicator"
]