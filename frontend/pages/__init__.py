"""
Streamlit pages for the Agentic Research Assistant.
"""

import streamlit as st


def chat_page():
    """Main chat page."""
    st.title("💬 Research Chat")
    st.markchat("Chat with your Agentic Research Assistant")
    # The main chat logic is in app.py


def dashboard_page():
    """Dashboard page showing research analytics."""
    st.title("📊 Research Dashboard")
    st.markdown("View your research statistics and analytics")
    
    # Placeholder for dashboard content
    st.info("Dashboard functionality coming soon")


def history_page():
    """Research history page."""
    st.title("📚 Research History")
    st.markdown("View your past research sessions")
    
    # Placeholder for history content
    st.info("History functionality coming soon")


def settings_page():
    """Settings page."""
    st.title("⚙️ Settings")
    st.markdown("Configure your research assistant preferences")
    
    # Placeholder for settings content
    st.info("Settings functionality coming soon")


# Export page functions
__all__ = ["chat_page", "dashboard_page", "history_page", "settings_page"]