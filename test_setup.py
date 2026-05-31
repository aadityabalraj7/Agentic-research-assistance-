"""
Basic test script for the Agentic Research Assistant.
Tests core components to ensure they work together.
"""

import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test core imports
        from backend.main import app
        print("✓ Main app imported")
        
        from backend.agents.planner_agent import planner_agent
        print("✓ Planner agent imported")
        
        from backend.agents.retrieval_agent import retrieval_agent
        print("✓ Retrieval agent imported")
        
        from backend.agents.web_agent import web_research_agent
        print("✓ Web agent imported")
        
        from backend.agents.verifier_agent import verifier_agent
        print("✓ Verifier agent imported")
        
        from backend.agents.writer_agent import writer_agent
        print("✓ Writer agent imported")
        
        from backend.agents.supervisor_agent import supervisor_agent
        print("✓ Supervisor agent imported")
        
        from backend.graph.research_graph import research_graph
        print("✓ Research graph imported")
        
        from backend.vectorstore.chroma_manager import chroma_manager
        print("✓ ChromaDB manager imported")
        
        from backend.memory.memory_store import memory_store
        print("✓ Memory store imported")
        
        from backend.tools.rag_tool import rag_tool
        print("✓ RAG tool imported")
        
        from backend.tools.search_tool import web_search_tool
        print("✓ Web search tool imported")
        
        from backend.tools.memory_tool import memory_tool
        print("✓ Memory tool imported")
        
        from backend.services.chat_service import chat_service
        print("✓ Chat service imported")
        
        from backend.services.document_processor import document_processor
        print("✓ Document processor imported")
        
        from backend.exports.export_manager import export_manager
        print("✓ Export manager imported")
        
        print("\n✓ All imports successful!")
        return True
        
    except Exception as e:
        print(f"\n✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality of core components."""
    print("\nTesting basic functionality...")
    
    try:
        # Test ChromaDB manager
        from backend.vectorstore.chroma_manager import chroma_manager
        stats = chroma_manager.get_collection_stats()
        print(f"✓ ChromaDB stats: {stats}")
        
        # Test memory store
        from backend.memory.memory_store import memory_store
        memory_count = len(memory_store.get_memory_entries(limit=1))
        print(f"✓ Memory store working, {memory_count} entries")
        
        # Test document processor
        from backend.services.document_processor import document_processor
        print("✓ Document processor initialized")
        
        # Test export manager
        from backend.exports.export_manager import export_manager
        print("✓ Export manager initialized")
        
        print("\n✓ Basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🧪 Testing Agentic Research Assistant Components")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test basic functionality
    if imports_ok:
        functionality_ok = test_basic_functionality()
    else:
        functionality_ok = False
    
    print("\n" + "=" * 50)
    if imports_ok and functionality_ok:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Copy .env.example to .env and add your API keys")
        print("2. Start the backend: uvicorn backend.main:app --reload") 
        print("3. Start the frontend: streamlit run frontend/app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return imports_ok and functionality_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)