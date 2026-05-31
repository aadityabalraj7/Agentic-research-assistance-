"""
Initialization script for the Agentic Research Assistant.
Creates necessary directories and verifies setup.
"""

import os
from pathlib import Path

def create_directories():
    """Create necessary directories for the application."""
    directories = [
        "uploads",
        "chroma_db", 
        "memory_db",
        "exports",
        "backend/uploads",
        "backend/chroma_db",
        "backend/memory_db", 
        "backend/exports"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def check_dependencies():
    """Check if critical dependencies are available."""
    dependencies = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"), 
        ("streamlit", "Streamlit"),
        ("langchain", "LangChain"),
        ("langgraph", "LangGraph"),
        ("chromadb", "ChromaDB"),
        ("sentence_transformers", "Sentence Transformers"),
        ("PyPDF2", "PyPDF2"),
        ("python-docx", "python-docx"),
        ("tavily-python", "Tavily Python"),
        ("reportlab", "ReportLab"),
        ("docx", "python-docx")
    ]
    
    missing = []
    for package, name in dependencies:
        try:
            __import__(package.replace("-", "_"))
            print(f"✓ {name} available")
        except ImportError:
            missing.append(name)
            print(f"✗ {name} missing")
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join([dep.lower().replace(" ", "-") for dep in missing]))
        return False
    else:
        print("\n✓ All dependencies available")
        return True

def check_env_file():
    """Check if environment file exists."""
    env_file = Path(".env")
    example_file = Path(".env.example")
    
    if env_file.exists():
        print("✓ .env file found")
        return True
    elif example_file.exists():
        print("⚠️  .env.example found, please copy to .env and fill in API keys")
        return False
    else:
        print("✗ No environment files found")
        return False

def main():
    """Main initialization function."""
    print("🔧 Initializing Agentic Research Assistant...")
    print("=" * 50)
    
    # Create directories
    create_directories()
    print()
    
    # Check dependencies
    deps_ok = check_dependencies()
    print()
    
    # Check environment
    env_ok = check_env_file()
    print()
    
    if deps_ok and env_ok:
        print("� Initialization complete! You can now:")
        print("   1. Copy .env.example to .env and add your API keys")
        print("   2. Start the backend: uvicorn backend.main:app --reload")
        print("   3. Start the frontend: streamlit run frontend/app.py")
    else:
        print("❌ Initialization incomplete. Please address the issues above.")

if __name__ == "__main__":
    main()