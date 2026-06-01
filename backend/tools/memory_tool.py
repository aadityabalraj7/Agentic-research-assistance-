"""
LangChain tool for memory operations.
"""

from typing import Any, List, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from datetime import datetime

from backend.memory.memory_store import memory_store
from backend.models.schemas import MemoryEntry, UserPreference


class MemoryToolInput(BaseModel):
    """Input for the memory tool."""
    operation: str = Field(description="Operation to perform: 'get_preference', 'set_preference', 'get_history', 'add_memory'")
    key: Optional[str] = Field(default=None, description="Key for preference operations")
    value: Optional[Any] = Field(default=None, description="Value for set operations")
    entry_type: Optional[str] = Field(default=None, description="Type of memory entry to retrieve")
    limit: int = Field(default=10, description="Maximum number of results to return")


class MemoryTool(BaseTool):
    """Tool for memory operations."""
    
    name: str = "memory"
    description: str = """
    Useful for storing and retrieving information from persistent memory.
    Use this to remember user preferences, recall previous conversations, or store research findings.
    """
    args_schema: type[BaseModel] = MemoryToolInput
    
    def _run(self, operation: str, key: Optional[str] = None, value: Optional[Any] = None, entry_type: Optional[str] = None, limit: int = 10) -> str:
        """Use the tool to perform memory operations."""
        try:
            if operation == "get_preference":
                if not key:
                    return "Error: Key required for get_preference operation"
                result = memory_store.get_preference(key)
                return f"Preference '{key}': {result}"
            
            elif operation == "set_preference":
                if not key:
                    return "Error: Key required for set_preference operation"
                memory_store.set_preference(key, value)
                return f"Preference '{key}' set successfully"
            
            elif operation == "get_history":
                entries = memory_store.get_memory_entries(entry_type=entry_type, limit=limit)
                if not entries:
                    return "No memory entries found"
                
                result = f"Found {len(entries)} memory entries:\n"
                for entry in entries[:5]:
                    result += f"- [{entry.type}] {entry.content[:100]}...\n"
                return result
            
            elif operation == "add_memory":
                if not value:
                    return "Error: Value required for add_memory operation"
                entry_type_final = entry_type or "general"
                entry = MemoryEntry(
                    id="",
                    content=str(value),
                    type=entry_type_final,
                    timestamp=datetime.now(),
                    metadata={}
                )
                entry_id = memory_store.add_memory_entry(entry)
                return f"Memory entry added with ID: {entry_id}"
            
            else:
                return f"Error: Unknown operation '{operation}'. Supported operations: get_preference, set_preference, get_history, add_memory"
                
        except Exception as e:
            return f"Error performing memory operation: {str(e)}"


# Global instance
memory_tool = MemoryTool()