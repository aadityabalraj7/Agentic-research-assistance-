"""
Persistent memory store for the Agentic Research Assistant.
"""

import json
import sqlite3
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import uuid

from backend.models.schemas import MemoryEntry, UserPreference, ChatSession


class MemoryStore:
    """Manages persistent memory for conversations, research, and preferences."""
    
    def __init__(self, db_path: str = "./memory_db"):
        """
        Initialize memory store.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.connection = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database with required tables."""
        try:
            # Ensure directory exists
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Connect to database
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            
            # Create tables
            self._create_tables()
            
        except Exception as e:
            print(f"Error initializing memory database: {e}")
            raise
    
    def _create_tables(self):
        """Create database tables for memory storage."""
        cursor = self.connection.cursor()
        
        # Memory entries table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_entries (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT
            )
        """)
        
        # User preferences table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        # Chat sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                messages TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        # Research history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS research_history (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                report TEXT NOT NULL,
                citations TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT
            )
        """)
        
        self.connection.commit()
    
    # Memory entry methods
    def add_memory_entry(self, entry: MemoryEntry) -> str:
        """Add a memory entry and return its ID."""
        if not entry.id:
            entry.id = str(uuid.uuid4())
        
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT INTO memory_entries (id, content, type, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (
            entry.id,
            entry.content,
            entry.type,
            entry.timestamp.isoformat(),
            json.dumps(entry.metadata)
        ))
        self.connection.commit()
        return entry.id
    
    def get_memory_entries(self, entry_type: Optional[str] = None, limit: int = 100) -> List[MemoryEntry]:
        """Get memory entries, optionally filtered by type."""
        cursor = self.connection.cursor()
        
        if entry_type:
            cursor.execute("""
                SELECT id, content, type, timestamp, metadata
                FROM memory_entries
                WHERE type = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (entry_type, limit))
        else:
            cursor.execute("""
                SELECT id, content, type, timestamp, metadata
                FROM memory_entries
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
        
        entries = []
        for row in cursor.fetchall():
            entries.append(MemoryEntry(
                id=row["id"],
                content=row["content"],
                type=row["type"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                metadata=json.loads(row["metadata"]) if row["metadata"] else {}
            ))
        
        return entries
    
    # User preference methods
    def set_preference(self, key: str, value: Any) -> None:
        """Set a user preference."""
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO user_preferences (key, value, updated_at)
            VALUES (?, ?, ?)
        """, (
            key,
            json.dumps(value),
            datetime.now().isoformat()
        ))
        self.connection.commit()
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference."""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT value FROM user_preferences WHERE key = ?
        """, (key,))
        
        row = cursor.fetchone()
        if row:
            try:
                return json.loads(row["value"])
            except json.JSONDecodeError:
                return row["value"]
        return default
    
    # Chat session methods
    def save_chat_session(self, session: ChatSession) -> str:
        """Save a chat session and return its ID."""
        if not session.id:
            session.id = str(uuid.uuid4())
        
        # Update timestamp
        session.updated_at = datetime.now()
        
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO chat_sessions (id, messages, created_at, updated_at)
            VALUES (?, ?, ?, ?)
        """, (
            session.id,
            json.dumps([msg.dict() for msg in session.messages]),
            session.created_at.isoformat(),
            session.updated_at.isoformat()
        ))
        self.connection.commit()
        return session.id
    
    def get_chat_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a chat session by ID."""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT id, messages, created_at, updated_at
            FROM chat_sessions
            WHERE id = ?
        """, (session_id,))
        
        row = cursor.fetchone()
        if row:
            messages_data = json.loads(row["messages"])
            # Note: In a real implementation, we'd properly reconstruct ChatMessage objects
            return ChatSession(
                id=row["id"],
                messages=[],  # Simplified for now
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"])
            )
        return None
    
    # Research history methods
    def add_research_record(self, query: str, report: str, citations: List[Dict], metadata: Optional[Dict] = None) -> str:
        """Add a research record to history."""
        record_id = str(uuid.uuid4())
        
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT INTO research_history (id, query, report, citations, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            record_id,
            query,
            report,
            json.dumps(citations),
            datetime.now().isoformat(),
            json.dumps(metadata) if metadata else None
        ))
        self.connection.commit()
        return record_id
    
    def get_research_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get research history records."""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT id, query, report, citations, timestamp, metadata
            FROM research_history
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        history = []
        for row in cursor.fetchall():
            history.append({
                "id": row["id"],
                "query": row["query"],
                "report": row["report"],
                "citations": json.loads(row["citations"]) if row["citations"] else [],
                "timestamp": row["timestamp"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
            })
        
        return history
    
    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()


# Global instance
memory_store = MemoryStore()