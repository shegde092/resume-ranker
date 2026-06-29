import logging
import sqlite3
from typing import List, Dict, Any
import os

logger = logging.getLogger(__name__)

class RankMemory:
    """
    SQLite persistence layer for candidate rankings.
    Logs each ranking session and candidate evaluations.
    """
    def __init__(self, db_path: str = "data/rank_history.db"):
        self.db_path = db_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize schema
        self._init_db()

    def _init_db(self):
        # Apply 30s timeout and WAL journal mode for concurrent SQLite safety
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ranking_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    final_score REAL,
                    confidence TEXT,
                    reason TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

    def log_session(self, session_id: str, candidates: List[Dict[str, Any]]):
        """
        Persists a batch of ranked candidates to SQLite safely.
        """
        if not candidates:
            return
            
        records = []
        for cand in candidates:
            records.append((
                session_id,
                str(cand.get("candidate_id", "UNKNOWN")),
                float(cand.get("final_score", 0.0)),
                str(cand.get("confidence", "UNKNOWN")),
                str(cand.get("reason", "No reason provided."))
            ))
            
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.executemany('''
                    INSERT INTO ranking_history (session_id, candidate_id, final_score, confidence, reason)
                    VALUES (?, ?, ?, ?, ?)
                ''', records)
                conn.commit()
                logger.info(f"Persisted {len(records)} candidate results for session {session_id}.")
        except Exception as e:
            logger.error(f"Failed to log session to SQLite: {e}")
