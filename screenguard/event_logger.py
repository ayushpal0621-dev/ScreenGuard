"""
ScreenGuard Event Logger

SQLite-based local event logging for privacy events.
Stores timestamps, event types, face counts, and actions taken.
Never stores webcam frames unless explicitly enabled.
"""

import sqlite3
import time
import logging
import os
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class EventLogger:
    """
    Local SQLite event logger for ScreenGuard.
    
    Events logged:
        - SAFE: Normal operation
        - UNKNOWN_PERSON: Unknown face detected
        - PRIVACY_THREAT: Confirmed privacy threat
        - ALERT: Alert triggered
        - LOCK: Computer locked
        - REGISTRATION: User registered
        - STARTED: Protection started
        - STOPPED: Protection stopped
        - ERROR: Error occurred
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            data_dir = Path(__file__).parent.parent / "data"
            os.makedirs(data_dir, exist_ok=True)
            self._db_path = str(data_dir / "screenguard_events.db")
        else:
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            self._db_path = db_path

        self._init_db()

    def _init_db(self):
        """Create the events table if it doesn't exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    face_count INTEGER DEFAULT 0,
                    primary_user_detected INTEGER DEFAULT 0,
                    unknown_person_detected INTEGER DEFAULT 0,
                    action TEXT,
                    duration REAL,
                    details TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)
            """)
        logger.info("Event database initialized at %s", self._db_path)

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def log_event(
        self,
        event_type: str,
        face_count: int = 0,
        primary_user_detected: bool = False,
        unknown_person_detected: bool = False,
        action: Optional[str] = None,
        duration: Optional[float] = None,
        details: Optional[str] = None,
    ):
        """Log a privacy event."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO events 
                        (timestamp, event_type, face_count, primary_user_detected,
                         unknown_person_detected, action, duration, details)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        time.time(),
                        event_type,
                        face_count,
                        1 if primary_user_detected else 0,
                        1 if unknown_person_detected else 0,
                        action,
                        duration,
                        details,
                    ),
                )
            logger.debug("Event logged: %s (faces=%d, action=%s)", event_type, face_count, action)
        except Exception as e:
            logger.error("Failed to log event: %s", e)

    def get_recent_events(self, limit: int = 50) -> list[dict]:
        """Get the most recent events."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (limit,)
                )
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error("Failed to fetch events: %s", e)
            return []

    def get_events_since(self, since_timestamp: float) -> list[dict]:
        """Get events since a given timestamp."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM events WHERE timestamp >= ? ORDER BY timestamp DESC",
                    (since_timestamp,),
                )
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error("Failed to fetch events: %s", e)
            return []

    def get_event_stats(self) -> dict:
        """Get summary statistics of logged events."""
        try:
            with self._get_connection() as conn:
                total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
                threats = conn.execute(
                    "SELECT COUNT(*) FROM events WHERE event_type = 'PRIVACY_THREAT'"
                ).fetchone()[0]
                alerts = conn.execute(
                    "SELECT COUNT(*) FROM events WHERE event_type = 'ALERT'"
                ).fetchone()[0]
                locks = conn.execute(
                    "SELECT COUNT(*) FROM events WHERE event_type = 'LOCK'"
                ).fetchone()[0]
                return {
                    "total_events": total,
                    "privacy_threats": threats,
                    "alerts_triggered": alerts,
                    "locks_triggered": locks,
                }
        except Exception as e:
            logger.error("Failed to get stats: %s", e)
            return {}

    def clear_all(self):
        """Delete all logged events."""
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM events")
            logger.info("All events cleared.")
        except Exception as e:
            logger.error("Failed to clear events: %s", e)
