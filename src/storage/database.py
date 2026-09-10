import os
import time
import sqlite3
import logging
from typing import Optional

from src.storage.models import SessionSummary, SessionEvent

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_path="~/.smart_driver_monitor/sessions.db"):
        self._db_path = db_path
        if db_path != ":memory:":
            self._db_path = os.path.expanduser(db_path)
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._create_tables()

    def _create_tables(self):
        cursor = self._conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time REAL,
                end_time REAL,
                duration_seconds REAL,
                total_yawns INTEGER DEFAULT 0,
                total_eye_closures INTEGER DEFAULT 0,
                total_distraction_events INTEGER DEFAULT 0,
                total_phone_events INTEGER DEFAULT 0,
                total_hand_down_events INTEGER DEFAULT 0,
                total_looking_away_events INTEGER DEFAULT 0,
                avg_safety_score REAL DEFAULT 0.0,
                max_risk_level TEXT DEFAULT 'SAFE'
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                timestamp REAL,
                event_type TEXT,
                severity TEXT,
                metric_value REAL DEFAULT 0.0,
                duration_seconds REAL DEFAULT 0.0,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        self._conn.commit()

    def save_session(self, summary: SessionSummary) -> int:
        cursor = self._conn.cursor()
        cursor.execute("""
            INSERT INTO sessions (
                start_time, end_time, duration_seconds,
                total_yawns, total_eye_closures, total_distraction_events,
                total_phone_events, total_hand_down_events, total_looking_away_events,
                avg_safety_score, max_risk_level
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            summary.start_time, summary.end_time, summary.duration_seconds,
            summary.total_yawns, summary.total_eye_closures, summary.total_distraction_events,
            summary.total_phone_events, summary.total_hand_down_events, summary.total_looking_away_events,
            summary.avg_safety_score, summary.max_risk_level,
        ))
        self._conn.commit()
        session_id = cursor.lastrowid
        logger.info("Session saved: id=%d", session_id)
        return session_id

    def save_event(self, session_id: int, event: SessionEvent):
        cursor = self._conn.cursor()
        cursor.execute("""
            INSERT INTO events (session_id, timestamp, event_type, severity, metric_value, duration_seconds)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session_id, event.timestamp, event.event_type, event.severity,
            event.metric_value, event.duration_seconds,
        ))
        self._conn.commit()

    def get_recent_sessions(self, limit=10):
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM sessions ORDER BY start_time DESC LIMIT ?",
            (limit,)
        )
        return cursor.fetchall()

    def close(self):
        if self._conn:
            self._conn.close()
