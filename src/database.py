"""SQLite Database module for Persistent Research History and Archival."""

import json
import sqlite3
from typing import List, Optional, Dict, Any
from src.config import ROOT_DIR
from src.models import ResearchReport

DB_PATH = ROOT_DIR / "research_history.db"


def get_connection() -> sqlite3.Connection:
    """Returns an active SQLite connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the SQLite schema for research reports."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS research_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                created_at TEXT NOT NULL,
                model_used TEXT NOT NULL,
                synthesis_mode TEXT NOT NULL,
                executive_summary TEXT NOT NULL,
                grounding_score REAL NOT NULL,
                total_sources INTEGER NOT NULL,
                execution_time REAL NOT NULL,
                report_json TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_report_to_db(report: ResearchReport) -> int:
    """Saves a completed ResearchReport to SQLite database."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO research_history (
                query, created_at, model_used, synthesis_mode,
                executive_summary, grounding_score, total_sources,
                execution_time, report_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report.query,
                report.created_at,
                report.model_used,
                report.synthesis_mode,
                report.executive_summary[:400],
                report.grounding_score,
                report.total_sources_scraped,
                report.execution_time_seconds,
                report.model_dump_json(),
            ),
        )
        conn.commit()
        return cursor.lastrowid


def get_recent_reports(limit: int = 15) -> List[Dict[str, Any]]:
    """Retrieves recent research query summaries from the database."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, query, created_at, model_used, grounding_score, execution_time, executive_summary
            FROM research_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_report_by_id(report_id: int) -> Optional[ResearchReport]:
    """Loads a full ResearchReport from JSON stored in the database."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT report_json FROM research_history WHERE id = ?",
            (report_id,),
        )
        row = cursor.fetchone()
        if row and row["report_json"]:
            try:
                data = json.loads(row["report_json"])
                return ResearchReport(**data)
            except Exception:
                return None
    return None


def clear_history():
    """Wipes all saved research history."""
    init_db()
    with get_connection() as conn:
        conn.execute("DELETE FROM research_history")
        conn.commit()
