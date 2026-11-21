
import sqlite3
from datetime import datetime
from pathlib import Path

# Store the database beside this Python file
DATABASE_PATH = Path(__file__).resolve().parent / "phishguard_history.db"


def get_connection():
    """Create a connection to the PhishGuard database."""
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    """Create the email history table if it doesn't exist."""
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS email_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                decision_strength REAL NOT NULL,
                analyzed_at TEXT NOT NULL
            )
        """)


def save_analysis(subject, prediction, decision_strength):
    """Save one email analysis to the database."""
    analyzed_at = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    with get_connection() as connection:
        connection.execute("""
            INSERT INTO email_history
            (subject, prediction, decision_strength, analyzed_at)
            VALUES (?, ?, ?, ?)
        """, (
            subject,
            int(prediction),
            float(decision_strength),
            analyzed_at
        ))


def get_history(limit=100):
    """Return the latest email analyses."""
    with get_connection() as connection:
        rows = connection.execute("""
            SELECT id, subject, prediction, decision_strength, analyzed_at
            FROM email_history
            ORDER BY id DESC
            LIMIT ?
        """, (int(limit),)).fetchall()

    return [dict(row) for row in rows]


def clear_history():
    """Delete saved analysis records."""
    with get_connection() as connection:
        connection.execute("DELETE FROM email_history")


def get_statistics():
    """Return statistics for all saved email analyses."""
    with get_connection() as connection:
        row = connection.execute("""
            SELECT
                COUNT(*) AS total_analyses,
                SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END)
                    AS spam_count,
                SUM(CASE WHEN prediction = 0 THEN 1 ELSE 0 END)
                    AS legitimate_count,
                AVG(decision_strength) AS average_strength
            FROM email_history
        """).fetchone()

    return {
        "total_analyses": row["total_analyses"] or 0,
        "spam_count": row["spam_count"] or 0,
        "legitimate_count": row["legitimate_count"] or 0,
        "average_strength": row["average_strength"] or 0.0
    }
