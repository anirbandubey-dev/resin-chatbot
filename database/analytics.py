"""Read-only SQLite analytics queries for SupportGPT conversations."""

import logging
import sqlite3
from dataclasses import dataclass

from config import DATABASE_PATH

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuestionFrequency:
    """One frequently asked customer question and its occurrence count."""

    question: str
    count: int


@dataclass(frozen=True)
class DailyChatCount:
    """Count of customer messages recorded for a calendar day."""

    date: str
    count: int


@dataclass(frozen=True)
class ChatAnalytics:
    """The dashboard metrics derived from persisted chat history."""

    total_chats: int
    average_response_length: float
    frequent_questions: list[QuestionFrequency]
    daily_chat_counts: list[DailyChatCount]
    positive_feedback: int
    negative_feedback: int


class AnalyticsDatabaseError(RuntimeError):
    """Raised when SQLite analytics cannot be calculated safely."""


def _connect() -> sqlite3.Connection:
    """Open the configured SQLite database for read-only analytics queries."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def _history_exists(connection: sqlite3.Connection) -> bool:
    """Return whether the lazy-created chat history table is available."""
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'chat_history'"
    ).fetchone()
    return row is not None


def _feedback_counts(connection: sqlite3.Connection) -> tuple[int, int]:
    """Return positive and negative feedback totals when the table exists."""
    table_exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'feedback'"
    ).fetchone()
    if table_exists is None:
        return 0, 0
    row = connection.execute(
        """
        SELECT
            COALESCE(SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END), 0) AS positive,
            COALESCE(SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END), 0) AS negative
        FROM feedback
        """
    ).fetchone()
    return int(row["positive"]), int(row["negative"])


def get_chat_analytics(question_limit: int = 5) -> ChatAnalytics:
    """Calculate dashboard metrics from the persisted SQLite chat history.

    ``total_chats`` represents customer messages. Assistant reply length is
    measured in characters, while daily counts are grouped by stored date.
    """
    if question_limit < 1:
        raise ValueError("question_limit must be greater than zero")

    try:
        with _connect() as connection:
            if not _history_exists(connection):
                positive_feedback, negative_feedback = _feedback_counts(connection)
                return ChatAnalytics(0, 0.0, [], [], positive_feedback, negative_feedback)

            total_chats = connection.execute(
                "SELECT COUNT(*) FROM chat_history WHERE role = 'user'"
            ).fetchone()[0]
            average_length = connection.execute(
                "SELECT COALESCE(AVG(LENGTH(message)), 0) FROM chat_history WHERE role = 'assistant'"
            ).fetchone()[0]
            frequent_rows = connection.execute(
                """
                SELECT MIN(message) AS question, COUNT(*) AS count
                FROM chat_history
                WHERE role = 'user' AND TRIM(message) <> ''
                GROUP BY LOWER(TRIM(message))
                ORDER BY count DESC, question ASC
                LIMIT ?
                """,
                (question_limit,),
            ).fetchall()
            daily_rows = connection.execute(
                """
                SELECT SUBSTR(timestamp, 1, 10) AS date, COUNT(*) AS count
                FROM chat_history
                WHERE role = 'user'
                GROUP BY date
                ORDER BY date ASC
                """
            ).fetchall()
            positive_feedback, negative_feedback = _feedback_counts(connection)
    except sqlite3.Error as error:
        logger.error("Could not calculate chat analytics: %s", type(error).__name__)
        raise AnalyticsDatabaseError("Could not calculate chat analytics") from error

    return ChatAnalytics(
        total_chats=int(total_chats),
        average_response_length=round(float(average_length), 1),
        frequent_questions=[QuestionFrequency(str(row["question"]), int(row["count"])) for row in frequent_rows],
        daily_chat_counts=[DailyChatCount(str(row["date"]), int(row["count"])) for row in daily_rows],
        positive_feedback=positive_feedback,
        negative_feedback=negative_feedback,
    )
