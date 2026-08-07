"""Reusable SQLite persistence functions for SupportGPT chat history."""

import logging
import sqlite3
from datetime import datetime
from typing import Final, Literal, TypedDict

from config import DATABASE_PATH

logger = logging.getLogger(__name__)

MessageRole = Literal["user", "assistant"]
FeedbackRating = Literal[-1, 1]
_VALID_ROLES: Final[frozenset[str]] = frozenset({"user", "assistant"})
_CREATE_CHAT_HISTORY_TABLE: Final[str] = """
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    message TEXT NOT NULL,
    timestamp TEXT NOT NULL
)
"""
_CREATE_FEEDBACK_TABLE: Final[str] = """
CREATE TABLE IF NOT EXISTS feedback (
    message_key TEXT PRIMARY KEY,
    rating INTEGER NOT NULL CHECK (rating IN (-1, 1)),
    message TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""


class StoredMessage(TypedDict):
    """A chat-history record returned from SQLite."""

    role: MessageRole
    message: str
    timestamp: str


class StoredFeedback(TypedDict):
    """A persisted response rating returned from SQLite."""

    message_key: str
    rating: FeedbackRating


class ChatHistoryDatabaseError(RuntimeError):
    """Raised when a chat-history database operation cannot be completed."""


def _connect() -> sqlite3.Connection:
    """Open a configured SQLite connection with row-name access enabled."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def _initialize_table(connection: sqlite3.Connection) -> None:
    """Create the chat history table when the database is first used."""
    connection.execute(_CREATE_CHAT_HISTORY_TABLE)


def _initialize_feedback_table(connection: sqlite3.Connection) -> None:
    """Create the response-feedback table when it is first needed."""
    connection.execute(_CREATE_FEEDBACK_TABLE)


def _timestamp_or_now(timestamp: str | None) -> str:
    """Return a supplied timestamp or a consistent timestamp for a new record."""
    return timestamp or datetime.now().strftime("%Y-%m-%d %H:%M")


def save_message(role: MessageRole, message: str, timestamp: str | None = None) -> None:
    """Persist one customer or assistant message in the SQLite history.

    Args:
        role: Either ``user`` or ``assistant``.
        message: Non-empty message text to store.
        timestamp: Optional creation time. The current local time is used when
            this value is not supplied.

    Raises:
        ValueError: If the role or message is invalid.
        ChatHistoryDatabaseError: If SQLite cannot save the record.
    """
    if role not in _VALID_ROLES:
        raise ValueError("role must be either 'user' or 'assistant'")
    if not message.strip():
        raise ValueError("message must not be empty")

    try:
        with _connect() as connection:
            _initialize_table(connection)
            connection.execute(
                "INSERT INTO chat_history (role, message, timestamp) VALUES (?, ?, ?)",
                (role, message, _timestamp_or_now(timestamp)),
            )
    except sqlite3.Error as error:
        logger.error("Could not save chat history message: %s", type(error).__name__)
        raise ChatHistoryDatabaseError("Could not save chat history message") from error


def get_messages() -> list[StoredMessage]:
    """Return all stored messages in their original conversation order.

    Raises:
        ChatHistoryDatabaseError: If SQLite cannot read the chat history.
    """
    try:
        with _connect() as connection:
            _initialize_table(connection)
            rows = connection.execute(
                "SELECT role, message, timestamp FROM chat_history ORDER BY id ASC"
            ).fetchall()
    except sqlite3.Error as error:
        logger.error("Could not read chat history: %s", type(error).__name__)
        raise ChatHistoryDatabaseError("Could not read chat history") from error

    return [
        {"role": row["role"], "message": row["message"], "timestamp": row["timestamp"]}
        for row in rows
    ]


def clear_history() -> None:
    """Delete all stored messages and their related feedback records.

    Raises:
        ChatHistoryDatabaseError: If SQLite cannot clear the chat history.
    """
    try:
        with _connect() as connection:
            _initialize_table(connection)
            _initialize_feedback_table(connection)
            connection.execute("DELETE FROM chat_history")
            connection.execute("DELETE FROM feedback")
    except sqlite3.Error as error:
        logger.error("Could not clear chat history: %s", type(error).__name__)
        raise ChatHistoryDatabaseError("Could not clear chat history") from error


def save_feedback(message_key: str, rating: FeedbackRating, message: str, timestamp: str) -> None:
    """Create or replace a customer's rating for one assistant response.

    Args:
        message_key: Stable application identifier for the assistant message.
        rating: ``1`` for positive feedback or ``-1`` for negative feedback.
        message: Assistant response text associated with the rating.
        timestamp: Timestamp assigned to the assistant response.
    """
    if not message_key.strip():
        raise ValueError("message_key must not be empty")
    if rating not in (-1, 1):
        raise ValueError("rating must be either -1 or 1")
    if not message.strip():
        raise ValueError("message must not be empty")

    try:
        with _connect() as connection:
            _initialize_feedback_table(connection)
            connection.execute(
                """
                INSERT INTO feedback (message_key, rating, message, timestamp)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(message_key) DO UPDATE SET
                    rating = excluded.rating,
                    message = excluded.message,
                    timestamp = excluded.timestamp,
                    created_at = CURRENT_TIMESTAMP
                """,
                (message_key, rating, message, timestamp),
            )
    except sqlite3.Error as error:
        logger.error("Could not save response feedback: %s", type(error).__name__)
        raise ChatHistoryDatabaseError("Could not save response feedback") from error


def get_feedback(message_key: str) -> FeedbackRating | None:
    """Return the stored rating for an assistant message, if one exists."""
    try:
        with _connect() as connection:
            _initialize_feedback_table(connection)
            row = connection.execute(
                "SELECT rating FROM feedback WHERE message_key = ?", (message_key,)
            ).fetchone()
    except sqlite3.Error as error:
        logger.error("Could not read response feedback: %s", type(error).__name__)
        raise ChatHistoryDatabaseError("Could not read response feedback") from error

    return None if row is None else row["rating"]
