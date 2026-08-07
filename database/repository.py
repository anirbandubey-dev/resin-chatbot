"""Persistence contracts reserved for the SQLite implementation phase."""

from typing import Protocol, Sequence

from chatbot.models import ChatMessage


class ConversationRepository(Protocol):
    """Defines future SQLite persistence operations for conversations."""

    def save_messages(self, conversation_id: str, messages: Sequence[ChatMessage]) -> None:
        """Persist all messages belonging to one conversation."""

    def get_messages(self, conversation_id: str) -> list[ChatMessage]:
        """Return a conversation transcript ordered by creation time."""
