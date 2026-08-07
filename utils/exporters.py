"""Export contracts reserved for later PDF and CSV delivery features."""

from typing import Protocol, Sequence

from chatbot.models import ChatMessage


class ConversationExporter(Protocol):
    """Defines the future export boundary for customer conversations."""

    def export(self, messages: Sequence[ChatMessage]) -> bytes:
        """Serialize a conversation into a downloadable document."""
