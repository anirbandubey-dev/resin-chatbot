"""Domain message types used by the conversation layer."""

from typing import Literal, TypedDict

MessageRole = Literal["user", "assistant"]


class ChatMessage(TypedDict):
    """A timestamped chat message preserved in a customer conversation."""

    id: str
    role: MessageRole
    content: str
    timestamp: str
