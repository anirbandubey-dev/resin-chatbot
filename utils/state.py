"""Streamlit session-state management for SupportGPT conversations."""

from datetime import datetime
import logging
from typing import cast
from uuid import uuid4

import streamlit as st

from chatbot.models import ChatMessage, MessageRole
from database.database import ChatHistoryDatabaseError, save_message

logger = logging.getLogger(__name__)


def initialize_session() -> None:
    """Initialize the state keys required by the application exactly once."""
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("page", "Support Chat")
    for message in st.session_state.messages:
        message.setdefault("id", uuid4().hex)


def get_messages() -> list[ChatMessage]:
    """Return the active conversation transcript from the current session."""
    return cast(list[ChatMessage], st.session_state.messages)


def add_message(role: MessageRole, content: str) -> ChatMessage:
    """Create and persist a timestamped message in the active conversation."""
    message: ChatMessage = {
        "id": uuid4().hex,
        "role": role,
        "content": content,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    get_messages().append(message)
    try:
        save_message(role, content, message["timestamp"])
    except (ChatHistoryDatabaseError, ValueError) as error:
        logger.error("Could not persist chat message: %s", type(error).__name__)
    return message


def clear_conversation() -> None:
    """Remove all messages from the active session conversation."""
    st.session_state.messages = []
