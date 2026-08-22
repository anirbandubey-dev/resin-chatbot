"""Compatibility service for callers using the earlier service boundary."""

from chatbot.chatbot import generate_response
from chatbot.models import ChatMessage


class SupportAssistant:
    """Delegate chat interface requests to the NVIDIA NIM integration layer."""

    def respond(self, prompt: str, history: list[ChatMessage]) -> str:
        """Return an NVIDIA NIM response while retaining the existing UI contract."""
        return generate_response(prompt, history)
