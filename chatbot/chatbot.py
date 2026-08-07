"""Streamlit-independent Gemini client integration for SupportGPT."""

import logging
from typing import Any, Sequence

from chatbot.models import ChatMessage
from config import GEMINI_API_KEY, GEMINI_MODEL, MAX_HISTORY_MESSAGES, SYSTEM_PROMPT
from vectorstore.rag import retrieve_context

logger = logging.getLogger(__name__)


class GeminiServiceError(RuntimeError):
    """Raised when SupportGPT cannot obtain a Gemini response safely."""


def _to_gemini_messages(history: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    """Convert recent Streamlit transcript messages to Gemini content objects."""
    recent_history = history[-MAX_HISTORY_MESSAGES:]
    return [
        {
            "role": "model" if message["role"] == "assistant" else "user",
            "parts": [{"text": message["content"]}],
        }
        for message in recent_history
    ]


def _build_prompt(user_input: str, retrieved_context: str) -> str:
    """Attach trusted RAG context to the current customer message when found."""
    if not retrieved_context:
        return user_input
    return f"""Reference context from the knowledge base:
{retrieved_context}

Customer message:
{user_input}"""


def generate_response(user_input: str, history: list[ChatMessage]) -> str:
    """Generate an assistant response using Gemini and the last ten messages.

    Args:
        user_input: The latest customer message.
        history: Timestamped Streamlit chat messages preceding ``user_input``.

    Returns:
        The assistant's text response only.

    Raises:
        GeminiServiceError: If Gemini configuration or the API is unavailable.
    """
    prompt = user_input.strip()
    if not prompt:
        return "Please enter a message so I can help."
    if not GEMINI_API_KEY:
        logger.warning("Gemini request skipped because GEMINI_API_KEY is not configured")
        raise GeminiServiceError("Gemini API key is not configured")

    try:
        from google import genai

        client = genai.Client(api_key=GEMINI_API_KEY)
        context = retrieve_context(prompt)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[*_to_gemini_messages(history), {"role": "user", "parts": [{"text": _build_prompt(prompt, context)}]}],
            config={"system_instruction": SYSTEM_PROMPT},
        )
    except Exception as error:
        import traceback

        logger.exception("Gemini response generation failed")
        traceback.print_exc()

        raise GeminiServiceError("Gemini request failed") from error

    if not getattr(response, "text", None):
        logger.warning("Gemini returned an empty response")
        raise GeminiServiceError("Gemini returned an empty response")
    return response.text
