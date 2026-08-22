"""Streamlit-independent NVIDIA NIM client integration for SupportGPT."""

import logging
import os
import re
from collections.abc import Iterator, Sequence
from functools import lru_cache

from openai import OpenAI

from chatbot.models import ChatMessage
from config import (
    MAX_HISTORY_MESSAGES,
    NVIDIA_NIM_API_KEY,
    NVIDIA_NIM_BASE_URL,
    NVIDIA_NIM_MAX_RETRIES,
    NVIDIA_NIM_MAX_TOKENS,
    NVIDIA_NIM_MODEL,
    NVIDIA_NIM_REASONING_EFFORT,
    NVIDIA_NIM_TEMPERATURE,
    NVIDIA_NIM_TIMEOUT_SECONDS,
    SYSTEM_PROMPT,
)
from vectorstore.rag import retrieve_context

logger = logging.getLogger(__name__)


class NIMServiceError(RuntimeError):
    """Raised when SupportGPT cannot obtain an NVIDIA NIM response safely."""


@lru_cache(maxsize=1)
def get_nim_client() -> OpenAI:
    """Create one reusable client so HTTP connections can be kept warm."""
    if not NVIDIA_NIM_API_KEY:
        raise NIMServiceError("NVIDIA NIM API key is not configured")
    return OpenAI(
        api_key=NVIDIA_NIM_API_KEY,
        base_url=NVIDIA_NIM_BASE_URL,
        timeout=NVIDIA_NIM_TIMEOUT_SECONDS,
        max_retries=NVIDIA_NIM_MAX_RETRIES,
    )


def _to_nim_messages(history: Sequence[ChatMessage]) -> list[dict[str, str]]:
    """Convert recent Streamlit transcript messages to OpenAI-style messages."""
    recent_history = history[-MAX_HISTORY_MESSAGES:]
    return [
        {
            "role": message["role"],
            "content": message["content"],
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


def _build_messages(user_input: str, history: list[ChatMessage]) -> list[dict[str, str]]:
    """Build a compact, OpenAI-compatible conversation for NIM."""
    prompt = user_input.strip()
    context = retrieve_context(prompt)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *_to_nim_messages(history),
        {"role": "user", "content": _build_prompt(prompt, context)},
    ]


def _build_completion_kwargs(messages: list[dict[str, str]], *, stream: bool) -> dict[str, object]:
    """Build a minimal NIM request and opt into extras only by explicit allowlist.

    NVIDIA-hosted models do not all accept the same optional OpenAI-compatible
    fields. The current DeepSeek V4 Flash 0731 request intentionally matches the
    confirmed working request and leaves ``reasoning_effort`` out. To enable it
    for a model after verifying provider support, set
    ``NVIDIA_NIM_REASONING_SUPPORTED_MODELS`` to a comma-separated model list.
    """
    request: dict[str, object] = {
        "model": NVIDIA_NIM_MODEL,
        "messages": messages,
        "max_tokens": NVIDIA_NIM_MAX_TOKENS,
        "temperature": NVIDIA_NIM_TEMPERATURE,
        "stream": stream,
    }

    supported_models = {
        model.strip()
        for model in os.getenv("NVIDIA_NIM_REASONING_SUPPORTED_MODELS", "").split(",")
        if model.strip()
    }
    if NVIDIA_NIM_MODEL in supported_models and NVIDIA_NIM_REASONING_EFFORT:
        request["reasoning_effort"] = NVIDIA_NIM_REASONING_EFFORT
    return request


def _iter_streamlit_chunks(response_text: str) -> Iterator[str]:
    """Yield complete words so Streamlit can render a response progressively."""
    yield from re.findall(r"\S+\s*", response_text)


def generate_response_stream(user_input: str, history: list[ChatMessage]) -> Iterator[str]:
    """Yield an NVIDIA NIM response in chunks for Streamlit's streaming UI.

    DeepSeek V4 Flash's hosted SSE route can time out before returning headers,
    while its standard chat-completion route is confirmed working. The model
    response is therefore requested normally and streamed to the UI in small,
    complete-word chunks.
    """
    prompt = user_input.strip()
    if not prompt:
        yield "Please enter a message so I can help."
        return
    if not NVIDIA_NIM_API_KEY:
        logger.warning("NIM request skipped because NVIDIA_NIM_API_KEY is not configured")
        raise NIMServiceError("NVIDIA NIM API key is not configured")

    try:
        response = get_nim_client().chat.completions.create(
            **_build_completion_kwargs(_build_messages(prompt, history), stream=False)
        )
    except NIMServiceError:
        raise
    except Exception as error:
        logger.exception(
            "NVIDIA NIM response generation failed (%s): %s",
            type(error).__name__,
            error,
        )
        print(f"[NVIDIA NIM] {type(error).__name__}: {error}", flush=True)
        import traceback

        traceback.print_exc()
        raise NIMServiceError("NVIDIA NIM request failed") from error

    response_text = response.choices[0].message.content if response.choices else None
    if not response_text:
        logger.warning("NVIDIA NIM returned an empty response")
        raise NIMServiceError("NVIDIA NIM returned an empty response")
    yield from _iter_streamlit_chunks(response_text)


def generate_response(user_input: str, history: list[ChatMessage]) -> str:
    """Return a complete NVIDIA NIM response for non-streaming callers.

    Args:
        user_input: The latest customer message.
        history: Timestamped Streamlit chat messages preceding ``user_input``.

    Returns:
        The assistant's text response only. The Streamlit UI uses
        :func:`generate_response_stream` so text appears immediately.

    Raises:
        NIMServiceError: If NIM configuration or the API is unavailable.
    """
    return "".join(generate_response_stream(user_input, history))
