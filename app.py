"""Streamlit entry point for Resin."""

import logging

import streamlit as st

from chatbot.chatbot import NIMServiceError, generate_response_stream
from config import APP_ICON, APP_TITLE, MAX_HISTORY_MESSAGES, NIM_ERROR_MESSAGE
from utils.logging_config import configure_logging
from utils.state import add_message, clear_conversation, get_messages, initialize_session
from utils.ui import (
    inject_styles,
    render_analytics,
    render_empty_state,
    render_knowledge_base,
    render_message,
    render_settings,
    render_sidebar,
)

logger = logging.getLogger(__name__)


def render_chat() -> None:
    """Render history, collect a prompt, and append the NVIDIA NIM response."""
    messages = get_messages()
    if messages:
        for message in messages:
            render_message(message)
    else:
        render_empty_state()

    prompt = st.chat_input("Message Resin…")
    if not prompt:
        return

    context = messages[-MAX_HISTORY_MESSAGES:]
    render_message(add_message("user", prompt))
    with st.chat_message("assistant", avatar=APP_ICON):
        try:
            response = st.write_stream(generate_response_stream(prompt, context))
        except NIMServiceError:
            logger.warning("NVIDIA NIM request could not be completed")
            response = NIM_ERROR_MESSAGE
            st.error("Resin couldn't connect right now. Please try again.")
            st.markdown(response)
        except Exception:
            logger.exception("Unexpected chat response failure")
            response = NIM_ERROR_MESSAGE
            st.error("Resin ran into a problem while processing your message. Please try again.")
            st.markdown(response)
    add_message("assistant", response)


def render_page(page: str) -> None:
    """Render the selected application destination."""
    if page == "Knowledge Base":
        render_knowledge_base()
    elif page == "Analytics":
        render_analytics()
    elif page == "Settings":
        render_settings()
    else:
        render_chat()


def main() -> None:
    """Configure and run the Resin Streamlit application."""
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide", initial_sidebar_state="expanded")
    configure_logging()
    initialize_session()
    inject_styles()

    page = render_sidebar()
    if page == "New Chat":
        clear_conversation()
        st.rerun()
    render_page(page)


if __name__ == "__main__":
    main()
