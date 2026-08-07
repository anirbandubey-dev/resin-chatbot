"""Reusable Streamlit presentation components for SupportGPT."""

from pathlib import Path
from typing import Literal, cast

import streamlit as st
import plotly.graph_objects as go

from chatbot.models import ChatMessage
from config import APP_ICON, APP_TITLE, PROJECT_ROOT
from database.analytics import AnalyticsDatabaseError, get_chat_analytics
from database.database import ChatHistoryDatabaseError, FeedbackRating, get_feedback, save_feedback

Page = Literal["Support Chat", "Knowledge Base", "Analytics", "Settings", "New Chat"]
_NAVIGATION_PAGES: tuple[Page, ...] = ("Support Chat", "Knowledge Base", "Analytics", "Settings")


def inject_styles() -> None:
    """Load the local dark-theme stylesheet."""
    css = (PROJECT_ROOT / "static" / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_sidebar() -> Page:
    """Render primary navigation and return the selected destination."""
    with st.sidebar:
        st.markdown(f"<div class='brand'><span>{APP_ICON}</span>{APP_TITLE}</div>", unsafe_allow_html=True)
        st.caption("AI customer support assistant")
        if st.button("＋ New Chat", use_container_width=True, type="primary"):
            return "New Chat"
        st.divider()
        selection = st.radio("Workspace", _NAVIGATION_PAGES, label_visibility="collapsed")
        st.divider()
        st.markdown("<div class='sidebar-footer'>● Service status: <b>Ready</b><br><small>Gemini 2.5 Flash</small></div>", unsafe_allow_html=True)
    return cast(Page, selection)


def render_empty_state() -> None:
    """Display a useful welcome state before the first customer message."""
    st.markdown("<section class='welcome'><div class='welcome-icon'>🤖</div><h1>How can I help today?</h1><p>Ask about orders, billing, account access, or a customer concern.</p></section>", unsafe_allow_html=True)
    for column, text in zip(st.columns(3), ("Track an order", "Explain a charge", "Reset account access")):
        column.markdown(f"<div class='suggestion'>{text}</div>", unsafe_allow_html=True)


def render_message(message: ChatMessage) -> None:
    """Render one timestamped conversation message as an accessible chat bubble."""
    avatar = "🧑" if message["role"] == "user" else APP_ICON
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        st.caption(message["timestamp"])
        if message["role"] == "assistant":
            render_feedback_controls(message)


def render_feedback_controls(message: ChatMessage) -> None:
    """Render and persist one positive or negative rating for an AI response."""
    try:
        current_rating = get_feedback(message["id"])
    except ChatHistoryDatabaseError:
        current_rating = None

    positive, negative, status = st.columns([1, 1, 8])
    if positive.button("👍", key=f"feedback-positive-{message['id']}", help="Helpful response"):
        _save_feedback(message, 1)
    if negative.button("👎", key=f"feedback-negative-{message['id']}", help="Unhelpful response"):
        _save_feedback(message, -1)
    if current_rating == 1:
        status.caption("Marked helpful")
    elif current_rating == -1:
        status.caption("Marked unhelpful")


def _save_feedback(message: ChatMessage, rating: FeedbackRating) -> None:
    """Save a rating without allowing feedback storage failures to break chat."""
    try:
        save_feedback(message["id"], rating, message["content"], message["timestamp"])
    except (ChatHistoryDatabaseError, ValueError):
        st.error("We couldn't save your feedback. Please try again.")
    else:
        st.toast("Thanks for your feedback.")


def render_knowledge_base() -> None:
    """Render the Phase 1 knowledge-base readiness view."""
    st.title("Knowledge base")
    st.caption("Trusted PDF retrieval will be available in a later phase.")
    st.info("Add approved support documents to `data/pdfs/` to prepare them for indexing.")


def render_analytics() -> None:
    """Render live dashboard metrics derived from SQLite chat history."""
    st.title("Analytics")
    st.caption("Conversation activity based on the stored SQLite chat history.")
    try:
        analytics = get_chat_analytics()
    except AnalyticsDatabaseError:
        st.error("Analytics are temporarily unavailable. Please try again later.")
        return

    total_chats, average_length, positive_feedback, negative_feedback = st.columns(4)
    total_chats.metric("Total chats", analytics.total_chats)
    average_length.metric("Average response length", f"{analytics.average_response_length:.0f} characters")
    positive_feedback.metric("Helpful", analytics.positive_feedback)
    negative_feedback.metric("Unhelpful", analytics.negative_feedback)

    st.subheader("Daily chat count")
    if analytics.daily_chat_counts:
        chart = go.Figure(
            go.Bar(
                x=[item.date for item in analytics.daily_chat_counts],
                y=[item.count for item in analytics.daily_chat_counts],
                marker_color="#7765ff",
                hovertemplate="%{x}: %{y} chats<extra></extra>",
            )
        )
        chart.update_layout(
            height=300,
            margin={"l": 0, "r": 0, "t": 20, "b": 0},
            paper_bgcolor="#0b1020",
            plot_bgcolor="#11192b",
            font={"color": "#e7ecf8"},
            xaxis_title=None,
            yaxis_title="Chats",
        )
        st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No customer messages have been saved to SQLite yet.")

    st.subheader("Most frequent questions")
    if analytics.frequent_questions:
        st.dataframe(
            [{"Question": item.question, "Count": item.count} for item in analytics.frequent_questions],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("Frequent questions will appear after customer messages are stored.")


def render_settings() -> None:
    """Render secure Gemini configuration guidance."""
    st.title("Settings")
    st.caption("Configure SupportGPT services without exposing credentials in the interface.")
    st.subheader("Gemini")
    st.code("GEMINI_API_KEY=your_api_key", language="bash")
    st.write("Store the key in `.env`, then restart the application.")
