"""Reusable Streamlit presentation components for Resin."""

from html import escape
from pathlib import Path
from typing import Literal, cast

import streamlit as st
import plotly.graph_objects as go

from chatbot.models import ChatMessage
from config import APP_ICON, APP_TITLE, PDF_DIRECTORY, PROJECT_ROOT
from utils.state import get_user_name, set_user_name
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
        st.caption("AI Assistant")
        if st.button("＋ New chat", width="stretch", type="primary"):
            return "New Chat"
        st.divider()
        selection = st.radio("Workspace", _NAVIGATION_PAGES, label_visibility="collapsed")
        st.divider()
        st.markdown("<div class='sidebar-footer'>● Service status: <b>Ready</b><br><small>Gemini • Online</small></div>", unsafe_allow_html=True)
    return cast(Page, selection)


def render_empty_state() -> None:
    """Render onboarding or a personalized welcome before the first message."""
    user_name = get_user_name()
    if not user_name:
        st.markdown(
            "<section class='welcome welcome-onboarding'>"
            "<div class='welcome-icon'>👋</div>"
            "<p class='eyebrow'>YOUR PERSONAL AI ASSISTANT</p>"
            "<h1>Hello! Welcome to <span>Resin.</span></h1>"
            "<p>Quick, reliable answers from your connected knowledge base.</p>"
            "<div class='capabilities'><span>✦ Gemini AI</span><span>◉ RAG + FAISS</span><span>▣ PDF Knowledge Base</span></div>"
            "</section>",
            unsafe_allow_html=True,
        )
        with st.form("name_onboarding", clear_on_submit=True):
            st.subheader("What should I call you?")
            name = st.text_input(
                "Your name",
                placeholder="Enter your name...",
                max_chars=60,
                key="resin_name_input",
                label_visibility="collapsed",
            )
            if st.form_submit_button("Continue →", type="primary", width="stretch"):
                cleaned_name = name.strip()
                if not cleaned_name:
                    st.warning("Please enter your name to continue.")
                elif len(cleaned_name) > 60:
                    st.warning("Please use a name with 60 characters or fewer.")
                else:
                    set_user_name(cleaned_name)
                    st.rerun()
        return

    st.markdown(
        "<section class='welcome welcome-personalized'>"
        "<div class='welcome-icon'>✦</div>"
        f"<h1>👋 Welcome, <span>{escape(user_name)}.</span></h1>"
        "<p>I'm Resin. How can I help you today?</p>"
        "</section>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='suggestions-label'>Try one of these common topics</div>", unsafe_allow_html=True)
    suggestions = (
        ("📦", "Track an order", "Get help with orders and delivery questions."),
        ("💳", "Explain a charge", "Understand billing and payment concerns."),
        ("🔐", "Account support", "Get help with account access and common issues."),
    )
    for column, (icon, title, description) in zip(st.columns(3), suggestions):
        column.markdown(
            f"<div class='suggestion'><span class='suggestion-icon'>{icon}</span>"
            f"<h3>{title}</h3><p>{description}</p></div>",
            unsafe_allow_html=True,
        )


def render_message(message: ChatMessage) -> None:
    """Render one Resin conversation message."""

    is_user = message["role"] == "user"

    avatar = "👤" if is_user else APP_ICON
    with st.chat_message(
        message["role"],
        avatar=avatar,
    ):
        st.markdown(message["content"])

        sender = "You" if is_user else "Resin"
        st.caption(f"{sender} · {message['timestamp']}")

        if not is_user:
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
    """Render the available PDFs used by the local knowledge base."""
    pdf_files = sorted(PDF_DIRECTORY.glob("*.pdf"), key=lambda path: path.name.lower())
    st.title("Knowledge base")
    st.caption("Documents available to Resin's local PDF retrieval system.")
    st.metric("Available PDF documents", len(pdf_files))
    if not pdf_files:
        st.info("Add approved PDF documents to `data/pdfs/` to make them available for retrieval.")
        return

    st.subheader("Available documents")
    for pdf_file in pdf_files:
        st.markdown(f"<div class='document-row'>▣ <span>{pdf_file.name}</span></div>", unsafe_allow_html=True)


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
    st.caption("Configure Resin services without exposing credentials in the interface.")
    st.subheader("Gemini")
    st.code("GEMINI_API_KEY=your_api_key", language="bash")
    st.write("Store the key in `.env`, then restart the application.")
