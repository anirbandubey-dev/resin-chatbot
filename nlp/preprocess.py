"""spaCy-based text preprocessing utilities for future SupportGPT retrieval."""

import logging
from functools import lru_cache

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token

from config import SPACY_MODEL

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_nlp(model_name: str = SPACY_MODEL) -> Language:
    """Load and cache the configured spaCy language pipeline.

    A blank English pipeline keeps local development and unit tests operational
    when the optional ``en_core_web_sm`` model has not been downloaded yet.
    """
    try:
        return spacy.load(model_name)
    except OSError:
        logger.warning("spaCy model '%s' is unavailable; using a blank English pipeline", model_name)
        return spacy.blank("en")


def is_content_token(token: Token) -> bool:
    """Return whether a token should remain after non-content filtering."""
    return not (token.is_space or token.is_punct or token.is_stop or not token.is_alpha)


def normalized_lemma(token: Token) -> str:
    """Return a lowercase lemma, falling back to the normalized token text."""
    lemma = token.lemma_.strip().lower()
    return token.lower_ if not lemma or lemma == "-pron-" else lemma


def clean_tokens(document: Doc) -> list[str]:
    """Remove stopwords and non-content tokens, then lemmatize the remainder."""
    return [normalized_lemma(token) for token in document if is_content_token(token)]


def preprocess_text(text: str, nlp: Language | None = None) -> str:
    """Lowercase, tokenize, remove stopwords, and lemmatize input text.

    Args:
        text: Raw customer or knowledge-base text.
        nlp: Optional injected spaCy pipeline for deterministic unit tests.

    Returns:
        Whitespace-separated cleaned text, or an empty string for empty input.
    """
    if not text or not text.strip():
        return ""

    pipeline = nlp or get_nlp()
    return " ".join(clean_tokens(pipeline(text.lower())))
