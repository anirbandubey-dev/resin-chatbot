"""NLP contracts reserved for the spaCy and NLTK processing phase."""

from typing import Protocol


class TextPreprocessor(Protocol):
    """Defines the future text-normalization boundary used before retrieval."""

    def preprocess(self, text: str) -> str:
        """Normalize a customer query without changing its meaning."""
