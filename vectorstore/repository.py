"""Vector-store contracts reserved for the future FAISS RAG implementation."""

from typing import Protocol, Sequence


class VectorRepository(Protocol):
    """Defines the future interface for PDF embedding storage and retrieval."""

    def index(self, document_id: str, chunks: Sequence[str]) -> None:
        """Index document chunks with their embeddings."""

    def search(self, query: str, limit: int = 5) -> list[str]:
        """Return the most relevant document chunks for a query."""
