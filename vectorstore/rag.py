"""Modular PDF retrieval service backed by PyPDF, LangChain, FAISS, and SBERT."""

import json
import logging
import threading
from functools import lru_cache
from typing import Any, TypedDict

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    FAISS_MANIFEST_PATH,
    FAISS_METADATA_PATH,
    PDF_DIRECTORY,
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_TOP_K,
    VECTORSTORE_DIRECTORY,
)

logger = logging.getLogger(__name__)


class RetrievedChunk(TypedDict):
    """A semantically relevant text chunk and its PDF provenance."""

    content: str
    source: str
    page: int


class RetrievalError(RuntimeError):
    """Raised when the local knowledge-base index cannot be created or read."""


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load and cache the Sentence Transformers embedding model."""
    return SentenceTransformer(EMBEDDING_MODEL)


class PDFRetriever:
    """Build, persist, and semantically query a local FAISS PDF index."""

    def __init__(self) -> None:
        """Create an initially unloaded retriever instance."""
        self._index: faiss.Index | None = None
        self._chunks: list[RetrievedChunk] = []
        self._manifest: dict[str, dict[str, int]] | None = None
        self._lock = threading.RLock()

    def retrieve(self, query: str, limit: int = RAG_TOP_K) -> list[RetrievedChunk]:
        """Return up to ``limit`` chunks selected through semantic similarity."""
        if not query.strip() or limit < 1:
            return []

        with self._lock:
            self._ensure_index()
            if self._index is None:
                return []
            query_vector = self._embed([query])
            _, positions = self._index.search(query_vector, min(limit, len(self._chunks)))
            return [self._chunks[position] for position in positions[0] if position >= 0]

    def _ensure_index(self) -> None:
        """Load a current persisted index or rebuild it when PDFs changed."""
        manifest = self._source_manifest()
        if self._manifest == manifest:
            return
        if self._load_index(manifest):
            return
        self._build_index(manifest)

    def _source_manifest(self) -> dict[str, dict[str, int]]:
        """Create a stable source signature from all configured PDF files."""
        PDF_DIRECTORY.mkdir(parents=True, exist_ok=True)
        return {
            pdf.name: {"modified_ns": pdf.stat().st_mtime_ns, "size": pdf.stat().st_size}
            for pdf in sorted(PDF_DIRECTORY.glob("*.pdf"))
        }

    def _load_index(self, manifest: dict[str, dict[str, int]]) -> bool:
        """Load an index only when its trusted local manifest matches current PDFs."""
        if not all(path.exists() for path in (FAISS_INDEX_PATH, FAISS_METADATA_PATH, FAISS_MANIFEST_PATH)):
            return False
        try:
            stored_manifest = json.loads(FAISS_MANIFEST_PATH.read_text(encoding="utf-8"))
            if stored_manifest != manifest:
                return False
            self._index = faiss.read_index(str(FAISS_INDEX_PATH))
            self._chunks = json.loads(FAISS_METADATA_PATH.read_text(encoding="utf-8"))
            self._manifest = manifest
            return True
        except (OSError, ValueError, RuntimeError) as error:
            logger.warning("Could not load persisted FAISS index: %s", type(error).__name__)
            return False

    def _build_index(self, manifest: dict[str, dict[str, int]]) -> None:
        """Extract PDFs, split them through LangChain, embed them, and save FAISS."""
        documents = self._load_pdf_documents()
        if not documents:
            self._index, self._chunks, self._manifest = None, [], manifest
            logger.info("No readable PDFs available for retrieval")
            return

        splitter = RecursiveCharacterTextSplitter(chunk_size=RAG_CHUNK_SIZE, chunk_overlap=RAG_CHUNK_OVERLAP)
        split_documents = splitter.split_documents(documents)
        chunks = [self._chunk_from_document(document) for document in split_documents if document.page_content.strip()]
        if not chunks:
            self._index, self._chunks, self._manifest = None, [], manifest
            return

        vectors = self._embed([chunk["content"] for chunk in chunks])
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self._index, self._chunks, self._manifest = index, chunks, manifest
        self._persist_index()
        logger.info("Built FAISS index with %d chunks from %d PDFs", len(chunks), len(manifest))

    def _load_pdf_documents(self) -> list[Document]:
        """Extract page text from all PDFs using PyPDF without failing the batch."""
        documents: list[Document] = []
        for pdf_path in sorted(PDF_DIRECTORY.glob("*.pdf")):
            try:
                reader = PdfReader(str(pdf_path))
                documents.extend(
                    Document(page_content=page.extract_text() or "", metadata={"source": pdf_path.name, "page": page_number})
                    for page_number, page in enumerate(reader.pages, start=1)
                )
            except Exception as error:
                logger.error("Could not read PDF '%s': %s", pdf_path.name, type(error).__name__)
        return documents

    def _embed(self, texts: list[str]) -> np.ndarray[Any, Any]:
        """Create normalized Sentence Transformer vectors for semantic search."""
        vectors = get_embedding_model().encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(vectors, dtype="float32")

    @staticmethod
    def _chunk_from_document(document: Document) -> RetrievedChunk:
        """Convert a LangChain document to serializable retrieval metadata."""
        return {"content": document.page_content, "source": str(document.metadata["source"]), "page": int(document.metadata["page"])}

    def _persist_index(self) -> None:
        """Persist index and JSON metadata without unsafe pickle deserialization."""
        if self._index is None or self._manifest is None:
            return
        VECTORSTORE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(FAISS_INDEX_PATH))
        FAISS_METADATA_PATH.write_text(json.dumps(self._chunks, ensure_ascii=False), encoding="utf-8")
        FAISS_MANIFEST_PATH.write_text(json.dumps(self._manifest, sort_keys=True), encoding="utf-8")


@lru_cache(maxsize=1)
def get_retriever() -> PDFRetriever:
    """Return the application-wide PDF retriever."""
    return PDFRetriever()


def retrieve_context(query: str, limit: int = RAG_TOP_K) -> str:
    """Return formatted top-three PDF context or an empty string when unavailable."""
    try:
        chunks = get_retriever().retrieve(query, limit)
    except Exception as error:
        logger.error("Knowledge retrieval failed: %s", type(error).__name__)
        return ""
    return "\n\n".join(
        f"Source: {chunk['source']} (page {chunk['page']})\n{chunk['content']}" for chunk in chunks
    )
