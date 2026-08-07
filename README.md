# SupportGPT – AI Customer Support Assistant

SupportGPT is a modular Streamlit application for professional AI-assisted
customer support. It includes a dark chat experience, Gemini 2.5 Flash,
session-based conversation history, logging, and PDF-grounded retrieval.

## Requirements

- Python 3.11+
- A Gemini API key from Google AI Studio

## Setup

1. Create and activate a Python 3.11+ virtual environment.
2. Install packages with `pip install -r requirements.txt`.
3. Download the English spaCy model with `python -m spacy download en_core_web_sm`.
4. Copy `.env.example` to `.env`.
5. Set `GEMINI_API_KEY` in `.env`; never commit this file.
6. Start SupportGPT with `streamlit run app.py`.

Place approved PDF knowledge sources in `data/pdfs/`. SupportGPT automatically
creates or refreshes the local FAISS index when the source files change.

## Phase 1 capabilities

- Responsive, dark ChatGPT-style Streamlit interface
- Timestamped chat history in `st.session_state`
- Gemini 2.5 Flash with a customer-support system instruction
- A ten-message context window on every model request
- Safe, user-facing errors and application logs in `logs/supportgpt.log`
- Semantic top-five PDF retrieval using PyPDF, LangChain, Sentence Transformers,
  and FAISS
- SQLite-backed analytics for chat volume, response length, and frequent questions

## Architecture

```text
SupportGPT/
├── chatbot/       # Gemini client, message domain model, compatibility service
├── database/      # SQLite persistence contracts (implementation deferred)
├── nlp/           # Text preprocessing contracts (implementation deferred)
├── vectorstore/   # PDF chunking, embeddings, and persistent FAISS retrieval
├── utils/         # Session state, UI, logging, and export contracts
├── data/pdfs/     # Approved source documents for a future RAG pipeline
├── static/         # Streamlit theme styles
├── logs/           # Runtime application logs
├── app.py          # Streamlit composition root
└── config.py       # Environment-backed application configuration
```

SQLite storage is available through a reusable repository module. NLP
processing, feedback, analytics persistence, and PDF/CSV exports retain clean
extension contracts until their implementation phases begin.
