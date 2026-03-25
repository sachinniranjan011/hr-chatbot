# HR Policy Chatbot

Starter project structure for an HR Policy Chatbot using a Retrieval-Augmented Generation (RAG) architecture.

## Project Layout

```text
HR Chatbot/
|-- documents/              # HR policy source files
|   |-- raw/
|   `-- processed/
|-- embeddings/             # Local vector store / embedding artifacts
|-- backend_api/            # FastAPI backend for ingestion and chat
|   `-- app/
|       |-- api/
|       |   `-- routes/
|       |-- core/
|       `-- services/
|-- frontend_ui/            # Frontend UI placeholder
`-- requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt
uvicorn backend_api.app.main:app --reload
```

## Document Chunking

Run the ingestion helper to read PDF and DOCX files from `documents/` and split them into 500-token chunks with 50-token overlap:

```bash
python ingest_documents.py
```

## Build Embeddings

Run the vector store builder to embed those chunks with `sentence-transformers/all-MiniLM-L6-v2` and save them to ChromaDB in `embeddings/`:

```bash
python build_embeddings.py
```
