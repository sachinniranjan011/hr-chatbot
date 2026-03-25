import hashlib
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from ingest_documents import DOCUMENTS_DIR, chunk_documents


EMBEDDINGS_DIR = Path("embeddings")
COLLECTION_NAME = "hr_policies"
MODEL_NAME = "all-MiniLM-L6-v2"


def build_and_store_embeddings() -> None:
    chunks = chunk_documents(DOCUMENTS_DIR)
    if not chunks:
        print("No chunks available to embed.")
        return

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(chunks, show_progress_bar=True).tolist()

    client = chromadb.PersistentClient(path=str(EMBEDDINGS_DIR.resolve()))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    ids = [
        f"chunk-{index}-{hashlib.sha256(chunk.encode('utf-8')).hexdigest()[:16]}"
        for index, chunk in enumerate(chunks)
    ]
    metadatas = [{"chunk_index": index} for index in range(len(chunks))]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB collection '{COLLECTION_NAME}'.")
    print(f"ChromaDB persisted to: {EMBEDDINGS_DIR.resolve()}")


if __name__ == "__main__":
    build_and_store_embeddings()
