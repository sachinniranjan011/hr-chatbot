from pathlib import Path

import chromadb
from docx import Document
from fastapi.testclient import TestClient
from langchain.text_splitter import RecursiveCharacterTextSplitter

from backend_api.app.main import app
from backend_api.app.services import rag_service
from ingest_documents import chunk_documents


class FakeSentenceTransformer:
    def __init__(self, _: str) -> None:
        pass

    def encode(self, inputs, show_progress_bar: bool = False):
        def embed(text: str) -> list[float]:
            normalized = text.lower()
            if "leave" in normalized:
                return [1.0, 0.0, 0.0]
            if "benefits" in normalized:
                return [0.0, 1.0, 0.0]
            return [0.0, 0.0, 1.0]

        if isinstance(inputs, str):
            return embed(inputs)
        return [embed(item) for item in inputs]


def _create_sample_docx(documents_dir: Path) -> None:
    documents_dir.mkdir(parents=True, exist_ok=True)
    document = Document()
    text = "Leave policy allows annual leave and sick leave for employees. " * 220
    document.add_paragraph(text)
    document.save(documents_dir / "leave_policy.docx")


def _create_sample_collection(persist_dir: Path):
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(name="hr_policies")
    documents = [
        "Employees receive 20 days of annual leave each year.",
        "Health benefits begin after probation is completed.",
        "Working hours are from 9 AM to 5 PM Monday through Friday.",
    ]
    embeddings = FakeSentenceTransformer("all-MiniLM-L6-v2").encode(documents)
    collection.upsert(
        ids=["doc-1", "doc-2", "doc-3"],
        documents=documents,
        embeddings=embeddings,
    )
    return collection


def test_document_chunking_produces_chunks_under_600_tokens(tmp_path: Path) -> None:
    _create_sample_docx(tmp_path)
    chunks = chunk_documents(tmp_path)

    assert chunks

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=50,
    )
    for chunk in chunks:
        assert splitter._length_function(chunk) < 600


def test_chromadb_returns_result_for_hr_query(tmp_path: Path) -> None:
    collection = _create_sample_collection(tmp_path / "embeddings")
    query = "what is the leave policy"
    query_embedding = FakeSentenceTransformer("all-MiniLM-L6-v2").encode(query)

    results = collection.query(query_embeddings=[query_embedding], n_results=3)

    assert results["documents"]
    assert len(results["documents"][0]) >= 1


def test_answer_hr_question_uses_mocked_watsonx(tmp_path: Path, monkeypatch) -> None:
    _create_sample_collection(tmp_path / "embeddings")
    monkeypatch.setattr(
        rag_service.settings,
        "chroma_persist_directory",
        str(tmp_path / "embeddings"),
    )
    monkeypatch.setattr(rag_service, "SentenceTransformer", FakeSentenceTransformer)

    class FakeWatsonxLLM:
        def generate(self, prompt: str) -> str:
            assert "You are an HR assistant. Use the context below to answer." in prompt
            assert "Question: what is the leave policy" in prompt
            assert "Employees receive 20 days of annual leave each year." in prompt
            return "Employees receive 20 days of annual leave each year."

    monkeypatch.setattr(rag_service, "WatsonxLLM", FakeWatsonxLLM)

    answer = rag_service.answer_hr_question("what is the leave policy")

    assert answer == "Employees receive 20 days of annual leave each year."


def test_health_endpoint_returns_200() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
