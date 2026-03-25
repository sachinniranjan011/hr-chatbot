from pathlib import Path

from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


DOCUMENTS_DIR = Path("documents")
SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def read_pdf(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def read_docx(file_path: Path) -> str:
    document = Document(str(file_path))
    paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text]
    return "\n".join(paragraphs).strip()


def load_document_text(file_path: Path) -> str:
    if file_path.suffix.lower() == ".pdf":
        return read_pdf(file_path)
    if file_path.suffix.lower() == ".docx":
        return read_docx(file_path)
    raise ValueError(f"Unsupported file type: {file_path.suffix}")


def chunk_documents(documents_dir: Path) -> list[str]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks: list[str] = []
    files = sorted(
        file_path
        for file_path in documents_dir.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        print(f"No PDF or DOCX files found in {documents_dir.resolve()}")
        return chunks

    for file_path in files:
        text = load_document_text(file_path)
        if not text.strip():
            print(f"Skipped empty document: {file_path}")
            continue

        file_chunks = splitter.split_text(text)
        print(f"{file_path}: {len(file_chunks)} chunks")
        chunks.extend(file_chunks)

    return chunks


if __name__ == "__main__":
    all_chunks = chunk_documents(DOCUMENTS_DIR)
    print(f"Total chunks created: {len(all_chunks)}")
