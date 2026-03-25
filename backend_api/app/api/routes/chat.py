from fastapi import APIRouter
from pydantic import BaseModel


router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    question: str


@router.post("/chat")
def chat(request: ChatRequest) -> dict[str, str]:
    return {
        "question": request.question,
        "answer": "RAG pipeline placeholder. Connect retrieval and generation here.",
    }
