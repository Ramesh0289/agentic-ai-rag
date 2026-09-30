from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.graph import RAGGraph

app = FastAPI(
    title="Agentic AI RAG Chatbot",
    description="Simple document-grounded RAG API using LangGraph and Pinecone.",
    version="1.0.0",
)

rag_graph = None


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)


class ChatResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: list[str]
    confidence_score: float


@app.on_event("startup")
def startup():
    global rag_graph
    rag_graph = RAGGraph()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if rag_graph is None:
        raise HTTPException(status_code=503, detail="RAG graph is not ready.")

    try:
        result = rag_graph.ask(request.query)
        return ChatResponse(query=request.query, **result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
