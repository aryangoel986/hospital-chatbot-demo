from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import RAGPipeline

app = FastAPI(title="Hospital Support Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGPipeline()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    latency_ms: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        result = rag.query(request.query)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            latency_ms=result["latency_ms"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_documents(file_paths: list[str]):
    if not file_paths:
        raise HTTPException(status_code=400, detail="No file paths provided")
    
    try:
        count = rag.ingest(file_paths)
        return {"message": f"Successfully ingested {count} documents"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
