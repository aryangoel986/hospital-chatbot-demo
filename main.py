from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import RAGPipeline

app = FastAPI(title="FinBot - Fintech Support Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGPipeline()

class QueryRequest(BaseModel):
    query: str
    account_id: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    latency_ms: float
    disclaimer: str

@app.get("/health")
def health():
    return {"status": "ok", "service": "finbot"}

@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if not request.account_id.strip():
        raise HTTPException(status_code=400, detail="Account ID is required")

    try:
        result = rag.query(request.query, request.account_id)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            latency_ms=result["latency_ms"],
            disclaimer="This information is for reference only and does not constitute financial advice."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_documents(file_paths: list[str]):
    if not file_paths:
        raise HTTPException(status_code=400, detail="No file paths provided")

    try:
        count = rag.ingest(file_paths)
        return {"message": f"Successfully ingested {count} financial documents"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/audit-log")
async def get_audit_log(account_id: str):
    # Returns query history for compliance reporting
    return {
        "account_id": account_id,
        "queries": [],
        "note": "Audit logging not yet implemented"
    }
