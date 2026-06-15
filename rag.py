import time
import faiss
import numpy as np
from typing import Any

EMBEDDING_DIM = 1536
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Financial document categories
FINTECH_DOC_TYPES = [
    "loan_agreement",
    "account_statement",
    "transaction_history",
    "compliance_policy",
    "interest_rate_schedule"
]

class RAGPipeline:
    def __init__(self):
        self.index = faiss.IndexFlatL2(EMBEDDING_DIM)
        self.chunks = []
        self.sources = []
        self.doc_types = []

    def chunk_text(self, text: str, source: str, doc_type: str = "general") -> list[dict]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + CHUNK_SIZE
            chunk = " ".join(words[start:end])
            chunks.append({
                "text": chunk,
                "source": source,
                "doc_type": doc_type
            })
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    def embed(self, text: str) -> np.ndarray:
        # Placeholder for real embedding call
        # In production: openai.embeddings.create(input=text, model="text-embedding-ada-002")
        np.random.seed(abs(hash(text)) % (2**32))
        return np.random.rand(EMBEDDING_DIM).astype("float32")

    def ingest(self, file_paths: list[str]) -> int:
        total = 0
        for path in file_paths:
            with open(path, "r") as f:
                text = f.read()
            doc_type = self._detect_doc_type(path)
            chunks = self.chunk_text(text, path, doc_type)
            for chunk in chunks:
                embedding = self.embed(chunk["text"])
                self.index.add(np.array([embedding]))
                self.chunks.append(chunk["text"])
                self.sources.append(chunk["source"])
                self.doc_types.append(chunk["doc_type"])
                total += 1
        return total

    def _detect_doc_type(self, path: str) -> str:
        path_lower = path.lower()
        for doc_type in FINTECH_DOC_TYPES:
            if doc_type.replace("_", "") in path_lower.replace("_", ""):
                return doc_type
        return "general"

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        if self.index.ntotal == 0:
            return []
        query_embedding = self.embed(query)
        distances, indices = self.index.search(
            np.array([query_embedding]), top_k
        )
        results = []
        for idx in indices[0]:
            if idx < len(self.chunks):
                results.append({
                    "text": self.chunks[idx],
                    "source": self.sources[idx],
                    "doc_type": self.doc_types[idx]
                })
        return results

    def query(self, user_query: str, account_id: str) -> dict[str, Any]:
        start = time.time()
        retrieved = self.retrieve(user_query)

        if not retrieved:
            context = "No financial documents ingested yet."
            source_list = []
        else:
            context = "\n\n".join([r["text"] for r in retrieved])
            source_list = list(set([r["source"] for r in retrieved]))

        # Placeholder for real LLM call
        # In production: pass context + query to OpenAI/Groq
        answer = (
            f"Based on your financial records (Account: {account_id}), "
            f"here is the response to: '{user_query}'. "
            f"Context used: {context[:200]}..."
        )

        latency_ms = (time.time() - start) * 1000
        return {
            "answer": answer,
            "sources": source_list,
            "latency_ms": round(latency_ms, 2)
        }
