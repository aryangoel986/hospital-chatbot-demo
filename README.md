# Hospital Support Chatbot

An AI-powered chatbot for HealthTech Solutions that helps hospital staff query internal medical documents using a RAG pipeline.

## Features
- Document ingestion and chunking
- Vector similarity search using FAISS
- FastAPI backend with streaming responses
- Sub-2-second response time

## Tech Stack
- Python, FastAPI, LangChain
- FAISS for vector storage
- OpenAI embeddings
- Uvicorn

## Setup
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## Usage
POST /chat with a JSON body:
```json
{ "query": "What is the post-op protocol for appendectomy?" }
```

## Architecture
User query → embedding → FAISS similarity search → top 3 chunks → LLM → response
