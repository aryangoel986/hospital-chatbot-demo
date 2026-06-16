# FinBot — Fintech Support Chatbot

An AI-powered chatbot for FinanceFlow Inc. that helps customers query their transaction history, loan status, and account information using a RAG pipeline.

## Features
- Financial document ingestion and chunking
- Vector similarity search using FAISS
- FastAPI backend with real-time responses
- PCI-DSS compliant data handling

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
{ "query": "What is the status of my loan application #LN-2041?" }
```

## Architecture
User query → embedding → FAISS similarity search → top 3 chunks → LLM → response

## Compliance
This system follows PCI-DSS guidelines for handling financial data.
No raw card data is stored or transmitted through this service.
