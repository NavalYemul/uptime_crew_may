# Day 5 — Enterprise RAG, Multi-Agent Orchestration & Deployment

Complete project for Day 5 of the LLM Engineering Bootcamp.

## What You'll Learn

- **Multi-agent LangGraph orchestration**: query rewrite → retrieval → synthesis with conditional routing
- **Hybrid search + RRF**: BM25 keyword search fused with dense semantic embeddings via Reciprocal Rank Fusion
- **Streaming responses**: FastAPI SSE (`StreamingResponse`) + Vercel AI SDK `useChat` integration
- **RAGAS proxy evaluation**: faithfulness, answer relevancy, context recall — no LLM API required
- **Production monitoring**: drift detection, cost tracking, LangSmith tracing
- **Deployment**: Docker + docker-compose + GitHub Actions GitOps pipeline

## Quick Start

```bash
# Install dependencies using the shared venv from day3
cd day5-enterprise-rag-deployment
/path/to/.venv/bin/pip install -e .

# Copy and fill in your API keys
cp .env.example .env

# Run all tests
pytest tests/ -v
```

## Project Structure

```
day5-enterprise-rag-deployment/
├── src/day5/
│   ├── multi_agent_orchestrator.py  — LangGraph pipeline + supervisor pattern
│   ├── hybrid_search.py             — BM25, DenseIndex, HybridSearcher, RRF
│   ├── streaming.py                 — StreamChunk, SSE helpers, StreamingOrchestrator
│   ├── evaluation.py                — RAGAS proxies, RAGMonitor, CostTracker
│   └── deployment.py                — DockerConfig, GitOps, auth, LangSmith patterns
├── tests/                           — 54 passing tests
├── notebooks/
│   ├── 01_multi_agent_orchestration.ipynb
│   ├── 02_hybrid_search_rrf.ipynb
│   ├── 03_streaming_frontend.ipynb
│   ├── 04_evaluation_ragas.ipynb
│   ├── 05_monitoring_auth.ipynb
│   └── 06_deployment_capstone.ipynb
└── enterprise-app/                  — Complete production RAG app
    ├── app.py       — FastAPI: /ask, /health, /status
    ├── ui.py        — Streamlit frontend
    ├── rag_pipeline.py  — BM25 + OpenAI embeddings + web fallback
    └── data/        — company_handbook.md, product_faq.txt
```

## Enterprise App

The `enterprise-app/` folder is a complete, runnable RAG application:

```bash
cd enterprise-app
pip install -r requirements.txt

# Set your API key
export OPENAI_API_KEY=sk-...

# Start FastAPI backend
uvicorn app:app --reload --port 8000

# In another terminal, start Streamlit UI
BACKEND_URL=http://127.0.0.1:8000 streamlit run ui.py

# Or run with Docker
docker-compose -f docker.yml up --build
```

## Key Concepts

### Reciprocal Rank Fusion (RRF)

```
RRF(d) = Σ 1 / (k + rank_i(d))    k=60 (Cormack et al., 2009)
```

Documents appearing in both BM25 and dense results get boosted — this is why hybrid search
consistently outperforms either method alone.

### LangGraph Multi-Agent Flow

```
START → query_rewrite → retrieval → [route] → synthesis → END
                                       ↓
                                   api_agent → synthesis → END
```

Conditional routing: if retrieval returns 0 docs, call the API agent first.

### SSE Streaming Format

```
data: {"type": "token", "content": "Hello "}

data: {"type": "token", "content": "world!"}

data: {"type": "done", "content": "", "metadata": {"total_tokens": 2}}

```
