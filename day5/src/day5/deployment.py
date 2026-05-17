"""
Deployment configuration patterns and helpers.

Provides:
- DockerConfig / generate_docker_compose: programmatic docker-compose generation
- show_gitops_pattern: GitHub Actions CI/CD workflow template
- show_auth_pattern: FastAPI JWT authentication middleware
- show_langsmith_tracing: LangSmith tracing setup
- get_deployment_checklist: production readiness checklist
"""

from dataclasses import dataclass, asdict
from typing import Optional
import json


# ---------------------------------------------------------------------------
# Docker compose generation
# ---------------------------------------------------------------------------

@dataclass
class DockerConfig:
    """Represents a docker-compose service configuration."""
    service_name: str
    image: str
    port_host: int
    port_container: int
    env_vars: list[str]
    depends_on: Optional[list[str]] = None
    healthcheck_path: Optional[str] = None


def generate_docker_compose(services: list[DockerConfig]) -> str:
    """Generate a docker-compose.yml string from service configs."""
    lines = ["version: \"3.9\"", "", "services:"]
    for svc in services:
        lines.append(f"  {svc.service_name}:")
        lines.append(f"    image: {svc.image}")
        lines.append(f"    ports:")
        lines.append(f"      - \"{svc.port_host}:{svc.port_container}\"")
        if svc.env_vars:
            lines.append(f"    environment:")
            for env in svc.env_vars:
                lines.append(f"      - {env}")
        if svc.depends_on:
            lines.append(f"    depends_on:")
            for dep in svc.depends_on:
                lines.append(f"      - {dep}")
        if svc.healthcheck_path:
            lines.append(f"    healthcheck:")
            lines.append(f"      test: [\"CMD\", \"curl\", \"-f\", \"http://localhost:{svc.port_container}{svc.healthcheck_path}\"]")
            lines.append(f"      interval: 30s")
            lines.append(f"      timeout: 10s")
            lines.append(f"      retries: 3")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# GitOps / CI-CD pattern
# ---------------------------------------------------------------------------

def show_gitops_pattern() -> str:
    """Returns a GitHub Actions CI/CD workflow pattern."""
    return '''
# .github/workflows/deploy.yml
name: Deploy RAG Agent

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -e ".[all]"
      - run: pytest tests/ -v

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build and push Docker image
        run: |
          docker build -t my-rag-agent:${{ github.sha }} .
          docker push my-rag-agent:${{ github.sha }}
      - name: Deploy to server
        run: |
          ssh deploy@myserver "cd /app && docker-compose pull && docker-compose up -d"
'''


# ---------------------------------------------------------------------------
# Auth pattern
# ---------------------------------------------------------------------------

def show_auth_pattern() -> str:
    """Returns FastAPI JWT auth middleware pattern."""
    return '''
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()
SECRET_KEY = "your-secret-key"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

@app.post("/ask")
async def ask(request: QueryRequest, user = Depends(verify_token)):
    # Only authenticated users can query
    return await process_query(request.query, user_id=user["sub"])
'''


# ---------------------------------------------------------------------------
# LangSmith tracing pattern
# ---------------------------------------------------------------------------

def show_langsmith_tracing() -> str:
    """Returns LangSmith tracing setup pattern."""
    return '''
import os
from langsmith import Client
from langsmith.wrappers import wrap_openai

# Enable tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]    = "your-langsmith-key"
os.environ["LANGCHAIN_PROJECT"]    = "day5-enterprise-rag"

# All LangChain/LangGraph calls are now automatically traced
# For custom spans:
from langsmith import traceable

@traceable(name="retrieval-agent")
def retrieve_docs(query: str) -> list[str]:
    # This function creates a named span in LangSmith
    return hybrid_searcher.search(query)

@traceable(name="synthesis-agent")
def synthesize(query: str, docs: list[str]) -> str:
    return llm.invoke(f"Answer: {query}\\nContext: {docs}")
'''


# ---------------------------------------------------------------------------
# Deployment checklist
# ---------------------------------------------------------------------------

def get_deployment_checklist() -> list[str]:
    """Production deployment checklist for a RAG agent."""
    return [
        "✅ All tests passing (pytest tests/ -v)",
        "✅ .env variables set (OPENAI_API_KEY, LANGCHAIN_API_KEY)",
        "✅ LangSmith tracing enabled and verified",
        "✅ Docker image builds successfully",
        "✅ /health endpoint returns 200",
        "✅ RAGAS eval score above threshold (faithfulness > 0.6)",
        "✅ Authentication middleware configured",
        "✅ Cost tracking initialized",
        "✅ Rate limiting configured",
        "✅ Logging to stdout (for container log aggregation)",
        "✅ README updated with setup instructions",
        "✅ docker-compose.yml committed to repo",
    ]
