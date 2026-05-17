"""Local-first RAG pipeline with PDF ingestion and web-search fallback."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rank_bm25 import BM25Okapi

for _root in [Path.cwd(), *Path.cwd().resolve().parents]:
    _env = _root / ".env"
    if _env.is_file():
        load_dotenv(_env)
        break

DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", "data"))
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
LOCAL_TOP_K = 4
MAX_TOOL_STEPS = 3
TIME_SENSITIVE_TERMS = {
    "breaking",
    "current",
    "currently",
    "latest",
    "live",
    "news",
    "recent",
    "recently",
    "today",
    "update",
    "updated",
    "yesterday",
}
DEMO_DOCUMENTS = [
    (
        "demo_overview",
        "This application is a local-first retrieval demo. It searches local documents first "
        "and only uses a web-search tool when the answer is missing, incomplete, or needs "
        "current information.",
    ),
    (
        "databricks_note",
        "Databricks is a unified analytics platform for data engineering, machine learning, "
        "and AI workloads. Teams often use it for lakehouse-style analytics and collaborative "
        "notebook workflows.",
    ),
    (
        "langchain_note",
        "LangChain helps developers build applications with large language models, including "
        "tool calling, retrieval pipelines, and structured prompting.",
    ),
    (
        "fastapi_note",
        "FastAPI is a Python framework for building APIs with automatic validation, type hints, "
        "and interactive documentation through Swagger UI and ReDoc.",
    ),
    (
        "retrieval_note",
        "Hybrid retrieval combines lexical search such as BM25 with semantic vector similarity. "
        "This often improves recall over using only one retrieval strategy.",
    ),
]
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_RRF_CAP = (1 / 60) + (1 / 60)


@dataclass
class DocumentChunk:
    content: str
    source: str
    kind: str


@dataclass
class SearchHit:
    content: str
    source: str
    kind: str
    confidence: float


@dataclass
class KnowledgeBase:
    chunks: list[DocumentChunk]
    bm25: BM25Okapi
    matrix: np.ndarray
    loaded_files: list[str]
    fallback_demo_docs: bool


_cache: dict[str, dict[str, Any]] = {}
_knowledge_base: KnowledgeBase | None = None
_llm: ChatOpenAI | None = None
_embeddings: OpenAIEmbeddings | None = None


def _cache_key(query: str) -> str:
    return hashlib.md5(query.encode("utf-8")).hexdigest()


def _require_openai() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your environment or a local .env file."
        )


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _require_openai()
        _llm = ChatOpenAI(
            model=os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            temperature=0,
        )
    return _llm


def _get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _require_openai()
        _embeddings = OpenAIEmbeddings()
    return _embeddings


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def _chunk_text(text: str, source: str, kind: str) -> list[DocumentChunk]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks: list[DocumentChunk] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + CHUNK_SIZE)
        chunk = cleaned[start:end]

        if end < len(cleaned):
            last_space = chunk.rfind(" ")
            if last_space > CHUNK_SIZE // 2:
                end = start + last_space
                chunk = cleaned[start:end]

        chunks.append(DocumentChunk(content=chunk.strip(), source=source, kind=kind))

        if end >= len(cleaned):
            break
        start = max(0, end - CHUNK_OVERLAP)

    return chunks


def _list_local_files() -> list[Path]:
    if not DATA_DIR.exists():
        return []

    allowed_suffixes = {".pdf", ".txt", ".md"}
    files: list[Path] = []
    for path in sorted(DATA_DIR.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() in allowed_suffixes:
            files.append(path)
    return files


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PDF support is unavailable because pypdf is not installed."
        ) from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _read_local_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_local_chunks() -> tuple[list[DocumentChunk], list[str], bool]:
    files = _list_local_files()
    chunks: list[DocumentChunk] = []
    loaded_files: list[str] = []

    for path in files:
        text = _read_local_file(path)
        file_chunks = _chunk_text(
            text=text,
            source=_relative_path(path),
            kind=path.suffix.lower().lstrip("."),
        )
        if not file_chunks:
            continue
        chunks.extend(file_chunks)
        loaded_files.append(_relative_path(path))

    if chunks:
        return chunks, loaded_files, False

    demo_chunks: list[DocumentChunk] = []
    for source, content in DEMO_DOCUMENTS:
        demo_chunks.extend(_chunk_text(content, source=source, kind="demo"))
    return demo_chunks, [], True


def _build_knowledge_base() -> KnowledgeBase:
    chunks, loaded_files, fallback_demo_docs = _load_local_chunks()
    tokenized_docs = [_tokenize(chunk.content) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_docs)
    matrix = np.asarray(
        _get_embeddings().embed_documents([chunk.content for chunk in chunks]),
        dtype=np.float32,
    )
    return KnowledgeBase(
        chunks=chunks,
        bm25=bm25,
        matrix=matrix,
        loaded_files=loaded_files,
        fallback_demo_docs=fallback_demo_docs,
    )


def _get_knowledge_base() -> KnowledgeBase:
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = _build_knowledge_base()
    return _knowledge_base


def _rrf_fusion(bm25_rank: np.ndarray, vector_rank: np.ndarray, k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for rank, idx in enumerate(bm25_rank):
        scores[int(idx)] = scores.get(int(idx), 0.0) + (1 / (k + rank))
    for rank, idx in enumerate(vector_rank):
        scores[int(idx)] = scores.get(int(idx), 0.0) + (1 / (k + rank))
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def _search_local_documents(query: str, top_k: int = LOCAL_TOP_K) -> list[SearchHit]:
    knowledge_base = _get_knowledge_base()
    tokenized_query = _tokenize(query)
    if not tokenized_query:
        return []

    bm25_scores = knowledge_base.bm25.get_scores(tokenized_query)
    bm25_rank = np.argsort(bm25_scores)[::-1][: max(top_k * 3, top_k)]

    query_embedding = np.asarray(
        _get_embeddings().embed_query(query),
        dtype=np.float32,
    )
    distances = np.linalg.norm(knowledge_base.matrix - query_embedding, axis=1)
    vector_rank = np.argsort(distances)[: max(top_k * 3, top_k)]

    fused = _rrf_fusion(bm25_rank, vector_rank)
    query_terms = set(tokenized_query)
    hits: list[SearchHit] = []

    for idx, score in fused[:top_k]:
        chunk = knowledge_base.chunks[idx]
        overlap = len(query_terms & set(_tokenize(chunk.content))) / max(1, len(query_terms))
        normalized_score = min(score / _RRF_CAP, 1.0)
        confidence = round((normalized_score * 0.55) + (overlap * 0.45), 3)
        hits.append(
            SearchHit(
                content=chunk.content,
                source=chunk.source,
                kind=chunk.kind,
                confidence=confidence,
            )
        )
    return hits


def _query_is_time_sensitive(query: str) -> bool:
    return any(term in TIME_SENSITIVE_TERMS for term in _tokenize(query))


def _should_force_web_search(query: str, hits: list[SearchHit]) -> bool:
    if _query_is_time_sensitive(query):
        return True
    if not hits:
        return True
    return hits[0].confidence < 0.35


def _format_local_context(hits: list[SearchHit]) -> str:
    if not hits:
        return "No local context was retrieved."

    sections = []
    for index, hit in enumerate(hits, start=1):
        sections.append(
            f"[Local document {index}] source={hit.source} confidence={hit.confidence}\n{hit.content}"
        )
    return "\n\n".join(sections)


def _dedupe_web_results(results: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in results:
        url = item.get("url", "")
        key = url or item.get("title", "")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _resolve_source_label(local_hits: list[SearchHit], web_results: list[dict[str, str]]) -> str:
    if local_hits and web_results:
        return "local+web_search"
    if web_results:
        return "web_search"
    return "local_knowledge_base"


def _web_search_payload(query: str) -> dict[str, Any]:
    try:
        from ddgs import DDGS
    except ModuleNotFoundError as exc:
        return {
            "query": query,
            "results": [],
            "error": "Web search is unavailable because the ddgs package is not installed.",
            "details": str(exc),
        }

    try:
        raw_results = DDGS(timeout=10).text(
            query,
            region="us-en",
            safesearch="moderate",
            max_results=5,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "query": query,
            "results": [],
            "error": "Web search failed.",
            "details": str(exc),
        }

    results: list[dict[str, str]] = []
    for item in raw_results or []:
        url = item.get("href") or item.get("url") or ""
        if not url:
            continue
        results.append(
            {
                "title": item.get("title", "Untitled result"),
                "url": url,
                "snippet": item.get("body", ""),
            }
        )

    return {"query": query, "results": _dedupe_web_results(results)}


@tool
def web_search(query: str) -> str:
    """Search the public web when the local knowledge base is missing the answer."""

    return json.dumps(_web_search_payload(query), ensure_ascii=True)


def _invoke_web_search_tool(query: str) -> dict[str, Any]:
    raw_output = web_search.invoke({"query": query})
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        return {
            "query": query,
            "results": [],
            "error": "The web_search tool returned an invalid payload.",
            "raw_output": raw_output,
        }


def _run_agent(
    user_query: str,
    local_hits: list[SearchHit],
) -> tuple[str, list[dict[str, str]], str, list[dict[str, str]]]:
    local_context = _format_local_context(local_hits)
    should_force_web_search = _should_force_web_search(user_query, local_hits)
    messages = [
        SystemMessage(
            content=(
                "You are a local-first retrieval assistant. Use local context whenever it answers "
                "the question. If the local context is missing, weak, or the question needs fresh "
                "information, call the web_search tool. Never pretend the local corpus contains "
                "facts it does not contain. If you use web search, include a short Sources section "
                "with the URLs you relied on."
            )
        ),
        HumanMessage(
            content=(
                f"User question:\n{user_query}\n\n"
                f"Local context:\n{local_context}\n\n"
                f"Local context appears sufficient: {'yes' if not should_force_web_search else 'no'}.\n"
                "Answer from local context if it is enough. Otherwise use the web_search tool."
            )
        ),
    ]

    llm_with_tools = _get_llm().bind_tools([web_search])
    web_results: list[dict[str, str]] = []
    tool_calls: list[dict[str, str]] = []
    strategy = "local_only"

    for _ in range(MAX_TOOL_STEPS):
        ai_message = llm_with_tools.invoke(messages)
        messages.append(ai_message)

        if not ai_message.tool_calls:
            if should_force_web_search and not web_results:
                strategy = "forced_web_fallback"
                payload = _invoke_web_search_tool(user_query)
                web_results.extend(payload.get("results", []))
                tool_calls.append(
                    {
                        "tool": "web_search",
                        "query": payload.get("query", user_query),
                        "mode": "forced_fallback",
                    }
                )
                messages.append(
                    HumanMessage(
                        content=(
                            "The application executed the web_search tool because local retrieval "
                            "looked insufficient. Use these search results to answer the question.\n\n"
                            f"{json.dumps(payload, ensure_ascii=True)}"
                        )
                    )
                )
                continue

            return (
                ai_message.content.strip(),
                _dedupe_web_results(web_results),
                strategy,
                tool_calls,
            )

        strategy = "agent_requested_web_search"
        for tool_call in ai_message.tool_calls:
            if tool_call.get("name") != "web_search":
                continue
            payload = _invoke_web_search_tool(
                tool_call.get("args", {}).get("query", user_query)
            )
            web_results.extend(payload.get("results", []))
            tool_calls.append(
                {
                    "tool": "web_search",
                    "query": payload.get("query", user_query),
                    "mode": "agent_requested",
                }
            )
            messages.append(
                ToolMessage(
                    content=json.dumps(payload, ensure_ascii=True),
                    tool_call_id=tool_call["id"],
                )
            )

    final_message = _get_llm().invoke(messages)
    return (
        final_message.content.strip(),
        _dedupe_web_results(web_results),
        strategy,
        tool_calls,
    )


def get_pipeline_status() -> dict[str, Any]:
    local_files = [_relative_path(path) for path in _list_local_files()]
    using_demo_docs = not local_files
    return {
        "data_directory": _relative_path(DATA_DIR),
        "data_files": local_files,
        "fallback_demo_docs": using_demo_docs,
        "web_search_tool": "ddgs",
        "model": os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
    }


def ask_question(user_query: str) -> dict[str, Any]:
    """Return answer and supporting metadata for the HTTP API and Streamlit UI."""

    query = user_query.strip()
    if not query:
        raise ValueError("Query cannot be empty.")

    key = _cache_key(query)
    if key in _cache:
        cached = dict(_cache[key])
        cached["cache_hit"] = True
        return cached

    local_hits = _search_local_documents(query)
    answer, web_results, strategy, tool_calls = _run_agent(query, local_hits)
    knowledge_base = _get_knowledge_base()

    output = {
        "answer": answer,
        "docs": [hit.content for hit in local_hits],
        "local_documents": [asdict(hit) for hit in local_hits],
        "web_results": web_results,
        "source": _resolve_source_label(local_hits, web_results),
        "tool_strategy": strategy,
        "tool_used": bool(tool_calls),
        "tool_calls": tool_calls,
        "cache_hit": False,
        "knowledge_base": {
            "data_directory": _relative_path(DATA_DIR),
            "loaded_files": knowledge_base.loaded_files,
            "fallback_demo_docs": knowledge_base.fallback_demo_docs,
        },
    }
    _cache[key] = output
    return dict(output)
