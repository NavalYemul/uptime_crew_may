"""Local-first RAG pipeline: hybrid retrieval, multi-tool agent, LangSmith tracing, RAGAS eval."""

from __future__ import annotations

import ast
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

# ── LangSmith tracing (Day 2/3/5 concept) ────────────────────────────────────
# Auto-enabled when LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY is set.
# Falls back to a no-op decorator when langsmith is not installed.
try:
    from langsmith import traceable
except ImportError:
    def traceable(name: str = "", **_kwargs):  # type: ignore[misc]
        def decorator(fn):
            return fn
        return decorator

# ── Optional tool dependencies ────────────────────────────────────────────────
try:
    from youtube_search import YoutubeSearch
    _YOUTUBE_AVAILABLE = True
except ImportError:
    _YOUTUBE_AVAILABLE = False

try:
    import wikipedia as _wiki
    _WIKIPEDIA_AVAILABLE = True
except ImportError:
    _WIKIPEDIA_AVAILABLE = False

# ── .env loading ──────────────────────────────────────────────────────────────
for _root in [Path.cwd(), *Path.cwd().resolve().parents]:
    _env = _root / ".env"
    if _env.is_file():
        load_dotenv(_env)
        break

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", "data"))
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
LOCAL_TOP_K = 4
MAX_TOOL_STEPS = 5          # increased — more tools may need multiple steps
SEMANTIC_CACHE_THRESHOLD = 0.92

TIME_SENSITIVE_TERMS = {
    "breaking", "current", "currently", "latest", "live",
    "news", "recent", "recently", "today", "update", "updated", "yesterday",
}

# Cost per 1K tokens (gpt-4o-mini pricing — Day 5 cost-tracking concept)
_MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.000150, "output": 0.000600},
    "gpt-4o":      {"input": 0.005000, "output": 0.015000},
    "gpt-4-turbo": {"input": 0.010000, "output": 0.030000},
}

# Safe math operators for the calculate tool
_SAFE_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
    ast.USub: lambda a: -a,
    ast.UAdd: lambda a: +a,
}

DEMO_DOCUMENTS = [
    (
        "demo_overview",
        "This application is a local-first retrieval demo. It searches local documents first "
        "and only uses a web-search tool when the answer is missing, incomplete, or needs "
        "current information.",
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
    (
        "langsmith_note",
        "LangSmith provides tracing, evaluation, and monitoring for LLM applications. "
        "The @traceable decorator creates named spans visible in the LangSmith dashboard.",
    ),
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_RRF_CAP = (1 / 60) + (1 / 60)


# ── Dataclasses ───────────────────────────────────────────────────────────────

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


# ── Module-level state ────────────────────────────────────────────────────────

# Semantic cache: list of {key, embedding, result} — Day 3 SemanticCache concept
_semantic_cache: list[dict[str, Any]] = []
_knowledge_base: KnowledgeBase | None = None
_llm: ChatOpenAI | None = None
_embeddings: OpenAIEmbeddings | None = None

# Cumulative metrics — Day 5 cost-tracking concept
_metrics: dict[str, Any] = {
    "total_queries": 0,
    "cache_hits": 0,
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "total_cost_usd": 0.0,
    "tool_call_counts": {},
}


# ── Helpers ───────────────────────────────────────────────────────────────────

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


# ── Chunking ──────────────────────────────────────────────────────────────────

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


# ── Knowledge base ────────────────────────────────────────────────────────────

def _list_local_files() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    allowed = {".pdf", ".txt", ".md"}
    return [
        p for p in sorted(DATA_DIR.rglob("*"))
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in allowed
    ]


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise RuntimeError("PDF support requires pypdf.") from exc
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_local_file(path: Path) -> str:
    return _read_pdf(path) if path.suffix.lower() == ".pdf" else \
        path.read_text(encoding="utf-8", errors="ignore")


def _load_local_chunks() -> tuple[list[DocumentChunk], list[str], bool]:
    chunks, loaded = [], []
    for path in _list_local_files():
        file_chunks = _chunk_text(_read_local_file(path), _relative_path(path), path.suffix.lower().lstrip("."))
        if file_chunks:
            chunks.extend(file_chunks)
            loaded.append(_relative_path(path))
    if chunks:
        return chunks, loaded, False
    demo = []
    for source, content in DEMO_DOCUMENTS:
        demo.extend(_chunk_text(content, source=source, kind="demo"))
    return demo, [], True


def _build_knowledge_base() -> KnowledgeBase:
    chunks, loaded_files, fallback = _load_local_chunks()
    tokenized = [_tokenize(c.content) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    matrix = np.asarray(
        _get_embeddings().embed_documents([c.content for c in chunks]),
        dtype=np.float32,
    )
    return KnowledgeBase(chunks=chunks, bm25=bm25, matrix=matrix,
                         loaded_files=loaded_files, fallback_demo_docs=fallback)


def _get_knowledge_base() -> KnowledgeBase:
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = _build_knowledge_base()
    return _knowledge_base


# ── Semantic cache (Day 3 concept) ────────────────────────────────────────────

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _semantic_cache_lookup(query: str) -> dict[str, Any] | None:
    """Return cached result if a semantically similar query was seen before."""
    if not _semantic_cache:
        return None
    q_vec = np.asarray(_get_embeddings().embed_query(query), dtype=np.float32)
    best_sim, best_result = 0.0, None
    for entry in _semantic_cache:
        sim = _cosine_similarity(q_vec, entry["embedding"])
        if sim > best_sim:
            best_sim, best_result = sim, entry["result"]
    if best_sim >= SEMANTIC_CACHE_THRESHOLD and best_result is not None:
        return best_result
    return None


def _semantic_cache_store(query: str, result: dict[str, Any]) -> None:
    """Store query embedding + result in semantic cache (max 128 entries, LRU)."""
    q_vec = np.asarray(_get_embeddings().embed_query(query), dtype=np.float32)
    _semantic_cache.append({"key": query, "embedding": q_vec, "result": result})
    if len(_semantic_cache) > 128:
        _semantic_cache.pop(0)


# ── Query rewriting (Day 3/5 concept) ─────────────────────────────────────────

def _rewrite_query(query: str) -> str:
    """
    Expand short or ambiguous queries to improve BM25 + vector retrieval recall.
    Strategies:
    - Very short query (≤3 words): append 'overview explanation'
    - Acronym-heavy query: add a clarifying suffix
    - Time-sensitive: leave unchanged (web search will handle it)
    """
    words = query.strip().split()
    if _query_is_time_sensitive(query):
        return query  # don't rewrite — let web search handle freshness
    if len(words) <= 3:
        return query + " overview explanation"
    return query


# ── Hybrid retrieval (BM25 + vector + RRF) ───────────────────────────────────

def _rrf_fusion(bm25_rank: np.ndarray, vector_rank: np.ndarray, k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion — RRF(d) = Σ 1/(k + rank_i). k=60 is standard."""
    scores: dict[int, float] = {}
    for rank, idx in enumerate(bm25_rank):
        scores[int(idx)] = scores.get(int(idx), 0.0) + (1 / (k + rank))
    for rank, idx in enumerate(vector_rank):
        scores[int(idx)] = scores.get(int(idx), 0.0) + (1 / (k + rank))
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


@traceable(name="hybrid-retrieval")
def _search_local_documents(query: str, top_k: int = LOCAL_TOP_K) -> list[SearchHit]:
    """Hybrid BM25 + semantic retrieval with RRF fusion. LangSmith span: hybrid-retrieval."""
    kb = _get_knowledge_base()
    tokens = _tokenize(query)
    if not tokens:
        return []

    bm25_scores = kb.bm25.get_scores(tokens)
    bm25_rank = np.argsort(bm25_scores)[::-1][: max(top_k * 3, top_k)]

    q_vec = np.asarray(_get_embeddings().embed_query(query), dtype=np.float32)
    distances = np.linalg.norm(kb.matrix - q_vec, axis=1)
    vector_rank = np.argsort(distances)[: max(top_k * 3, top_k)]

    fused = _rrf_fusion(bm25_rank, vector_rank)
    query_terms = set(tokens)
    hits: list[SearchHit] = []

    for idx, score in fused[:top_k]:
        chunk = kb.chunks[idx]
        overlap = len(query_terms & set(_tokenize(chunk.content))) / max(1, len(query_terms))
        normalized = min(score / _RRF_CAP, 1.0)
        confidence = round((normalized * 0.55) + (overlap * 0.45), 3)
        hits.append(SearchHit(content=chunk.content, source=chunk.source,
                               kind=chunk.kind, confidence=confidence))
    return hits


# ── Query helpers ─────────────────────────────────────────────────────────────

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
    return "\n\n".join(
        f"[Local document {i}] source={h.source} confidence={h.confidence}\n{h.content}"
        for i, h in enumerate(hits, 1)
    )


def _dedupe_web_results(results: list[dict[str, str]]) -> list[dict[str, str]]:
    seen, out = set(), []
    for item in results:
        key = item.get("url") or item.get("title", "")
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _resolve_source_label(local_hits: list[SearchHit], web_results: list[dict]) -> str:
    if local_hits and web_results:
        return "local+web_search"
    if web_results:
        return "web_search"
    return "local_knowledge_base"


# ── Tools ─────────────────────────────────────────────────────────────────────

def _web_search_payload(query: str) -> dict[str, Any]:
    try:
        from ddgs import DDGS
    except ModuleNotFoundError as exc:
        return {"query": query, "results": [], "error": "ddgs not installed.", "details": str(exc)}
    try:
        raw = DDGS(timeout=10).text(query, region="us-en", safesearch="moderate", max_results=5)
    except Exception as exc:  # noqa: BLE001
        return {"query": query, "results": [], "error": "Web search failed.", "details": str(exc)}
    results = [
        {"title": r.get("title", "Untitled"), "url": r.get("href") or r.get("url", ""), "snippet": r.get("body", "")}
        for r in (raw or []) if r.get("href") or r.get("url")
    ]
    return {"query": query, "results": _dedupe_web_results(results)}


@tool
def web_search(query: str) -> str:
    """Search the public web when the local knowledge base is missing the answer or needs current information."""
    return json.dumps(_web_search_payload(query), ensure_ascii=True)


@tool
def youtube_search_tool(query: str) -> str:
    """Search YouTube for videos about a topic. Use when the user asks for video tutorials, demos, or talks."""
    if not _YOUTUBE_AVAILABLE:
        return json.dumps({"query": query, "results": [], "error": "youtube_search package not installed."})
    try:
        results_raw = YoutubeSearch(query, max_results=3).to_dict()
        results = [
            {
                "title": v.get("title", ""),
                "url": f"https://youtube.com{v.get('url_suffix', '')}",
                "views": v.get("views", ""),
                "duration": v.get("duration", ""),
            }
            for v in results_raw
        ]
        return json.dumps({"query": query, "results": results}, ensure_ascii=True)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"query": query, "results": [], "error": str(exc)})


@tool
def wikipedia_search(query: str) -> str:
    """Look up a topic on Wikipedia for factual background information and definitions."""
    if not _WIKIPEDIA_AVAILABLE:
        return json.dumps({"query": query, "summary": "", "error": "wikipedia package not installed."})
    try:
        _wiki.set_lang("en")
        page = _wiki.page(query, auto_suggest=True)
        summary = page.summary[:600] + ("..." if len(page.summary) > 600 else "")
        return json.dumps({"query": query, "title": page.title, "summary": summary, "url": page.url}, ensure_ascii=True)
    except _wiki.exceptions.DisambiguationError as exc:
        # Pick the first option
        try:
            page = _wiki.page(exc.options[0])
            summary = page.summary[:600]
            return json.dumps({"query": query, "title": page.title, "summary": summary, "url": page.url})
        except Exception:
            return json.dumps({"query": query, "summary": "", "error": f"Disambiguation: {exc.options[:3]}"})
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"query": query, "summary": "", "error": str(exc)})


@tool
def calculate(expression: str) -> str:
    """
    Evaluate a safe mathematical expression and return the result.
    Supports: +, -, *, /, ** (power). Examples: '15 * 47', '2 ** 10', '(100 + 50) / 3'.
    """
    def _eval(node: ast.AST):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            op_fn = _SAFE_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op_fn(_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op_fn = _SAFE_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op_fn(_eval(node.operand))
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")

    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval(tree.body)
        return json.dumps({"expression": expression, "result": result})
    except Exception as exc:
        return json.dumps({"expression": expression, "result": None, "error": str(exc)})


_ALL_TOOLS = [web_search, youtube_search_tool, wikipedia_search, calculate]
_TOOL_MAP = {t.name: t for t in _ALL_TOOLS}


def _dispatch_tool(tool_name: str, args: dict, query_fallback: str) -> dict[str, Any]:
    """Invoke any registered tool by name, return parsed JSON payload."""
    t = _TOOL_MAP.get(tool_name)
    if t is None:
        return {"error": f"Unknown tool: {tool_name}"}
    try:
        raw = t.invoke(args)
        return json.loads(raw) if isinstance(raw, str) else raw
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


# ── Cost tracking (Day 5 concept) ─────────────────────────────────────────────

def _compute_cost(input_tokens: int, output_tokens: int) -> float:
    """Compute USD cost for a single LLM call using gpt-4o-mini pricing."""
    model = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    prices = _MODEL_PRICING.get(model, _MODEL_PRICING["gpt-4o-mini"])
    return round((input_tokens * prices["input"] + output_tokens * prices["output"]) / 1000, 8)


def _extract_token_usage(ai_message) -> tuple[int, int]:
    """Extract (input_tokens, output_tokens) from an AIMessage safely."""
    try:
        meta = ai_message.usage_metadata or {}
        return meta.get("input_tokens", 0), meta.get("output_tokens", 0)
    except AttributeError:
        try:
            meta = ai_message.response_metadata.get("token_usage", {})
            return meta.get("prompt_tokens", 0), meta.get("completion_tokens", 0)
        except (AttributeError, KeyError):
            return 0, 0


# ── RAGAS-proxy evaluation (Day 2/3/5 concept) ────────────────────────────────

def _compute_faithfulness(answer: str, contexts: list[str]) -> float:
    """
    Proxy faithfulness: what fraction of answer content words appear in context?
    Production RAGAS uses an LLM judge per claim; this is a fast word-overlap proxy.
    """
    if not answer or not contexts:
        return 0.0
    stop = {"the", "a", "an", "is", "in", "of", "and", "to", "it", "for", "was", "are", "with", "on"}
    ctx_words = set(_tokenize(" ".join(contexts))) - stop
    ans_words = set(_tokenize(answer)) - stop
    if not ans_words:
        return 0.0
    return round(len(ans_words & ctx_words) / len(ans_words), 3)


def _compute_answer_relevancy(question: str, answer: str) -> float:
    """
    Proxy answer relevancy: do key question words appear in the answer?
    Production RAGAS generates N questions from the answer and measures similarity.
    """
    if not question or not answer:
        return 0.0
    stop = {"what", "how", "why", "when", "where", "is", "are", "the", "a", "an", "do", "does"}
    q_words = set(_tokenize(question)) - stop
    a_words = set(_tokenize(answer)) - stop
    if not q_words:
        return 0.0
    return round(len(q_words & a_words) / len(q_words), 3)


# ── Agent loop ────────────────────────────────────────────────────────────────

@traceable(name="agent-loop")
def _run_agent(
    user_query: str,
    local_hits: list[SearchHit],
) -> tuple[str, list[dict], str, list[dict], int, int]:
    """
    Multi-tool ReAct agent loop. LangSmith span: agent-loop.
    Returns (answer, web_results, strategy, tool_calls, input_tokens, output_tokens).
    """
    local_context = _format_local_context(local_hits)
    force_web = _should_force_web_search(user_query, local_hits)

    messages = [
        SystemMessage(content=(
            "You are a local-first retrieval assistant. You have four tools:\n"
            "  • web_search        — search the public web for current information\n"
            "  • youtube_search_tool — find YouTube videos about a topic\n"
            "  • wikipedia_search  — look up factual definitions and background\n"
            "  • calculate         — evaluate safe math expressions\n\n"
            "Use local context whenever it answers the question. Call tools only when needed. "
            "Never invent facts the local corpus does not contain."
        )),
        HumanMessage(content=(
            f"User question:\n{user_query}\n\n"
            f"Local context:\n{local_context}\n\n"
            f"Local context appears sufficient: {'yes' if not force_web else 'no'}.\n"
            "Answer from local context if it is enough. Otherwise use the appropriate tool."
        )),
    ]

    llm_with_tools = _get_llm().bind_tools(_ALL_TOOLS)
    web_results: list[dict] = []
    tool_calls_log: list[dict] = []
    strategy = "local_only"
    total_in, total_out = 0, 0

    for _ in range(MAX_TOOL_STEPS):
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        i_tok, o_tok = _extract_token_usage(ai_msg)
        total_in += i_tok
        total_out += o_tok

        if not ai_msg.tool_calls:
            if force_web and not web_results:
                # Forced web fallback: LLM chose not to call web but we override
                strategy = "forced_web_fallback"
                payload = _dispatch_tool("web_search", {"query": user_query}, user_query)
                web_results.extend(payload.get("results", []))
                tool_calls_log.append({"tool": "web_search", "query": user_query, "mode": "forced_fallback"})
                messages.append(HumanMessage(content=(
                    "The app executed web_search because local retrieval was insufficient.\n\n"
                    f"{json.dumps(payload, ensure_ascii=True)}"
                )))
                continue
            return ai_msg.content.strip(), _dedupe_web_results(web_results), strategy, tool_calls_log, total_in, total_out

        strategy = "agent_requested_tool"
        for tc in ai_msg.tool_calls:
            tool_name = tc.get("name", "")
            tool_args = tc.get("args", {})
            payload = _dispatch_tool(tool_name, tool_args, user_query)

            # Collect web-style results for evidence panel
            if tool_name == "web_search":
                web_results.extend(payload.get("results", []))
            elif tool_name == "youtube_search_tool":
                for v in payload.get("results", []):
                    web_results.append({"title": v["title"], "url": v["url"], "snippet": v.get("duration", "")})
            elif tool_name == "wikipedia_search":
                if payload.get("url"):
                    web_results.append({"title": payload.get("title", "Wikipedia"), "url": payload["url"], "snippet": payload.get("summary", "")[:200]})

            tool_calls_log.append({"tool": tool_name, "query": str(tool_args), "mode": "agent_requested"})
            _metrics["tool_call_counts"][tool_name] = _metrics["tool_call_counts"].get(tool_name, 0) + 1

            messages.append(ToolMessage(content=json.dumps(payload, ensure_ascii=True), tool_call_id=tc["id"]))

    # Exhausted steps — get final answer
    final = _get_llm().invoke(messages)
    f_in, f_out = _extract_token_usage(final)
    return final.content.strip(), _dedupe_web_results(web_results), strategy, tool_calls_log, total_in + f_in, total_out + f_out


# ── Public API ────────────────────────────────────────────────────────────────

def get_pipeline_status() -> dict[str, Any]:
    local_files = [_relative_path(p) for p in _list_local_files()]
    return {
        "data_directory": _relative_path(DATA_DIR),
        "data_files": local_files,
        "fallback_demo_docs": not local_files,
        "tools": [t.name for t in _ALL_TOOLS],
        "langsmith_tracing": os.environ.get("LANGCHAIN_TRACING_V2", "false"),
        "langsmith_project": os.environ.get("LANGCHAIN_PROJECT", ""),
        "model": os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        "semantic_cache_entries": len(_semantic_cache),
    }


def get_metrics() -> dict[str, Any]:
    """Return cumulative usage metrics for the /metrics endpoint."""
    return dict(_metrics)


@traceable(name="rag-ask")
def ask_question(user_query: str) -> dict[str, Any]:
    """
    Top-level entry point. LangSmith span: rag-ask.
    Returns answer + metadata including eval scores, token usage, and cost.
    """
    query = user_query.strip()
    if not query:
        raise ValueError("Query cannot be empty.")

    _metrics["total_queries"] += 1

    # ── Semantic cache lookup (Day 3 SemanticCache concept) ──────────────────
    cached = _semantic_cache_lookup(query)
    if cached:
        _metrics["cache_hits"] += 1
        result = dict(cached)
        result["cache_hit"] = True
        return result

    # ── Query rewriting (Day 3/5 concept) ────────────────────────────────────
    rewritten = _rewrite_query(query)

    # ── Hybrid retrieval ──────────────────────────────────────────────────────
    local_hits = _search_local_documents(rewritten)

    # ── Agent loop ────────────────────────────────────────────────────────────
    answer, web_results, strategy, tool_calls, in_tok, out_tok = _run_agent(query, local_hits)

    # ── Cost tracking (Day 5 concept) ────────────────────────────────────────
    cost = _compute_cost(in_tok, out_tok)
    _metrics["total_input_tokens"] += in_tok
    _metrics["total_output_tokens"] += out_tok
    _metrics["total_cost_usd"] = round(_metrics["total_cost_usd"] + cost, 8)

    # ── RAGAS-proxy evaluation (Day 2/3/5 concept) ───────────────────────────
    contexts = [h.content for h in local_hits]
    faithfulness = _compute_faithfulness(answer, contexts)
    answer_relevancy = _compute_answer_relevancy(query, answer)

    kb = _get_knowledge_base()
    output: dict[str, Any] = {
        "answer": answer,
        "docs": [h.content for h in local_hits],
        "local_documents": [asdict(h) for h in local_hits],
        "web_results": web_results,
        "source": _resolve_source_label(local_hits, web_results),
        "tool_strategy": strategy,
        "tool_used": bool(tool_calls),
        "tool_calls": tool_calls,
        "cache_hit": False,
        "rewritten_query": rewritten if rewritten != query else None,
        "eval": {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
        },
        "usage": {
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cost_usd": cost,
        },
        "knowledge_base": {
            "data_directory": _relative_path(DATA_DIR),
            "loaded_files": kb.loaded_files,
            "fallback_demo_docs": kb.fallback_demo_docs,
        },
    }

    _semantic_cache_store(query, output)
    return dict(output)
