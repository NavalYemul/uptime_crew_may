"""
RAGAS-proxy evaluation metrics and production monitoring for RAG systems.

Provides lightweight, dependency-free proxies for:
- faithfulness
- answer relevancy
- context recall

Also includes RAGMonitor for drift detection and CostTracker for API cost tracking.
"""

import math
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# RAGAS metric proxies
# ---------------------------------------------------------------------------

@dataclass
class RAGASResult:
    """Results from RAGAS-style evaluation."""
    faithfulness: float        # 0-1: is the answer grounded in the retrieved docs?
    answer_relevancy: float    # 0-1: does the answer address the question?
    context_recall: float      # 0-1: are relevant docs retrieved?
    overall: float             # weighted average


def compute_faithfulness(answer: str, contexts: list[str]) -> float:
    """
    Proxy for RAGAS faithfulness: what fraction of answer sentences
    are supported by at least one context?

    Production: use an LLM to check each claim in the answer against contexts.
    Proxy: TF-IDF cosine similarity between answer and concatenated contexts.
    """
    if not answer or not contexts:
        return 0.0

    context_text = " ".join(contexts).lower()
    answer_lower = answer.lower()

    # Get unique words
    context_words = set(context_text.split())
    answer_words  = set(answer_lower.split())
    stop_words    = {"the", "a", "an", "is", "in", "of", "and", "to", "it", "for"}

    answer_content  = answer_words  - stop_words
    context_content = context_words - stop_words

    if not answer_content:
        return 0.0

    overlap = answer_content & context_content
    return round(len(overlap) / len(answer_content), 3)


def compute_answer_relevancy(question: str, answer: str) -> float:
    """
    Proxy for RAGAS answer relevancy: does the answer address the question?

    Production: generate N questions from the answer, compute similarity to original.
    Proxy: word overlap between question and answer (after stripping punctuation).
    """
    if not question or not answer:
        return 0.0

    import re
    def tokenize(text: str) -> set[str]:
        # Strip punctuation and lowercase
        return set(re.sub(r"[^\w\s]", "", text.lower()).split())

    q_words = tokenize(question)
    a_words = tokenize(answer)
    stop    = {"what", "how", "why", "when", "where", "is", "are", "the", "a", "an"}

    q_content = q_words - stop
    a_content = a_words - stop

    if not q_content:
        return 0.0

    overlap = q_content & a_content
    return round(len(overlap) / len(q_content), 3)


def compute_context_recall(
    question: str,
    contexts: list[str],
    ground_truth: Optional[str] = None
) -> float:
    """
    Proxy for RAGAS context recall: are the retrieved contexts relevant?
    If no ground truth: use question-context overlap.
    """
    if not contexts:
        return 0.0

    reference = ground_truth or question
    ref_words = set(reference.lower().split())

    max_overlap = 0.0
    for ctx in contexts:
        ctx_words = set(ctx.lower().split())
        if not ctx_words:
            continue
        overlap = len(ref_words & ctx_words) / len(ref_words | ctx_words)
        max_overlap = max(max_overlap, overlap)

    return round(max_overlap, 3)


def evaluate_rag_response(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: Optional[str] = None,
    weights: tuple[float, float, float] = (0.4, 0.4, 0.2)
) -> RAGASResult:
    """Full RAGAS proxy evaluation."""
    f = compute_faithfulness(answer, contexts)
    r = compute_answer_relevancy(question, answer)
    c = compute_context_recall(question, contexts, ground_truth)
    overall = round(weights[0]*f + weights[1]*r + weights[2]*c, 3)
    return RAGASResult(
        faithfulness=f,
        answer_relevancy=r,
        context_recall=c,
        overall=overall
    )


# ---------------------------------------------------------------------------
# Production monitoring
# ---------------------------------------------------------------------------

@dataclass
class MonitoringMetrics:
    """Tracks LLM usage and quality over time."""
    total_queries: int = 0
    total_tokens: int = 0
    avg_faithfulness: float = 0.0
    avg_latency_ms: float = 0.0
    error_rate: float = 0.0
    queries_below_threshold: int = 0  # faithfulness < threshold


class RAGMonitor:
    """
    Tracks production metrics for drift detection and alerting.
    In production: write to Prometheus/Grafana or LangSmith datasets.
    """

    def __init__(self, faithfulness_threshold: float = 0.5):
        self.threshold = faithfulness_threshold
        self._records: list[dict] = []

    def record(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        tokens: int = 0,
        latency_ms: float = 0.0
    ) -> RAGASResult:
        result = evaluate_rag_response(query, answer, contexts)
        self._records.append({
            "query": query,
            "answer": answer,
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "tokens": tokens,
            "latency_ms": latency_ms
        })
        return result

    def get_metrics(self) -> MonitoringMetrics:
        if not self._records:
            return MonitoringMetrics()
        n = len(self._records)
        return MonitoringMetrics(
            total_queries=n,
            total_tokens=sum(r["tokens"] for r in self._records),
            avg_faithfulness=round(sum(r["faithfulness"] for r in self._records) / n, 3),
            avg_latency_ms=round(sum(r["latency_ms"] for r in self._records) / n, 1),
            error_rate=0.0,
            queries_below_threshold=sum(
                1 for r in self._records if r["faithfulness"] < self.threshold
            )
        )

    def detect_drift(self, window: int = 10) -> dict:
        """
        Compare recent vs historical faithfulness to detect degradation.
        Returns {"drifted": bool, "delta": float, "action": str}
        """
        if len(self._records) < window * 2:
            return {"drifted": False, "delta": 0.0, "action": "insufficient data"}

        recent   = self._records[-window:]
        historic = self._records[:-window]

        recent_avg   = sum(r["faithfulness"] for r in recent)   / len(recent)
        historic_avg = sum(r["faithfulness"] for r in historic) / len(historic)
        delta = historic_avg - recent_avg

        return {
            "drifted": delta > 0.1,
            "delta": round(delta, 3),
            "action": "retune retrieval" if delta > 0.1 else "no action needed"
        }


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

class CostTracker:
    """Tracks API costs across sessions."""

    PRICE_PER_1K = {
        "gpt-4o-mini":    {"input": 0.000150, "output": 0.000600},
        "gpt-4o":         {"input": 0.005000, "output": 0.015000},
        "claude-3-haiku": {"input": 0.000250, "output": 0.001250},
    }

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._sessions: list[dict] = []

    def record(self, input_tokens: int, output_tokens: int, query: str = "") -> float:
        prices = self.PRICE_PER_1K.get(self.model, self.PRICE_PER_1K["gpt-4o-mini"])
        cost = (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1000
        self._sessions.append({
            "query": query, "input": input_tokens, "output": output_tokens, "cost": cost
        })
        return round(cost, 6)

    def total_cost(self) -> float:
        return round(sum(s["cost"] for s in self._sessions), 6)

    def summary(self) -> dict:
        if not self._sessions:
            return {"sessions": 0, "total_cost_usd": 0.0, "avg_cost_usd": 0.0}
        n = len(self._sessions)
        total = self.total_cost()
        return {
            "sessions": n,
            "total_cost_usd": total,
            "avg_cost_usd": round(total / n, 6),
            "model": self.model
        }
