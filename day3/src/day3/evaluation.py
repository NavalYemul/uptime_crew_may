"""
evaluation.py — RAGAS Evaluation, Golden Dataset & A/B Testing
===============================================================
Covers: RAGAS metrics, golden Q/A pairs, A/B tests between RAG variants,
        faithfulness targeting, evaluation-driven iteration.

Evaluation-driven development:
  1. Define golden dataset of question + expected answer pairs.
  2. Run your RAG pipeline on ALL questions.
  3. Score each answer on faithfulness, relevancy, precision, recall.
  4. Establish a BASELINE.
  5. Change ONE thing (chunking, retrieval, model).
  6. Re-evaluate and compare against baseline.
  7. Keep changes that improve the scores. Revert the rest.

This is the engineering discipline that separates production RAG from demos.

Run: python -m day3.evaluation
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


# ══════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════

@dataclass
class QAPair:
    """A golden question-answer pair for evaluation.

    Fields:
        question:     The evaluation question.
        ground_truth: The ideal answer (written by a domain expert).
        category:     Product category for stratified analysis.
    """
    question:     str
    ground_truth: str
    category:     str


@dataclass
class EvalResult:
    """RAGAS evaluation scores for a single Q&A pair.

    Fields:
        question:          The evaluated question.
        answer:            The RAG pipeline's generated answer.
        ground_truth:      The ideal expected answer.
        faithfulness:      0–1. Is the answer supported by the retrieved context?
                           <0.85 = answer has hallucinated content.
        answer_relevancy:  0–1. Does the answer actually address the question?
                           <0.70 = answer is off-topic or too vague.
        context_precision: 0–1. Signal-to-noise ratio of retrieved documents.
                           <0.60 = retrieved chunks are mostly irrelevant.
        context_recall:    0–1. Did retrieval find all needed information?
                           <0.70 = answers missing key facts from ground truth.
    """
    question:          str
    answer:            str
    ground_truth:      str
    faithfulness:      float
    answer_relevancy:  float
    context_precision: float
    context_recall:    float

    @property
    def passed(self) -> bool:
        """True if this result meets production quality thresholds.

        Thresholds:
          faithfulness >= 0.85  (answer is grounded in context)
          answer_relevancy >= 0.7  (answer addresses the question)
        """
        return self.faithfulness >= 0.85 and self.answer_relevancy >= 0.7

    @property
    def average(self) -> float:
        """Mean of all four RAGAS metrics."""
        return round(
            (self.faithfulness + self.answer_relevancy +
             self.context_precision + self.context_recall) / 4, 4
        )


# ══════════════════════════════════════════════════════
# GOLDEN DATASET
# ══════════════════════════════════════════════════════

GOLDEN_DATASET: list[QAPair] = [
    QAPair(
        "What is the best laptop for machine learning?",
        "MacBook Pro 14 M3 Pro (₹1,99,900) with M3 Pro chip, 18GB unified memory, "
        "and 18-hour battery is best for ML and data science.",
        "laptop",
    ),
    QAPair(
        "Which smartphone has the best camera system?",
        "Samsung Galaxy S24 Ultra has a 200MP main camera with quad camera system "
        "including 10MP telephoto at 3x and 50MP at 5x zoom.",
        "smartphone",
    ),
    QAPair(
        "What headphones have the best noise cancellation?",
        "Sony WH-1000XM5 has industry-leading ANC with 8 microphones and dual "
        "processors, offering 30-hour battery life.",
        "headphones",
    ),
    QAPair(
        "Which laptop is best for travelling executives?",
        "Lenovo ThinkPad X1 Carbon Gen 12 at 1.12kg with MIL-SPEC durability "
        "and 57Wh battery is best for travelling executives.",
        "laptop",
    ),
    QAPair(
        "What is the price of Apple AirPods Pro?",
        "Apple AirPods Pro 2nd generation is priced at ₹24,900.",
        "headphones",
    ),
    QAPair(
        "Which tablet supports a stylus pen?",
        "Both iPad Pro 12.9-inch M2 (with Apple Pencil 2nd gen) and Samsung "
        "Galaxy Tab S9 Ultra (S Pen included) support stylus input.",
        "tablet",
    ),
    QAPair(
        "What is the best gaming laptop under 1.5 lakh?",
        "ASUS ROG Zephyrus G14 (₹1,39,990) with AMD Ryzen 9 8945HS and RX 7700S "
        "with 144Hz display is best gaming laptop under ₹1.5 lakh.",
        "laptop",
    ),
    QAPair(
        "Which phone supports wireless charging?",
        "OnePlus 12 supports 50W wireless charging alongside 100W SUPERVOOC "
        "wired charging.",
        "smartphone",
    ),
    QAPair(
        "What is the best mouse for productivity?",
        "Logitech MX Master 3S (₹9,995) with MagSpeed scroll wheel and Flow "
        "multi-device feature is best for productivity.",
        "accessories",
    ),
    QAPair(
        "Which power bank can charge a MacBook?",
        "Anker 737 PowerCore 24K (₹6,999) with 140W output can charge MacBook Pro "
        "from 0 to 50% in 45 minutes.",
        "accessories",
    ),
]


# ══════════════════════════════════════════════════════
# RAGAS PROXY METRICS
# ══════════════════════════════════════════════════════

def compute_ragas_proxy(
    question:     str,
    answer:       str,
    context:      str,
    ground_truth: str,
) -> EvalResult:
    """Compute RAGAS metrics using TF-IDF cosine and word overlap proxies.

    Real RAGAS uses an LLM judge (GPT-4) to evaluate each metric. These
    proxy implementations give the same conceptual signal at zero cost:

    Faithfulness proxy:
      TF-IDF cosine similarity between the answer and the context.
      High faithfulness = answer words closely match context words.
      If the answer introduces words/concepts not in context, score drops.

    Answer Relevancy proxy:
      Word overlap between the answer and the question.
      High relevancy = answer uses words from the question (on-topic).

    Context Precision proxy:
      Proxy for signal-to-noise ratio. Estimated as min(len(context)/500, 1.0).
      Longer context = more noise (retrieval fetched irrelevant documents).
      Short, focused context = high precision.

    Context Recall proxy:
      TF-IDF cosine similarity between context and ground truth.
      High recall = context contains the information in the ground truth answer.

    Production RAGAS (requires OPENAI_API_KEY):
      from ragas import evaluate
      from ragas.metrics import faithfulness, answer_relevancy
      dataset = Dataset.from_dict({...})
      evaluate(dataset, metrics=[faithfulness, answer_relevancy, ...])

    Args:
        question:     The evaluation question.
        answer:       The RAG pipeline's generated answer.
        context:      The retrieved context provided to the LLM.
        ground_truth: The ideal expected answer.

    Returns:
        EvalResult with all four metric scores.
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

    def tfidf_sim(a: str, b: str) -> float:
        """TF-IDF cosine similarity between two texts."""
        if not a.strip() or not b.strip():
            return 0.0
        try:
            vec = TfidfVectorizer(min_df=1).fit([a, b])
            m = vec.transform([a, b])
            return float(sk_cosine(m[0], m[1])[0, 0])
        except Exception:
            return 0.0

    def word_overlap(a: str, b: str) -> float:
        """Jaccard word overlap between two texts."""
        wa = set(a.lower().split())
        wb = set(b.lower().split())
        if not wb:
            return 0.0
        return len(wa & wb) / max(len(wb), 1)

    # Faithfulness: is the answer grounded in the context?
    faithfulness = tfidf_sim(answer, context)

    # Answer relevancy: does the answer address the question?
    answer_relevancy = word_overlap(answer, question)

    # Context precision: proxy via context length (longer = more noise)
    context_precision = min(1.0, len(context) / 500) if context else 0.0
    # Invert: shorter focused context = higher precision
    context_precision = round(1.0 - context_precision * 0.3, 4)

    # Context recall: does context contain ground truth information?
    context_recall = tfidf_sim(context, ground_truth)

    return EvalResult(
        question=question,
        answer=answer,
        ground_truth=ground_truth,
        faithfulness=round(faithfulness, 4),
        answer_relevancy=round(answer_relevancy, 4),
        context_precision=round(max(0.0, context_precision), 4),
        context_recall=round(context_recall, 4),
    )


# ══════════════════════════════════════════════════════
# RAG EVALUATOR
# ══════════════════════════════════════════════════════

class RAGEvaluator:
    """Evaluate a RAG pipeline against a golden Q&A dataset.

    Usage:
      evaluator = RAGEvaluator(pipeline)
      results   = evaluator.run(GOLDEN_DATASET)
      summary   = evaluator.summary(results)
      failing   = evaluator.failing_questions(results)
    """

    def __init__(self, pipeline):
        """
        Args:
            pipeline: An AdvancedRAGPipeline instance (or any object
                      with a .query(question) method returning an object
                      with .answer and .reranked_chunks attributes).
        """
        self._pipeline = pipeline

    def run(self, qa_pairs: list[QAPair]) -> list[EvalResult]:
        """Run all Q&A pairs through the pipeline and score results.

        Args:
            qa_pairs: List of golden Q&A pairs to evaluate.

        Returns:
            List of EvalResult, one per Q&A pair.
        """
        results = []
        for qa in qa_pairs:
            try:
                rag_result = self._pipeline.query(qa.question)
                context = " ".join(rag_result.reranked_chunks)
                answer  = rag_result.answer

                eval_result = compute_ragas_proxy(
                    question=qa.question,
                    answer=answer,
                    context=context,
                    ground_truth=qa.ground_truth,
                )
                results.append(eval_result)
            except Exception as e:
                # Don't let one bad query fail the whole evaluation
                results.append(EvalResult(
                    question=qa.question,
                    answer=f"ERROR: {e}",
                    ground_truth=qa.ground_truth,
                    faithfulness=0.0,
                    answer_relevancy=0.0,
                    context_precision=0.0,
                    context_recall=0.0,
                ))
        return results

    def summary(self, results: list[EvalResult]) -> dict:
        """Compute aggregate statistics across all evaluation results.

        Returns:
            Dict with mean/std for each metric, plus overall pass_rate.
            Keys: faithfulness, answer_relevancy, context_precision,
                  context_recall, pass_rate, num_evaluated.
        """
        import numpy as np

        if not results:
            return {"error": "No results to summarize"}

        metrics = {
            "faithfulness":      [r.faithfulness      for r in results],
            "answer_relevancy":  [r.answer_relevancy   for r in results],
            "context_precision": [r.context_precision  for r in results],
            "context_recall":    [r.context_recall     for r in results],
        }

        summary = {}
        for metric, values in metrics.items():
            summary[metric] = {
                "mean": round(float(np.mean(values)), 4),
                "std":  round(float(np.std(values)),  4),
                "min":  round(float(np.min(values)),  4),
                "max":  round(float(np.max(values)),  4),
            }

        passed = sum(1 for r in results if r.passed)
        summary["pass_rate"]     = round(passed / len(results), 4)
        summary["num_evaluated"] = len(results)
        summary["num_passed"]    = passed
        summary["num_failed"]    = len(results) - passed

        return summary

    def failing_questions(self, results: list[EvalResult]) -> list[EvalResult]:
        """Return results that failed quality thresholds.

        A result fails if faithfulness < 0.85 OR answer_relevancy < 0.70.

        Args:
            results: List of EvalResult from run().

        Returns:
            Subset of results where passed == False.
        """
        return [r for r in results if not r.passed]

    def save_baseline(
        self,
        results: list[EvalResult],
        path:    str = "baseline.json",
    ) -> dict:
        """Save evaluation results as a reproducible baseline.

        Run once, commit the JSON, then compare future pipeline variants
        against this baseline to measure improvement or regression.

        Args:
            results: List of EvalResult from run().
            path:    Output file path for the JSON baseline.

        Returns:
            The baseline dict that was saved.
        """
        from datetime import datetime
        import numpy as np

        summary = self.summary(results)
        baseline = {
            "timestamp":   datetime.now().isoformat(),
            "num_queries": len(results),
            "summary":     summary,
            "per_question": [
                {
                    "question":         r.question,
                    "faithfulness":     r.faithfulness,
                    "answer_relevancy": r.answer_relevancy,
                    "passed":           r.passed,
                }
                for r in results
            ],
        }

        Path(path).write_text(json.dumps(baseline, indent=2))
        print(f"[Evaluation] Baseline saved → {path}")
        return baseline


# ══════════════════════════════════════════════════════
# A/B TESTING
# ══════════════════════════════════════════════════════

def ab_test(
    pipeline_a:  Any,
    pipeline_b:  Any,
    qa_pairs:    list[QAPair],
    name_a:      str = "variant_a",
    name_b:      str = "variant_b",
) -> dict:
    """Run an A/B test between two RAG pipeline variants.

    Both pipelines receive the same questions. Scores are compared
    per metric and an overall winner is declared.

    Args:
        pipeline_a: First pipeline variant (any object with .query() method).
        pipeline_b: Second pipeline variant.
        qa_pairs:   Evaluation questions to test both on.
        name_a:     Display name for variant A.
        name_b:     Display name for variant B.

    Returns:
        Dict with per-metric winner, overall winner, and full comparison data.
    """
    evaluator_a = RAGEvaluator(pipeline_a)
    evaluator_b = RAGEvaluator(pipeline_b)

    t_a = time.perf_counter()
    results_a = evaluator_a.run(qa_pairs)
    latency_a = (time.perf_counter() - t_a) * 1000

    t_b = time.perf_counter()
    results_b = evaluator_b.run(qa_pairs)
    latency_b = (time.perf_counter() - t_b) * 1000

    summary_a = evaluator_a.summary(results_a)
    summary_b = evaluator_b.summary(results_b)

    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    winners = {}
    for metric in metrics:
        mean_a = summary_a[metric]["mean"]
        mean_b = summary_b[metric]["mean"]
        if mean_a > mean_b:
            winners[metric] = name_a
        elif mean_b > mean_a:
            winners[metric] = name_b
        else:
            winners[metric] = "tie"

    # Overall winner: most metric wins
    a_wins = sum(1 for w in winners.values() if w == name_a)
    b_wins = sum(1 for w in winners.values() if w == name_b)

    if a_wins > b_wins:
        overall_winner = name_a
    elif b_wins > a_wins:
        overall_winner = name_b
    else:
        # Tiebreak: faithfulness
        fa = summary_a["faithfulness"]["mean"]
        fb = summary_b["faithfulness"]["mean"]
        overall_winner = name_a if fa >= fb else name_b

    return {
        "overall_winner": overall_winner,
        "metric_winners": winners,
        name_a: {
            "summary":     summary_a,
            "latency_ms":  round(latency_a, 1),
            "pass_rate":   summary_a["pass_rate"],
        },
        name_b: {
            "summary":     summary_b,
            "latency_ms":  round(latency_b, 1),
            "pass_rate":   summary_b["pass_rate"],
        },
    }


# ══════════════════════════════════════════════════════
# CHUNKING ABLATION
# ══════════════════════════════════════════════════════

def chunking_ablation(
    text_corpus: list[str],
    qa_pairs:    list[QAPair],
) -> dict:
    """Test which chunking strategy produces the best RAG evaluation scores.

    Ablation study: hold everything constant EXCEPT chunking strategy.
    This isolates the effect of chunking on retrieval quality.

    Strategies tested:
      - fixed     : fixed-size character windows
      - recursive : paragraph → sentence → word hierarchy
      - character : single separator (paragraph breaks)

    Note: semantic chunking is excluded here (requires encoder model).

    Args:
        text_corpus: List of document texts to chunk and index.
        qa_pairs:    Golden Q&A pairs to evaluate each variant.

    Returns:
        Dict keyed by strategy name with summary metrics per strategy.
    """
    from day3.chunking import fixed_size_chunk, recursive_chunk, character_chunk
    from day3.rag_pipeline import AdvancedRAGPipeline, mock_claude_response

    def _demo_embed(texts):
        import numpy as np
        vecs = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**31))
            v = rng.standard_normal(384).astype(np.float32)
            v = v / np.linalg.norm(v)
            vecs.append(v)
        return np.array(vecs)

    strategies = {
        "fixed":     lambda text: fixed_size_chunk(text, chunk_size=400, overlap=80),
        "recursive": lambda text: recursive_chunk(text, chunk_size=400, overlap=80),
        "character": lambda text: character_chunk(text, separator="\n\n"),
    }

    ablation_results = {}

    for strategy_name, chunk_fn in strategies.items():
        # Chunk all documents with this strategy
        all_chunks = []
        for text in text_corpus:
            all_chunks.extend(chunk_fn(text))

        # Build pipeline
        pipeline = AdvancedRAGPipeline(
            use_hybrid=False,  # dense-only for ablation (isolate chunking effect)
            use_reranker=False,
            use_cache=False,
            top_k=3,
            rerank_top_k=3,
            mock_embed_fn=_demo_embed,
        )
        documents = [{"text": c.text, "metadata": {"strategy": strategy_name}} for c in all_chunks]
        pipeline.index(documents)

        # Evaluate
        evaluator = RAGEvaluator(pipeline)
        results = evaluator.run(qa_pairs)
        summary = evaluator.summary(results)

        ablation_results[strategy_name] = {
            "num_chunks":       len(all_chunks),
            "faithfulness":     summary["faithfulness"]["mean"],
            "answer_relevancy": summary["answer_relevancy"]["mean"],
            "context_recall":   summary["context_recall"]["mean"],
            "pass_rate":        summary["pass_rate"],
        }

    return ablation_results


# ══════════════════════════════════════════════════════
# MAIN DEMO
# ══════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("RAG EVALUATION: RAGAS METRICS, GOLDEN DATASET & A/B TEST")
    print("=" * 70)

    from day3.rag_pipeline import build_product_rag

    def _demo_embed(texts):
        import numpy as np
        vecs = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**31))
            v = rng.standard_normal(384).astype(np.float32)
            v = v / np.linalg.norm(v)
            vecs.append(v)
        return np.array(vecs)

    print(f"\n[1] Golden dataset: {len(GOLDEN_DATASET)} questions")
    for qa in GOLDEN_DATASET[:3]:
        print(f"  [{qa.category}] {qa.question[:70]}")
    print(f"  ... and {len(GOLDEN_DATASET) - 3} more")

    print("\n[2] Building RAG pipeline...")
    pipeline = build_product_rag(mock_embed_fn=_demo_embed)

    print("\n[3] Running evaluation on golden dataset...")
    evaluator = RAGEvaluator(pipeline)
    results = evaluator.run(GOLDEN_DATASET)

    print("\n[4] Evaluation Summary:")
    summary = evaluator.summary(results)
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        m = summary[metric]
        print(f"  {metric:<22}: mean={m['mean']:.4f} std={m['std']:.4f} "
              f"min={m['min']:.4f} max={m['max']:.4f}")
    print(f"\n  Pass rate: {summary['pass_rate']:.1%} "
          f"({summary['num_passed']}/{summary['num_evaluated']} questions)")

    print("\n[5] Failing questions:")
    failing = evaluator.failing_questions(results)
    if failing:
        for r in failing[:3]:
            print(f"  - '{r.question[:60]}' | faith={r.faithfulness:.3f} rel={r.answer_relevancy:.3f}")
    else:
        print("  All questions passed!")

    print("\n[6] Saving baseline...")
    evaluator.save_baseline(results, "day3_baseline.json")

    print("\n[7] A/B Test: hybrid vs dense-only")
    pipeline_b = build_product_rag(mock_embed_fn=_demo_embed)
    pipeline_b.use_hybrid = False  # variant B: dense-only

    ab_results = ab_test(pipeline, pipeline_b, GOLDEN_DATASET[:5],
                         name_a="hybrid_rrf", name_b="dense_only")
    print(f"  Overall winner: {ab_results['overall_winner']}")
    for metric, winner in ab_results["metric_winners"].items():
        print(f"  {metric:<22}: {winner}")


if __name__ == "__main__":
    main()
