"""
Day 5 — Enterprise RAG, Multi-Agent Orchestration & Deployment.

Key exports:
    multi_agent_orchestrator: build_orchestrator, run_orchestrator, build_supervisor_orchestrator
    hybrid_search:            BM25Index, DenseIndex, HybridSearcher, reciprocal_rank_fusion, rerank_mock, query_rewrite
    streaming:                StreamChunk, stream_tokens, stream_with_sources, collect_stream, StreamingOrchestrator
    evaluation:               evaluate_rag_response, RAGASResult, RAGMonitor, CostTracker
    deployment:               generate_docker_compose, DockerConfig, show_gitops_pattern, show_auth_pattern, get_deployment_checklist
"""

from day5.multi_agent_orchestrator import (
    build_orchestrator,
    run_orchestrator,
    build_supervisor_orchestrator,
    OrchestratorState,
    SupervisorState,
)

from day5.hybrid_search import (
    BM25Index,
    DenseIndex,
    HybridSearcher,
    reciprocal_rank_fusion,
    rerank_mock,
    query_rewrite,
)

from day5.streaming import (
    StreamChunk,
    stream_tokens,
    stream_with_sources,
    collect_stream,
    StreamingOrchestrator,
    fastapi_streaming_example,
    vercel_ai_sdk_example,
)

from day5.evaluation import (
    compute_faithfulness,
    compute_answer_relevancy,
    compute_context_recall,
    evaluate_rag_response,
    RAGASResult,
    RAGMonitor,
    CostTracker,
    MonitoringMetrics,
)

from day5.deployment import (
    DockerConfig,
    generate_docker_compose,
    show_gitops_pattern,
    show_auth_pattern,
    show_langsmith_tracing,
    get_deployment_checklist,
)

__all__ = [
    # orchestrator
    "build_orchestrator",
    "run_orchestrator",
    "build_supervisor_orchestrator",
    "OrchestratorState",
    "SupervisorState",
    # hybrid search
    "BM25Index",
    "DenseIndex",
    "HybridSearcher",
    "reciprocal_rank_fusion",
    "rerank_mock",
    "query_rewrite",
    # streaming
    "StreamChunk",
    "stream_tokens",
    "stream_with_sources",
    "collect_stream",
    "StreamingOrchestrator",
    "fastapi_streaming_example",
    "vercel_ai_sdk_example",
    # evaluation
    "compute_faithfulness",
    "compute_answer_relevancy",
    "compute_context_recall",
    "evaluate_rag_response",
    "RAGASResult",
    "RAGMonitor",
    "CostTracker",
    "MonitoringMetrics",
    # deployment
    "DockerConfig",
    "generate_docker_compose",
    "show_gitops_pattern",
    "show_auth_pattern",
    "show_langsmith_tracing",
    "get_deployment_checklist",
]
