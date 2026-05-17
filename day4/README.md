# Day 4 — Agents, LangGraph & MCP

## Setup (one venv for everything)

```bash
cd day4-agents-langgraph-mcp
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[all]"
```

This single venv covers both `src/day4/` modules and all `langgraph-tutorials/` notebooks.

## Run tests

```bash
pytest tests/ -v
```

## Project layout

```
day4-agents-langgraph-mcp/
├── src/day4/                  # Core modules
│   ├── langgraph_basics.py    # StateGraph, conditional routing
│   ├── tools_agents.py        # @tool, BaseTool, ReAct, guardrails
│   ├── memory_persistence.py  # MemorySaver, ShortTermMemory, LongTermMemory
│   ├── human_in_loop.py       # interrupt(), ApprovalWorkflow
│   ├── multi_agent.py         # Supervisor, subgraphs, agentic RAG
│   ├── mcp_integration.py     # MCP tools, producer patterns, tiktoken
│   └── prompt_engineering.py  # Zero-shot, few-shot, CoT, tiktoken
├── tests/                     # 110 passing tests
├── notebooks/                 # 6 Jupyter notebooks (01-06)
├── langgraph-tutorials/       # Tutorial notebooks (16 examples)
├── pyproject.toml             # Single dependency file for the whole project
└── DAY4_TOPICS.txt            # Topic reference
```

## Notebooks

| Notebook | Topics |
|----------|--------|
| 01_langgraph_basics | StateGraph, conditional routing, prompt engineering, tiktoken |
| 02_tools_agents | @tool, BaseTool, Pydantic, Instructor, guardrails, ReAct |
| 03_memory_persistence | ConversationBuffer, MemorySaver, thread isolation, time travel |
| 04_human_in_loop | interrupt(), Command(resume), structured action proposals |
| 05_multi_agent | Supervisor pattern, subgraphs, agentic RAG |
| 06_mcp_integration | JSON-RPC 2.0, MCP producer, Spring Boot wrapper, tiktoken |
