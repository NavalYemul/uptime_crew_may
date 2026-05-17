"""
mcp_integration.py — Model Context Protocol (MCP) Integration
==============================================================
Covers: MCP overview, MultiServerMCPClient, MCP tools as LangChain tools,
        building LangGraph agents with MCP tools.

What is MCP?
  Model Context Protocol (Anthropic, 2024) is a standard interface for
  connecting AI models to external tools, APIs, and data sources.

  Instead of hardcoding tool integrations, MCP lets you:
    - Connect to any MCP server (filesystem, databases, APIs, GitHub, etc.)
    - Each server exposes tools, resources, and prompts via a standard interface
    - Your agent discovers available tools at runtime

  MCP is to AI tools what REST is to web APIs.

Architecture:
  Your Agent  <->  MCP Client  <->  MCP Server(s)
                                  |-- filesystem server (read/write files)
                                  |-- github server (repos, issues, PRs)
                                  |-- postgres server (database queries)
                                  |-- brave-search server (web search)
                                  +-- custom business logic servers

Run: python -m day4.mcp_integration
"""

from __future__ import annotations

from typing import TypedDict, Annotated, Optional, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# ══════════════════════════════════════════════════════
# MCP CONCEPTS (shown as formatted strings for teaching)
# ══════════════════════════════════════════════════════

def show_mcp_setup() -> str:
    """Return the full MCP client setup code for classroom display.

    This is valid Python that runs when langchain-mcp-adapters is installed.
    Install: pip install langchain-mcp-adapters mcp

    Returns:
        Multi-line string with MCP setup code.
    """
    return '''
# --- MCP Setup ----------------------------------------------------------------
# pip install langchain-mcp-adapters mcp
# ------------------------------------------------------------------------------

import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

load_dotenv()

# 1. Define which MCP servers to connect to
mcp_config = {
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        "transport": "stdio",
    },
    "brave-search": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
        "env": {"BRAVE_API_KEY": "your_brave_api_key"},
        "transport": "stdio",
    },
}

# 2. Connect and discover tools
async def run_mcp_agent():
    async with MultiServerMCPClient(mcp_config) as client:
        tools = client.get_tools()   # dynamically discovered at runtime
        print(f"Connected tools: {[t.name for t in tools]}")

        llm = ChatOpenAI(model="gpt-4o-mini")
        agent = create_react_agent(llm, tools)

        result = await agent.ainvoke({
            "messages": [HumanMessage("List files in /tmp and search for LangGraph docs")]
        })
        print(result["messages"][-1].content)

asyncio.run(run_mcp_agent())
# ------------------------------------------------------------------------------
'''


def show_mcp_servers() -> list[dict]:
    """Return a catalogue of popular MCP servers.

    Returns:
        List of dicts describing MCP servers.
    """
    return [
        {
            "name":     "filesystem",
            "package":  "@modelcontextprotocol/server-filesystem",
            "tools":    ["read_file", "write_file", "list_directory", "search_files"],
            "use_case": "Let the agent read/write local files without manual code",
        },
        {
            "name":     "github",
            "package":  "@modelcontextprotocol/server-github",
            "tools":    ["create_issue", "list_pull_requests", "get_file_contents", "push_files"],
            "use_case": "Agent creates GitHub issues, PRs, and reads repo code",
        },
        {
            "name":     "postgres",
            "package":  "@modelcontextprotocol/server-postgres",
            "tools":    ["query", "describe_table", "list_tables"],
            "use_case": "Agent queries your production database in natural language",
        },
        {
            "name":     "brave-search",
            "package":  "@modelcontextprotocol/server-brave-search",
            "tools":    ["brave_web_search", "brave_local_search"],
            "use_case": "Real-time web search without DuckDuckGo rate limits",
        },
        {
            "name":     "slack",
            "package":  "@modelcontextprotocol/server-slack",
            "tools":    ["send_message", "list_channels", "get_channel_history"],
            "use_case": "Agent posts to Slack, reads channel history, sends alerts",
        },
        {
            "name":     "databricks",
            "package":  "databricks-labs/mcp-server-databricks",
            "tools":    ["run_job", "query_sql", "list_notebooks", "run_notebook"],
            "use_case": "Agent triggers Databricks jobs, runs SQL, manages notebooks",
        },
    ]


# ══════════════════════════════════════════════════════
# MOCK MCP TOOLS (no MCP installation required)
# ══════════════════════════════════════════════════════

@tool
def mcp_filesystem_read(file_path: str) -> str:
    """Mock MCP filesystem read_file tool.

    In production: this is provided by the @modelcontextprotocol/server-filesystem MCP server.

    Args:
        file_path: Path to file to read.

    Returns:
        File contents string.
    """
    mock_files = {
        "/tmp/products.txt": "Product catalog: 20 items including laptops, phones, headphones.",
        "/tmp/config.json":  '{"model": "gpt-4o-mini", "temperature": 0, "max_tokens": 512}',
        "/tmp/report.md":    "# Monthly Report\n\nRevenue: Rs 45,00,000\nGrowth: +12%",
    }
    return mock_files.get(file_path, f"File not found: {file_path}")


@tool
def mcp_github_create_issue(title: str, body: str, repo: str = "demo/repo") -> str:
    """Mock MCP GitHub create_issue tool.

    In production: provided by @modelcontextprotocol/server-github MCP server.

    Args:
        title: Issue title.
        body:  Issue description.
        repo:  GitHub repo in owner/repo format.

    Returns:
        Issue URL string.
    """
    issue_number = abs(hash(title)) % 1000 + 1
    return f"Issue #{issue_number} created: https://github.com/{repo}/issues/{issue_number}"


@tool
def mcp_databricks_run_sql(query: str) -> str:
    """Mock MCP Databricks SQL query tool.

    In production: provided by a Databricks MCP server.

    Args:
        query: SQL query to run on Databricks.

    Returns:
        Query result as string.
    """
    query_lower = query.lower()
    if "count" in query_lower:
        return "Result: count(*) = 1423"
    elif "revenue" in query_lower or "amount" in query_lower:
        return "Result: total_revenue = 45000000.00 INR"
    elif "select" in query_lower:
        return "Result: 10 rows returned (laptop: 450, headphones: 280, phones: 693)"
    else:
        return "Query executed successfully. 0 rows affected."


class AgentState(TypedDict):
    """Agent state for MCP-enabled agent."""
    messages: Annotated[list[BaseMessage], add_messages]


def build_mcp_agent(llm, mock_mode: bool = True):
    """Build a LangGraph agent using MCP tools.

    In mock mode: uses the mock tools above (no MCP installation required).
    In real mode: uses langchain-mcp-adapters to connect to real MCP servers.

    Args:
        llm:       LangChain LLM with .bind_tools() method.
        mock_mode: If True, use mock tools. If False, try real MCP.

    Returns:
        Compiled LangGraph agent graph.
    """
    if mock_mode:
        tools = [mcp_filesystem_read, mcp_github_create_issue, mcp_databricks_run_sql]
    else:
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            raise NotImplementedError(
                "Real MCP requires async context manager. "
                "Use: async with MultiServerMCPClient(config) as client: tools = client.get_tools()"
            )
        except ImportError:
            print("[MCP] langchain-mcp-adapters not installed — using mock tools")
            tools = [mcp_filesystem_read, mcp_github_create_issue, mcp_databricks_run_sql]

    from day4.tools_agents import MockLLMWithTools

    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def agent_node(state: AgentState) -> dict:
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")

    return g.compile()


# ══════════════════════════════════════════════════════
# MAIN DEMO
# ══════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("MCP INTEGRATION DEMO")
    print("=" * 70)

    print("\n[1] Available MCP servers:")
    for server in show_mcp_servers():
        print(f"\n  [{server['name']}] — {server['package']}")
        print(f"    Tools:     {', '.join(server['tools'][:3])}")
        print(f"    Use case:  {server['use_case']}")

    print("\n[2] Mock MCP tools:")
    print(f"  filesystem_read('/tmp/products.txt'): {mcp_filesystem_read.invoke({'file_path': '/tmp/products.txt'})[:60]}")
    print(f"  github_create_issue('Bug: crash'): {mcp_github_create_issue.invoke({'title': 'Bug: crash', 'body': 'App crashes on startup'})}")
    print(f"  databricks_run_sql('SELECT count(*) FROM products'): {mcp_databricks_run_sql.invoke({'query': 'SELECT count(*) FROM products'})}")

    print("\n[3] MCP Setup code (real implementation):")
    print(show_mcp_setup()[:600])

    print("\n[4] MCP vs Direct Tool Integration:")
    print("  Direct:  hardcode each API's SDK and auth (maintainability nightmare)")
    print("  MCP:     standard protocol — add new servers without changing agent code")
    print("  Analogy: MCP is to AI tools what USB is to hardware peripherals")


if __name__ == "__main__":
    main()
