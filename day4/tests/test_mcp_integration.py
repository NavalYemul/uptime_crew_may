"""Tests for day4.mcp_integration"""
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from day4.mcp_integration import (
    show_mcp_setup, show_mcp_servers,
    mcp_filesystem_read, mcp_github_create_issue, mcp_databricks_run_sql,
    build_mcp_agent
)
from day4.tools_agents import MockLLMWithTools


class TestMCPConcepts:
    def test_show_mcp_setup_returns_code(self):
        code = show_mcp_setup()
        assert "MultiServerMCPClient" in code
        assert "langchain-mcp-adapters" in code
        assert "async" in code

    def test_show_mcp_servers_returns_list(self):
        servers = show_mcp_servers()
        assert isinstance(servers, list)
        assert len(servers) >= 4

    def test_mcp_servers_have_required_fields(self):
        for server in show_mcp_servers():
            assert "name" in server
            assert "package" in server
            assert "tools" in server
            assert "use_case" in server

    def test_databricks_server_included(self):
        servers = show_mcp_servers()
        names = [s["name"] for s in servers]
        assert "databricks" in names


class TestMockMCPTools:
    def test_filesystem_read_known_file(self):
        result = mcp_filesystem_read.invoke({"file_path": "/tmp/products.txt"})
        assert "Product" in result or "catalog" in result

    def test_filesystem_read_unknown_file(self):
        result = mcp_filesystem_read.invoke({"file_path": "/tmp/nonexistent.txt"})
        assert "not found" in result.lower() or "File" in result

    def test_github_create_issue_returns_url(self):
        result = mcp_github_create_issue.invoke({"title": "Test bug", "body": "Something broke"})
        assert "github.com" in result
        assert "Issue #" in result

    def test_databricks_count_query(self):
        result = mcp_databricks_run_sql.invoke({"query": "SELECT count(*) FROM products"})
        assert "count" in result.lower() or "1423" in result

    def test_databricks_revenue_query(self):
        result = mcp_databricks_run_sql.invoke({"query": "SELECT SUM(amount) as revenue FROM sales"})
        assert "revenue" in result.lower() or "45000000" in result


class TestMCPAgent:
    def test_mcp_agent_builds(self):
        llm = MockLLMWithTools(
            tool_name    = "mcp_filesystem_read",
            tool_args    = {"file_path": "/tmp/products.txt"},
            final_answer = "The products file contains 20 items.",
        )
        agent = build_mcp_agent(llm, mock_mode=True)
        assert agent is not None

    def test_mcp_agent_runs(self):
        llm = MockLLMWithTools(
            tool_name    = "mcp_filesystem_read",
            tool_args    = {"file_path": "/tmp/products.txt"},
            final_answer = "Found product catalog.",
        )
        agent = build_mcp_agent(llm, mock_mode=True)
        result = agent.invoke({"messages": [HumanMessage("Read the products file")]})
        assert "messages" in result
        assert len(result["messages"]) > 1
