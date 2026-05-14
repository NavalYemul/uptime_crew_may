"""Tests for day3.airflow_demo — task functions, idempotency, SimpleDAGRunner."""

import pytest
import json
import shutil
from pathlib import Path

from day3.airflow_demo import (
    extract_products,
    chunk_and_embed,
    index_to_vectorstore,
    validate_index,
    notify_complete,
    SimpleDAGRunner,
    create_rag_ingestion_dag_definition,
    show_airflow_code,
    try_real_airflow,
)

TEST_DATE = "2099-12-31"  # Far future date — won't conflict with other test runs


def cleanup():
    """Remove temp DAG output dir after tests."""
    shutil.rmtree(f"/tmp/dag_outputs/{TEST_DATE}", ignore_errors=True)


@pytest.fixture(autouse=True)
def clean_dag_outputs():
    """Clean up before and after each test."""
    cleanup()
    yield
    cleanup()


# ── extract_products ──────────────────────────────────────────

def test_extract_products_returns_dict():
    result = extract_products(execution_date=TEST_DATE)
    assert isinstance(result, dict)


def test_extract_products_has_records():
    result = extract_products(execution_date=TEST_DATE)
    assert "records" in result
    assert isinstance(result["records"], list)
    assert result["count"] > 0


def test_extract_products_has_execution_date():
    result = extract_products(execution_date=TEST_DATE)
    assert result["execution_date"] == TEST_DATE


def test_extract_products_idempotency():
    result1 = extract_products(execution_date=TEST_DATE)
    result2 = extract_products(execution_date=TEST_DATE)
    # Same count — second call hit cache
    assert result1["count"] == result2["count"]


# ── chunk_and_embed ───────────────────────────────────────────

def test_chunk_and_embed_returns_chunks():
    extract_out = extract_products(execution_date=TEST_DATE)
    result = chunk_and_embed(extract_out)
    assert "chunks" in result
    assert isinstance(result["chunks"], list)
    assert result["count"] > 0


def test_chunk_and_embed_more_than_records():
    extract_out = extract_products(execution_date=TEST_DATE)
    chunk_out = chunk_and_embed(extract_out)
    # Chunking should produce >= as many chunks as original records
    assert chunk_out["count"] >= extract_out["count"]


# ── index_to_vectorstore ──────────────────────────────────────

def test_index_to_vectorstore_returns_indexed():
    extract_out = extract_products(execution_date=TEST_DATE)
    chunk_out   = chunk_and_embed(extract_out)
    result = index_to_vectorstore(chunk_out)
    assert "indexed" in result
    assert result["indexed"] >= 0


def test_index_to_vectorstore_has_collection_name():
    extract_out = extract_products(execution_date=TEST_DATE)
    chunk_out   = chunk_and_embed(extract_out)
    result = index_to_vectorstore(chunk_out)
    assert "collection_name" in result
    assert len(result["collection_name"]) > 0


# ── validate_index ────────────────────────────────────────────

def test_validate_index_returns_dict():
    extract_out = extract_products(execution_date=TEST_DATE)
    chunk_out   = chunk_and_embed(extract_out)
    index_out   = index_to_vectorstore(chunk_out)
    result = validate_index(index_out)
    assert isinstance(result, dict)


def test_validate_index_has_all_passed():
    extract_out = extract_products(execution_date=TEST_DATE)
    chunk_out   = chunk_and_embed(extract_out)
    index_out   = index_to_vectorstore(chunk_out)
    result = validate_index(index_out)
    assert "all_passed" in result
    assert isinstance(result["all_passed"], bool)


def test_validate_index_queries_tested():
    extract_out = extract_products(execution_date=TEST_DATE)
    chunk_out   = chunk_and_embed(extract_out)
    index_out   = index_to_vectorstore(chunk_out)
    result = validate_index(index_out)
    assert result["queries_tested"] > 0


# ── SimpleDAGRunner ───────────────────────────────────────────

def test_simple_dag_runner_executes_all_tasks():
    runner = SimpleDAGRunner("test_dag")
    runner.add_task("task_a", lambda execution_date: {"val": 1, "execution_date": execution_date}, depends_on=[])
    runner.add_task("task_b", lambda ti: {"val": ti["val"] + 1, "execution_date": ti["execution_date"]}, depends_on=["task_a"])

    results = runner.run(TEST_DATE)
    assert len(results) == 2
    assert all(r.status == "success" for r in results)


def test_simple_dag_runner_topological_order():
    runner = SimpleDAGRunner("order_test")
    executed = []

    def task_first(execution_date):
        executed.append("first")
        return {"execution_date": execution_date}

    def task_second(ti):
        executed.append("second")
        return {}

    runner.add_task("first",  task_first,  depends_on=[])
    runner.add_task("second", task_second, depends_on=["first"])
    runner.run(TEST_DATE)

    assert executed.index("first") < executed.index("second")


def test_simple_dag_runner_task_results_have_timing():
    runner = SimpleDAGRunner("timing_test")
    runner.add_task("solo", lambda execution_date: {}, depends_on=[])
    results = runner.run(TEST_DATE)
    assert results[0].duration_s >= 0.0


def test_simple_dag_runner_full_pipeline():
    runner = SimpleDAGRunner("full_pipeline_test")
    runner.add_task("extract",  extract_products,    depends_on=[])
    runner.add_task("chunk",    chunk_and_embed,      depends_on=["extract"])
    runner.add_task("index",    index_to_vectorstore, depends_on=["chunk"])
    runner.add_task("validate", validate_index,       depends_on=["index"])
    runner.add_task("notify",   notify_complete,      depends_on=["validate"])

    results = runner.run(TEST_DATE)
    assert len(results) == 5
    assert all(r.status == "success" for r in results)


# ── show_airflow_code ─────────────────────────────────────────

def test_show_airflow_code_returns_string():
    code = show_airflow_code()
    assert isinstance(code, str)
    assert len(code) > 100


def test_show_airflow_code_contains_dag():
    code = show_airflow_code()
    assert "DAG" in code


def test_show_airflow_code_contains_task_chain():
    code = show_airflow_code()
    assert ">>" in code
