"""test_pipeline.py — Data pipeline, DAG & data quality tests"""

import numpy as np
import pandas as pd
import pytest

from day2.pipeline import (
    extract_from_source,
    etl_transform,
    elt_load_raw,
    elt_transform_in_warehouse,
    batch_processor,
    stream_processor,
    build_ingestion_dag,
    DAG,
    Task,
)
from day2.data_quality import (
    validate_completeness,
    validate_uniqueness,
    validate_distribution,
    run_validation_suite,
    great_expectations_concept,
)


# ── Extract ───────────────────────────────────────────

def test_extract_count():
    records = extract_from_source(30)
    assert len(records) == 30


def test_extract_fields():
    records = extract_from_source(5)
    for r in records:
        assert hasattr(r, "id") and r.id.startswith("TXN-")
        assert hasattr(r, "amount")
        assert hasattr(r, "category")
        assert hasattr(r, "timestamp")


# ── ETL ───────────────────────────────────────────────

def test_etl_transform_drops_invalid():
    records = extract_from_source(50)
    df      = etl_transform(records)
    # Should not contain "N/A" amounts
    assert (df["amount"] == "N/A").sum() == 0


def test_etl_transform_types():
    records = extract_from_source(20)
    df      = etl_transform(records)
    assert df["amount"].dtype == float
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


def test_etl_transform_has_derived_cols():
    records = extract_from_source(20)
    df      = etl_transform(records)
    assert "year" in df.columns
    assert "month" in df.columns
    assert (df["year"] == 2024).all()


# ── ELT ───────────────────────────────────────────────

def test_elt_bronze_preserves_all():
    records = extract_from_source(20)
    bronze  = elt_load_raw(records)
    assert len(bronze) == 20


def test_elt_bronze_has_ingestion_col():
    records = extract_from_source(5)
    bronze  = elt_load_raw(records)
    assert "_ingested_at" in bronze.columns


def test_elt_silver_removes_invalid():
    records = extract_from_source(30)
    bronze  = elt_load_raw(records)
    silver  = elt_transform_in_warehouse(bronze)
    assert silver["amount"].dtype == float


def test_elt_silver_fewer_than_bronze():
    records = extract_from_source(50)
    bronze  = elt_load_raw(records)
    silver  = elt_transform_in_warehouse(bronze)
    # Silver may have fewer rows if any amounts were N/A
    assert len(silver) <= len(bronze)


# ── Batch & Stream ────────────────────────────────────

def test_batch_processor_has_categories(sample_df):
    result = batch_processor(sample_df)
    assert "category" in result.columns
    assert "total_amount" in result.columns
    assert len(result) > 0


def test_batch_processor_totals_positive(sample_df):
    result = batch_processor(sample_df)
    assert (result["total_amount"] > 0).all()


def test_stream_processor_yields_summaries():
    data   = [{"amount": float(i * 10)} for i in range(1, 26)]
    stream = iter(data)
    summaries = list(stream_processor(stream, window_size=5))
    assert len(summaries) == 5   # 25 records / 5 window = 5 yields


def test_stream_processor_window_avg():
    data      = [{"amount": 100.0}] * 20
    summaries = list(stream_processor(iter(data), window_size=5))
    for s in summaries:
        assert s["window_avg"] == pytest.approx(100.0)


def test_stream_processor_has_required_keys():
    data      = [{"amount": float(i)} for i in range(10)]
    summaries = list(stream_processor(iter(data), window_size=5))
    for s in summaries:
        assert {"total_processed", "window_avg", "window_max", "window_min"} <= s.keys()


# ── DAG ───────────────────────────────────────────────

def test_dag_topological_order():
    dag = DAG("test")
    dag.add_task(Task("A", fn=lambda x: "A_done", depends_on=[]))
    dag.add_task(Task("B", fn=lambda x: "B_done", depends_on=["A"]))
    dag.add_task(Task("C", fn=lambda x: "C_done", depends_on=["B"]))

    outputs = dag.run(initial_input="start")
    assert outputs["A"] == "A_done"
    assert outputs["B"] == "B_done"
    assert outputs["C"] == "C_done"


def test_dag_cycle_raises():
    dag = DAG("cyclic")
    dag.add_task(Task("A", fn=lambda x: x, depends_on=["B"]))
    dag.add_task(Task("B", fn=lambda x: x, depends_on=["A"]))
    with pytest.raises(ValueError, match="cycle"):
        dag._topological_sort()


def test_dag_parallel_tasks():
    dag = DAG("parallel")
    dag.add_task(Task("root",  fn=lambda x: 42,     depends_on=[]))
    dag.add_task(Task("left",  fn=lambda x: x * 2,  depends_on=["root"]))
    dag.add_task(Task("right", fn=lambda x: x + 10, depends_on=["root"]))

    outputs = dag.run()
    assert outputs["left"]  == 84
    assert outputs["right"] == 52


def test_ingestion_dag_runs():
    records = extract_from_source(20)
    dag     = build_ingestion_dag(records)
    outputs = dag.run()
    assert "generate_report" in outputs
    assert outputs["generate_report"]["total_revenue"] > 0


# ── Data Quality ──────────────────────────────────────

def test_validate_completeness_pass(sample_df):
    report = validate_completeness(sample_df, ["id", "name", "amount"])
    assert report.passed is True


def test_validate_completeness_fails_on_missing_col(sample_df):
    report = validate_completeness(sample_df, ["nonexistent_col"])
    assert report.passed is False
    assert any("nonexistent_col" in e for e in report.errors)


def test_validate_completeness_warns_on_nulls():
    df = pd.DataFrame({"x": [1.0, None, 3.0, None, 5.0]})
    report = validate_completeness(df, ["x"])
    # 2/5 = 40% nulls → should be an error
    assert report.passed is False


def test_validate_uniqueness_pass(sample_df):
    report = validate_uniqueness(sample_df, "id")
    assert report.passed is True
    assert report.checks_passed == 1


def test_validate_uniqueness_fail_on_dupes():
    df = pd.DataFrame({"id": ["A", "B", "A", "C"]})
    report = validate_uniqueness(df, "id")
    assert report.passed is False
    assert any("duplicate" in e.lower() for e in report.errors)


def test_validate_distribution_pass(sample_df):
    report = validate_distribution(sample_df, "amount", 0, 10000)
    assert report.passed is True


def test_validate_distribution_fail_out_of_range():
    df = pd.DataFrame({"amount": [1.0, 2.0, 999999.0]})
    report = validate_distribution(df, "amount", 0, 100)
    assert report.passed is False


def test_run_validation_suite_returns_reports(sample_df):
    reports = run_validation_suite(sample_df)
    assert len(reports) >= 2


def test_ge_concept_has_required_keys():
    r = great_expectations_concept()
    assert "core_concepts" in r
    assert "common_expectations" in r
    assert len(r["common_expectations"]) >= 5
