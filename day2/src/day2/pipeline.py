"""
pipeline.py — Data Pipeline Architecture
==========================================
Covers: ETL vs ELT, batch vs stream, DAG concept,
        PySpark basics, Parquet/CSV/JSON interchange.

Run:  python -m day2.pipeline
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator
import tempfile

import pandas as pd
import numpy as np


# ══════════════════════════════════════════════════════
# ETL vs ELT
# ══════════════════════════════════════════════════════
"""
ETL (Extract → Transform → Load)
  - Transform data BEFORE loading into the warehouse
  - Traditional approach: data warehouse can't handle raw messy data
  - Tools: Informatica, SSIS, Talend, custom Python scripts
  - When to use: target system has strict schema (relational DB, legacy warehouse)

ELT (Extract → Load → Transform)
  - Load raw data FIRST, transform INSIDE the warehouse using SQL/Spark
  - Modern approach: cloud warehouses (BigQuery, Snowflake, Databricks) are powerful
  - Tools: dbt, Databricks Delta Live Tables, Fivetran + dbt
  - When to use: modern cloud data lake / lakehouse (almost always now)

Why ELT won:
  1. Storage is cheap → keep raw data forever (audit trail)
  2. Modern warehouses can transform at scale
  3. Re-transforming is easy when business logic changes
  4. dbt makes SQL transformations version-controlled and testable
"""


class PipelinePattern(str, Enum):
    ETL = "etl"
    ELT = "elt"


@dataclass
class RawRecord:
    """Simulates a raw record as it arrives from a source system."""
    id:        str
    name:      str
    amount:    str         # raw string — could be "1,234.56" or "N/A"
    category:  str
    timestamp: str         # ISO string


def extract_from_source(n: int = 50) -> list[RawRecord]:
    """
    Simulates extracting raw records from a source API or database.
    Notice: amounts are strings with commas, timestamps are strings.
    This is realistic — source systems rarely give you clean types.
    """
    rng = np.random.default_rng(42)
    categories = ["sales", "refund", "subscription", "ad_spend", "royalty"]

    records = []
    for i in range(n):
        amount = rng.uniform(10, 10000)
        amount_str = f"{amount:,.2f}" if rng.random() > 0.1 else "N/A"
        records.append(RawRecord(
            id        = f"TXN-{i:06d}",
            name      = f"Customer {chr(65 + (i % 26))}{i:03d}",
            amount    = amount_str,
            category  = categories[i % len(categories)],
            timestamp = datetime(2024, 1 + (i % 12), 1 + (i % 28)).isoformat(),
        ))
    return records


def etl_transform(records: list[RawRecord]) -> pd.DataFrame:
    """
    ETL pattern: transform BEFORE loading.
    All cleaning and type-casting happens here in Python before touching the DB.
    """
    rows = []
    skipped = 0
    for r in records:
        # Clean amount
        if r.amount == "N/A":
            skipped += 1
            continue
        clean_amount = float(r.amount.replace(",", ""))
        rows.append({
            "id":        r.id,
            "name":      r.name.strip().title(),
            "amount":    clean_amount,
            "category":  r.category.lower(),
            "timestamp": pd.to_datetime(r.timestamp),
            "year":      pd.to_datetime(r.timestamp).year,
            "month":     pd.to_datetime(r.timestamp).month,
        })
    print(f"[ETL] Transformed {len(rows)} records (skipped {skipped} invalid)")
    return pd.DataFrame(rows)


def elt_load_raw(records: list[RawRecord]) -> pd.DataFrame:
    """
    ELT pattern: load raw data AS-IS into the 'bronze' layer.
    Preserve everything — even the messy N/A values.
    """
    df = pd.DataFrame([
        {"id": r.id, "name": r.name, "amount": r.amount,
         "category": r.category, "timestamp": r.timestamp,
         "_ingested_at": datetime.now().isoformat()}
        for r in records
    ])
    print(f"[ELT] Loaded {len(df)} raw records to bronze layer")
    return df


def elt_transform_in_warehouse(bronze_df: pd.DataFrame) -> pd.DataFrame:
    """
    ELT pattern: transform INSIDE the warehouse (simulated with pandas).
    In production this would be a dbt model or Databricks SQL notebook.
    """
    silver = bronze_df.copy()
    silver = silver[silver["amount"] != "N/A"]                   # filter nulls
    silver["amount"]    = silver["amount"].str.replace(",", "").astype(float)
    silver["name"]      = silver["name"].str.strip().str.title()
    silver["category"]  = silver["category"].str.lower()
    silver["timestamp"] = pd.to_datetime(silver["timestamp"])
    silver["year"]      = silver["timestamp"].dt.year
    silver["month"]     = silver["timestamp"].dt.month
    print(f"[ELT] Transformed {len(silver)} records in warehouse (silver layer)")
    return silver


# ══════════════════════════════════════════════════════
# BATCH vs STREAM
# ══════════════════════════════════════════════════════
"""
Batch Processing:
  - Process data in bulk on a schedule (hourly, daily, weekly)
  - High throughput, higher latency
  - Examples: end-of-day reports, model retraining, data warehouse ETL
  - Tools: Apache Spark, pandas, Databricks Jobs

Stream Processing:
  - Process data continuously as each record arrives
  - Low latency (milliseconds to seconds)
  - Examples: fraud detection, live dashboards, IoT sensor alerts
  - Tools: Apache Kafka + Flink/Spark Streaming, AWS Kinesis, Azure Event Hubs

Choosing:
  Need results in < 1 second? → Stream
  Batch jobs < 15 minutes acceptable? → Batch (simpler, cheaper)
  Need both? → Lambda architecture (batch + speed layer) or Kappa (stream only)
"""


def batch_processor(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate a batch job: aggregate sales by category and month.
    Runs on a schedule (e.g., every midnight).
    Processes ALL records in one go.
    """
    return (
        df.groupby(["category", "month"])
        .agg(
            total_amount = ("amount", "sum"),
            num_txns     = ("amount", "count"),
            avg_amount   = ("amount", "mean"),
        )
        .round(2)
        .reset_index()
        .sort_values(["month", "total_amount"], ascending=[True, False])
    )


def stream_processor(
    record_stream: Iterator[dict],
    window_size: int = 5,
) -> Iterator[dict]:
    """
    Simulate a streaming processor: compute running average over a sliding window.
    In production: Apache Kafka → Flink/Spark Structured Streaming.

    This generator yields a summary after every `window_size` records.
    """
    window: list[float] = []
    total_processed = 0

    for record in record_stream:
        if record.get("amount") and isinstance(record["amount"], (int, float)):
            window.append(record["amount"])
            if len(window) > window_size:
                window.pop(0)
            total_processed += 1

            if total_processed % window_size == 0:
                yield {
                    "total_processed": total_processed,
                    "window_avg":      round(sum(window) / len(window), 2),
                    "window_max":      round(max(window), 2),
                    "window_min":      round(min(window), 2),
                    "timestamp":       datetime.now().isoformat(),
                }


# ══════════════════════════════════════════════════════
# DAG CONCEPT
# ══════════════════════════════════════════════════════
"""
DAG = Directed Acyclic Graph

A pipeline is a DAG where:
  - Nodes = tasks (functions / operators)
  - Edges = dependencies (task A must complete before task B starts)
  - Acyclic = no circular dependencies (prevents infinite loops)

Production DAG tools:
  - Apache Airflow (most popular, Python-based)
  - Databricks Workflows (drag-and-drop + Python)
  - Prefect / Dagster (modern alternatives)
  - dbt (DAG for SQL transformations)

Example DAG for a RAG pipeline:
  extract → validate → chunk → embed → load_vector_db
                 ↘ generate_report
"""


@dataclass
class Task:
    name:         str
    fn:           Callable
    depends_on:   list[str] = field(default_factory=list)
    description:  str = ""


class DAG:
    """
    Simple DAG executor for data pipelines.
    Topologically sorts tasks and runs them in dependency order.
    """

    def __init__(self, name: str):
        self.name  = name
        self.tasks: dict[str, Task] = {}
        self._results: dict[str, Any] = {}

    def add_task(self, task: Task) -> None:
        self.tasks[task.name] = task

    def _topological_sort(self) -> list[str]:
        """Kahn's algorithm for topological sort."""
        in_degree = {name: 0 for name in self.tasks}
        for task in self.tasks.values():
            for dep in task.depends_on:
                in_degree[task.name] = in_degree.get(task.name, 0) + 1

        queue = [name for name, deg in in_degree.items() if deg == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for task in self.tasks.values():
                if node in task.depends_on:
                    in_degree[task.name] -= 1
                    if in_degree[task.name] == 0:
                        queue.append(task.name)

        if len(order) != len(self.tasks):
            raise ValueError("DAG has a cycle!")
        return order

    def run(self, initial_input: Any = None) -> dict[str, Any]:
        """Execute all tasks in topological order."""
        order = self._topological_sort()
        print(f"\n[DAG:{self.name}] Execution order: {' → '.join(order)}")

        self._results["__input__"] = initial_input

        for task_name in order:
            task = self.tasks[task_name]
            # Gather inputs from dependencies
            if task.depends_on:
                dep_results = {dep: self._results[dep] for dep in task.depends_on}
                inp = list(dep_results.values())[0] if len(dep_results) == 1 else dep_results
            else:
                inp = initial_input

            t0 = time.perf_counter()
            result = task.fn(inp)
            elapsed = time.perf_counter() - t0

            self._results[task_name] = result
            print(f"  ✓ {task_name} ({elapsed*1000:.1f}ms)")

        return self._results


def build_ingestion_dag(records: list[RawRecord]) -> DAG:
    """
    Build a DAG for the Day 2 ingestion pipeline:
    extract → load_raw → transform → aggregate → report
    """
    dag = DAG("rag_ingestion_pipeline")

    # Task 1: load raw (bronze)
    dag.add_task(Task(
        name        = "load_bronze",
        fn          = lambda _: elt_load_raw(records),
        depends_on  = [],
        description = "Load raw records to bronze layer",
    ))

    # Task 2: transform to silver (depends on bronze)
    dag.add_task(Task(
        name        = "transform_silver",
        fn          = elt_transform_in_warehouse,
        depends_on  = ["load_bronze"],
        description = "Clean and type-cast to silver layer",
    ))

    # Task 3: aggregate (depends on silver)
    dag.add_task(Task(
        name        = "aggregate_gold",
        fn          = batch_processor,
        depends_on  = ["transform_silver"],
        description = "Aggregate metrics to gold layer",
    ))

    # Task 4: report (depends on gold)
    def generate_report(df: pd.DataFrame) -> dict:
        return {
            "total_categories": df["category"].nunique(),
            "total_revenue":    round(float(df["total_amount"].sum()), 2),
            "top_category":     df.nlargest(1, "total_amount")["category"].iloc[0],
            "generated_at":     datetime.now().isoformat(),
        }

    dag.add_task(Task(
        name        = "generate_report",
        fn          = generate_report,
        depends_on  = ["aggregate_gold"],
        description = "Generate summary report",
    ))

    return dag


# ══════════════════════════════════════════════════════
# PYSPARK BASICS
# ══════════════════════════════════════════════════════

def pyspark_demo(df_pandas: pd.DataFrame) -> dict | str:
    """
    PySpark basics: create a Spark DataFrame, apply transformations.

    PySpark vs Pandas:
      Pandas:  in-memory, single machine, 100M rows max comfortably
      PySpark: distributed across a cluster, billion+ rows, lazy evaluation

    Lazy evaluation: transformations are NOT executed until you call
    .show(), .collect(), .count(), or .write() — the plan is built first,
    then optimised and executed.

    Requires Java 11+. Gracefully degrades if not available.
    """
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F

        spark = (SparkSession.builder
                 .appName("day2-demo")
                 .master("local[2]")    # use 2 local threads
                 .config("spark.ui.enabled", "false")
                 .config("spark.sql.shuffle.partitions", "4")
                 .getOrCreate())
        spark.sparkContext.setLogLevel("ERROR")

        # Convert pandas → Spark DataFrame
        sdf = spark.createDataFrame(df_pandas)

        # Transformations (lazy — not executed yet)
        result = (
            sdf
            .filter(F.col("amount") > 100)
            .groupBy("category")
            .agg(
                F.count("*").alias("count"),
                F.round(F.avg("amount"), 2).alias("avg_amount"),
                F.round(F.sum("amount"), 2).alias("total"),
            )
            .orderBy(F.desc("total"))
        )

        # Action — executes the plan
        rows = result.collect()

        with tempfile.TemporaryDirectory() as tmpdir:
            sdf.write.parquet(f"{tmpdir}/output", mode="overwrite")
            parquet_files = list(Path(tmpdir, "output").glob("*.parquet"))
            parquet_count = len(parquet_files)

        spark.stop()

        return {
            "spark_rows":    sdf.count(),
            "spark_schema":  str(sdf.schema.simpleString()),
            "grouped_result": [r.asDict() for r in rows],
            "parquet_parts": parquet_count,
        }

    except Exception as e:
        return f"PySpark not available: {type(e).__name__}: {str(e)[:100]}. " \
               f"Install Java 11+ and run: uv pip install 'day2-embeddings-rag[spark]'"


# ══════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    import pprint

    print("=" * 60)
    print("DATA PIPELINE DEMO")
    print("=" * 60)

    # ETL vs ELT
    print("\n[1] Extract raw records")
    records = extract_from_source(50)
    print(f"  Extracted {len(records)} raw records")

    print("\n[2] ETL pattern (transform before load)")
    df_etl = etl_transform(records)
    print(df_etl.head(3).to_string(index=False))

    print("\n[3] ELT pattern (load raw, transform in warehouse)")
    bronze = elt_load_raw(records)
    silver = elt_transform_in_warehouse(bronze)
    print(f"  Bronze: {len(bronze)} rows | Silver: {len(silver)} rows")

    print("\n[4] Batch aggregation (gold layer)")
    gold = batch_processor(silver)
    print(gold.head(6).to_string(index=False))

    print("\n[5] Stream simulation (sliding window)")
    stream = ({"amount": float(r.amount.replace(",", ""))}
              for r in records if r.amount != "N/A")
    for summary in stream_processor(stream, window_size=5):
        print(f"  [{summary['total_processed']}] "
              f"avg={summary['window_avg']} max={summary['window_max']}")

    print("\n[6] DAG Execution")
    dag     = build_ingestion_dag(records)
    outputs = dag.run()
    print("  Report:", outputs.get("generate_report"))

    print("\n[7] PySpark Demo")
    spark_result = pyspark_demo(silver)
    if isinstance(spark_result, dict):
        print(f"  Spark rows: {spark_result['spark_rows']}")
        for row in spark_result.get("grouped_result", [])[:4]:
            print(f"    {row}")
    else:
        print(f"  {spark_result}")
