"""
Medallion ingestion pipeline (bronze → silver → gold → report).
Mirrors the DAG in 02_data_pipelines.ipynb; runs once via @once schedule.
Uses stdlib only so Airflow can load without pandas in the venv.
"""

from __future__ import annotations

import csv
import json
import random
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

DATA_DIR = Path("/tmp/data_pipelines_ingestion")
BRONZE_PATH = DATA_DIR / "bronze.csv"
SILVER_PATH = DATA_DIR / "silver.csv"
GOLD_PATH = DATA_DIR / "gold.csv"
REPORT_PATH = DATA_DIR / "report.json"

CATEGORIES = ["sales", "refund", "subscription", "ad_spend", "royalty"]


def _extract_raw(n: int = 60) -> list[dict]:
    rng = random.Random(42)
    rows = []
    for i in range(n):
        amount = rng.uniform(10, 10000)
        amount_str = f"{amount:,.2f}" if rng.random() > 0.1 else "N/A"
        rows.append(
            {
                "id": f"TXN-{i:06d}",
                "name": f"Customer {chr(65 + (i % 26))}{i:03d}",
                "amount": amount_str,
                "category": CATEGORIES[i % len(CATEGORIES)],
                "timestamp": datetime(2024, 1 + (i % 12), 1 + (i % 28)).isoformat(),
                "_ingested_at": datetime.now().isoformat(),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def load_bronze(**context):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = _extract_raw()
    _write_csv(BRONZE_PATH, rows, list(rows[0].keys()))
    print(f"[bronze] loaded {len(rows)} raw records → {BRONZE_PATH}")


def transform_silver(**context):
    silver = []
    for row in _read_csv(BRONZE_PATH):
        if row["amount"] == "N/A":
            continue
        ts = datetime.fromisoformat(row["timestamp"])
        silver.append(
            {
                "id": row["id"],
                "name": row["name"].strip().title(),
                "amount": float(row["amount"].replace(",", "")),
                "category": row["category"].lower(),
                "timestamp": row["timestamp"],
                "year": ts.year,
                "month": ts.month,
            }
        )
    _write_csv(SILVER_PATH, silver, ["id", "name", "amount", "category", "timestamp", "year", "month"])
    print(f"[silver] transformed {len(silver)} records → {SILVER_PATH}")


def aggregate_gold(**context):
    buckets: dict[tuple[str, int], dict] = {}
    for row in _read_csv(SILVER_PATH):
        key = (row["category"], int(row["month"]))
        amount = float(row["amount"])
        if key not in buckets:
            buckets[key] = {"category": row["category"], "month": int(row["month"]), "total_amount": 0.0, "num_txns": 0}
        buckets[key]["total_amount"] += amount
        buckets[key]["num_txns"] += 1

    gold = []
    for b in buckets.values():
        b["avg_amount"] = round(b["total_amount"] / b["num_txns"], 2)
        b["total_amount"] = round(b["total_amount"], 2)
        gold.append(b)
    gold.sort(key=lambda r: (r["month"], -r["total_amount"]))
    _write_csv(GOLD_PATH, gold, ["category", "month", "total_amount", "num_txns", "avg_amount"])
    print(f"[gold] aggregated {len(gold)} rows → {GOLD_PATH}")


def generate_report(**context):
    gold = _read_csv(GOLD_PATH)
    total_revenue = sum(float(r["total_amount"]) for r in gold)
    top = max(gold, key=lambda r: float(r["total_amount"]))
    categories = {r["category"] for r in gold}
    report = {
        "total_categories": len(categories),
        "total_revenue": round(total_revenue, 2),
        "top_category": top["category"],
        "generated_at": datetime.now().isoformat(),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"[report] {report}")
    return report


with DAG(
    dag_id="data_pipelines_ingestion",
    start_date=datetime(2024, 1, 1),
    schedule="@once",
    catchup=False,
    tags=["day2", "medallion", "demo"],
) as dag:
    t_bronze = PythonOperator(task_id="load_bronze", python_callable=load_bronze)
    t_silver = PythonOperator(task_id="transform_silver", python_callable=transform_silver)
    t_gold = PythonOperator(task_id="aggregate_gold", python_callable=aggregate_gold)
    t_report = PythonOperator(task_id="generate_report", python_callable=generate_report)

    t_bronze >> t_silver >> t_gold >> t_report
