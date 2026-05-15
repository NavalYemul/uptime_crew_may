
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import csv

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator


def generate_sales(**context):
    out = Path("/tmp/raw_sales.csv")
    rows = [
        {"order_id": 101, "amount": 500},
        {"order_id": 102, "amount": 700},
        {"order_id": 103, "amount": 900},
    ]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["order_id", "amount"])
        w.writeheader()
        w.writerows(rows)


def add_tax(**context):
    inp = Path("/tmp/raw_sales.csv")
    out = Path("/tmp/final_sales.csv")

    with inp.open() as f_in, out.open("w", newline="") as f_out:
        r = csv.DictReader(f_in)
        w = csv.DictWriter(f_out, fieldnames=["order_id", "amount", "tax", "total"])
        w.writeheader()

        for row in r:
            amount = float(row["amount"])
            tax = amount * 0.18
            total = amount + tax
            w.writerow(
                {
                    "order_id": int(row["order_id"]),
                    "amount": amount,
                    "tax": tax,
                    "total": total,
                }
            )


with DAG(
    dag_id="sales_pipeline_demo",
    start_date=datetime(2024, 1, 1),
    schedule="*/2 * * * *",  # cron: every 2 minutes
    catchup=False,
    tags=["demo"],
) as dag:
    t1 = PythonOperator(task_id="generate_sales", python_callable=generate_sales)

    t2 = PythonOperator(task_id="add_tax", python_callable=add_tax)

    t3 = BashOperator(task_id="done", bash_command="echo Pipeline finished")

    # Task dependencies
    t1 >> t2 >> t3
