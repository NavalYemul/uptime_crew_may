# Airflow DAGs — Day 3 RAG Pipeline

Two working DAGs for the Day 3 course module.

## DAGs

### 1. `sales_pipeline_demo` — Simple ETL (recurring)
- **Schedule:** every 2 minutes (`*/2 * * * *`)
- **Tasks:** `generate_sales` → `add_tax` → `done`
- Generates a 3-row CSV, adds 18% GST, prints completion
- Good for understanding `PythonOperator` and `BashOperator`

### 2. `data_pipelines_ingestion` — Medallion Architecture (run once)
- **Schedule:** `@once`
- **Tasks:** `load_bronze` → `transform_silver` → `aggregate_gold` → `generate_report`
- 60 synthetic transactions: raw CSV → cleaned → aggregated → JSON report
- Mirrors the Day 2 data pipeline notebook — same logic as an Airflow DAG
- Output files written to `/tmp/data_pipelines_ingestion/`

---

## Setup (macOS)

```bash
# 1. Create and activate venv
python3 -m venv airflow_venv
source airflow_venv/bin/activate

# 2. Install Airflow (constrained install)
pip install "apache-airflow==2.9.3" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.3/constraints-3.12.txt"

# 3. Point AIRFLOW_HOME to this folder
export AIRFLOW_HOME=$(pwd)/airflow_home

# 4. Copy DAGs
mkdir -p $AIRFLOW_HOME/dags
cp dags/*.py $AIRFLOW_HOME/dags/

# 5. Start Airflow standalone (scheduler + webserver together)
airflow standalone
```

After `airflow standalone` starts:
- Web UI: http://localhost:8080
- Username: `admin`
- Password: shown in terminal output (also in `airflow_home/standalone_admin_password.txt`)

---

## Running a DAG manually

```bash
# Trigger a DAG run from the CLI
airflow dags trigger sales_pipeline_demo
airflow dags trigger data_pipelines_ingestion

# Check task status
airflow tasks states-for-dag-run data_pipelines_ingestion <run_id>

# View output files (medallion DAG)
cat /tmp/data_pipelines_ingestion/report.json
```

---

## Databricks Workflows equivalent

In Databricks, replace this Airflow setup with **Databricks Workflows**:
- No infrastructure to manage (fully managed)
- Native Unity Catalog integration
- Delta Lake as the bronze/silver/gold storage layer
- Same DAG dependency logic: task A → task B → task C
