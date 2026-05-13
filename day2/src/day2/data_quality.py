"""
data_quality.py — Data Validation & Quality Checks
====================================================
Covers: pandera schema validation, Great Expectations concepts,
        data quality rules, assertion suites, pipeline health checks.

Run:  python -m day2.data_quality
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

try:
    import pandera as pa
    from pandera import Column, DataFrameSchema, Check
    PANDERA_OK = True
except ImportError:
    PANDERA_OK = False
    print("[data_quality] pandera not installed: uv pip install pandera")


# ══════════════════════════════════════════════════════
# CONCEPT: DATA QUALITY
# ══════════════════════════════════════════════════════
"""
Data quality is the #1 cause of bad ML models and wrong business decisions.
"Garbage in, garbage out" — a model trained on bad data gives bad predictions.

The 6 dimensions of data quality:
  Completeness  — no missing required values
  Validity      — values conform to expected format/range
  Consistency   — same concept represented the same way across tables
  Accuracy      — values reflect reality
  Uniqueness    — no duplicate records
  Timeliness    — data is up-to-date

Tools:
  pandera      — schema validation for pandas DataFrames (pythonic, testable)
  Great Expectations — enterprise-grade, generates HTML reports, has "suites"
  dbt tests    — data quality as SQL assertions in your dbt models
  Databricks   — Delta Lake constraints + DLT expectations

TODAY: pandera (simple, composable) + conceptual Great Expectations overview.
"""


# ══════════════════════════════════════════════════════
# PANDERA SCHEMA DEFINITIONS
# ══════════════════════════════════════════════════════

def build_transaction_schema() -> "pa.DataFrameSchema | None":
    """
    Define a schema for transaction data.
    Pandera checks every row against these rules.
    Equivalent to SQL CHECK constraints, but in Python.

    Great Expectations equivalent:
      expect_column_values_to_not_be_null("amount")
      expect_column_values_to_be_between("amount", 0, 1_000_000)
      expect_column_values_to_be_in_set("category", ["sales", "refund", ...])
    """
    if not PANDERA_OK:
        return None

    return pa.DataFrameSchema(
        columns={
            "id": pa.Column(
                str,
                checks=pa.Check.str_startswith("TXN-"),
                nullable=False,
                unique=True,
                description="Transaction ID — must start with 'TXN-'",
            ),
            "name": pa.Column(
                str,
                checks=pa.Check(lambda s: s.str.len() > 0, element_wise=False),
                nullable=False,
                description="Customer name — non-empty string",
            ),
            "amount": pa.Column(
                float,
                checks=[
                    pa.Check.greater_than(0),
                    pa.Check.less_than_or_equal_to(1_000_000),
                ],
                nullable=False,
                description="Transaction amount — positive, ≤ 1M",
            ),
            "category": pa.Column(
                str,
                checks=pa.Check.isin(["sales", "refund", "subscription", "ad_spend", "royalty"]),
                nullable=False,
                description="Transaction category — one of allowed values",
            ),
            "timestamp": pa.Column(
                "datetime64[ns]",
                checks=pa.Check(
                    lambda s: s >= pd.Timestamp("2020-01-01"),
                    element_wise=False,
                    error="Timestamps must be after 2020-01-01",
                ),
                nullable=False,
                description="Event timestamp — after 2020-01-01",
            ),
        },
        coerce=True,          # attempt type coercion before checking
        strict=False,         # allow extra columns
    )


def build_embedding_schema(expected_dim: int = 384) -> "pa.DataFrameSchema | None":
    """
    Schema for an embedding result DataFrame.
    Validates that the pipeline produced correct-dimensional vectors.
    """
    if not PANDERA_OK:
        return None

    return pa.DataFrameSchema(
        columns={
            "doc_id":    pa.Column(str, nullable=False, unique=True),
            "text":      pa.Column(str, checks=pa.Check(lambda s: s.str.len() > 0, element_wise=False)),
            "embedding": pa.Column(object, nullable=False),       # list/array
            "dim":       pa.Column(int, checks=pa.Check.equal_to(expected_dim)),
            "norm":      pa.Column(float, checks=pa.Check.in_range(0.99, 1.01)),  # unit vector check
        },
    )


# ══════════════════════════════════════════════════════
# VALIDATION RESULT
# ══════════════════════════════════════════════════════

@dataclass
class ValidationReport:
    table_name:    str
    total_rows:    int
    passed:        bool
    errors:        list[str] = field(default_factory=list)
    warnings:      list[str] = field(default_factory=list)
    checks_run:    int = 0
    checks_passed: int = 0
    timestamp:     str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def pass_rate(self) -> float:
        return round(self.checks_passed / max(self.checks_run, 1), 4)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"[{status}] {self.table_name}: "
            f"{self.checks_passed}/{self.checks_run} checks passed ({self.pass_rate:.0%})",
        ]
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  WARN:  {w}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════
# VALIDATION FUNCTIONS
# ══════════════════════════════════════════════════════

def validate_with_pandera(
    df: pd.DataFrame,
    schema: "pa.DataFrameSchema",
    table_name: str = "data",
) -> ValidationReport:
    """
    Validate a DataFrame against a pandera schema.
    Returns a ValidationReport with all errors collected.
    """
    report = ValidationReport(
        table_name = table_name,
        total_rows = len(df),
        passed     = False,
    )

    if not PANDERA_OK:
        report.errors.append("pandera not installed")
        return report

    try:
        schema.validate(df, lazy=True)
        report.passed     = True
        report.checks_run = len(schema.columns)
        report.checks_passed = len(schema.columns)
    except pa.errors.SchemaErrors as e:
        failure_df = e.failure_cases
        report.checks_run    = len(schema.columns)
        report.checks_passed = len(schema.columns) - failure_df["schema_context"].nunique()
        for _, row in failure_df.iterrows():
            report.errors.append(
                f"{row.get('column', '?')}: {row.get('check', '?')} failed "
                f"(e.g. value={row.get('failure_case', '?')})"
            )

    return report


def validate_completeness(df: pd.DataFrame, required_cols: list[str]) -> ValidationReport:
    """Check for missing values in required columns."""
    report = ValidationReport(
        table_name = "completeness_check",
        total_rows = len(df),
        passed     = True,
    )

    for col in required_cols:
        report.checks_run += 1
        if col not in df.columns:
            report.errors.append(f"Column '{col}' is missing from DataFrame")
            report.passed = False
        else:
            null_count = df[col].isna().sum()
            null_rate  = null_count / len(df)
            if null_count > 0:
                if null_rate > 0.1:   # > 10% null = error
                    report.errors.append(f"'{col}': {null_count} nulls ({null_rate:.1%})")
                    report.passed = False
                else:
                    report.warnings.append(f"'{col}': {null_count} nulls ({null_rate:.1%})")
                    report.checks_passed += 1
            else:
                report.checks_passed += 1

    return report


def validate_uniqueness(df: pd.DataFrame, key_col: str) -> ValidationReport:
    """Check for duplicate primary keys."""
    report = ValidationReport(
        table_name   = f"uniqueness_{key_col}",
        total_rows   = len(df),
        passed       = True,
        checks_run   = 1,
    )

    if key_col not in df.columns:
        report.errors.append(f"Key column '{key_col}' not found")
        report.passed = False
        return report

    dupes = df[key_col].duplicated().sum()
    if dupes > 0:
        examples = df[df[key_col].duplicated(keep=False)][key_col].head(3).tolist()
        report.errors.append(f"{dupes} duplicate values in '{key_col}'. Examples: {examples}")
        report.passed = False
    else:
        report.checks_passed = 1

    return report


def validate_distribution(
    df: pd.DataFrame,
    col: str,
    expected_min: float,
    expected_max: float,
    max_outlier_pct: float = 0.05,
) -> ValidationReport:
    """
    Check that a numeric column's distribution is within expected bounds.
    Catches data quality regressions: e.g., a bug that sends all amounts as 0.
    """
    report = ValidationReport(
        table_name = f"distribution_{col}",
        total_rows = len(df),
        passed     = True,
        checks_run = 3,
    )

    if col not in df.columns:
        report.errors.append(f"Column '{col}' not found")
        report.passed = False
        return report

    series = df[col].dropna()

    # Check 1: range
    if series.min() < expected_min or series.max() > expected_max:
        report.errors.append(
            f"'{col}' range [{series.min():.2f}, {series.max():.2f}] "
            f"outside expected [{expected_min}, {expected_max}]"
        )
        report.passed = False
    else:
        report.checks_passed += 1

    # Check 2: not all identical (constant feature = useless)
    if series.nunique() <= 1:
        report.errors.append(f"'{col}' has only {series.nunique()} unique value(s) — likely a bug")
        report.passed = False
    else:
        report.checks_passed += 1

    # Check 3: outlier rate
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr     = q3 - q1
    outliers = ((series < q1 - 3 * iqr) | (series > q3 + 3 * iqr)).sum()
    outlier_pct = outliers / len(series)
    if outlier_pct > max_outlier_pct:
        report.warnings.append(f"'{col}': {outliers} outliers ({outlier_pct:.1%})")
        report.checks_passed += 1   # warning, not error
    else:
        report.checks_passed += 1

    return report


# ══════════════════════════════════════════════════════
# GREAT EXPECTATIONS OVERVIEW (conceptual + simple demo)
# ══════════════════════════════════════════════════════

def great_expectations_concept() -> dict:
    """
    Great Expectations key concepts (no actual GE installed needed):

    Expectation: a declarative assertion about your data.
      "I expect column 'amount' to be between 0 and 1,000,000"

    Expectation Suite: a collection of expectations for one dataset.

    Checkpoint: runs a suite against a batch of data and generates results.

    Data Docs: auto-generated HTML report showing pass/fail for each expectation.

    In production (Databricks):
      import great_expectations as ge
      context = ge.get_context()
      suite   = context.suites.add(ExpectationSuite(name="transactions"))
      suite.add_expectation(ExpectationConfiguration(
          expectation_type = "expect_column_values_to_not_be_null",
          kwargs           = {"column": "amount"},
      ))
      results = context.run_checkpoint("transactions_checkpoint")
    """
    return {
        "core_concepts": {
            "Expectation":      "One assertion: 'amount must be > 0'",
            "ExpectationSuite": "Named collection of expectations for one table",
            "Checkpoint":       "Runs a suite against a data batch, produces ValidationResult",
            "DataDocs":         "HTML report — auto-generated, shareable with stakeholders",
        },
        "pandera_vs_ge": {
            "pandera":           "Code-first, Pythonic, easy to test, no infra needed",
            "great_expectations": "Enterprise-grade, UI reports, data profiling, team collaboration",
            "recommendation":    "Use pandera in pipelines; GE for stakeholder reporting & auditing",
        },
        "common_expectations": [
            "expect_column_values_to_not_be_null",
            "expect_column_values_to_be_between",
            "expect_column_values_to_be_in_set",
            "expect_column_values_to_match_regex",
            "expect_table_row_count_to_be_between",
            "expect_column_to_exist",
            "expect_column_values_to_be_unique",
            "expect_column_mean_to_be_between",
        ],
    }


# ══════════════════════════════════════════════════════
# FULL VALIDATION SUITE
# ══════════════════════════════════════════════════════

def run_validation_suite(df: pd.DataFrame) -> list[ValidationReport]:
    """
    Run a complete data quality assertion suite on a DataFrame.
    In production: run this at the start of every pipeline step.
    If any check fails, raise an exception to halt the pipeline.
    """
    reports = []

    schema = build_transaction_schema()
    if schema:
        reports.append(validate_with_pandera(df, schema, "transactions"))

    reports.append(validate_completeness(
        df, required_cols=["id", "name", "amount", "category", "timestamp"]
    ))
    reports.append(validate_uniqueness(df, key_col="id"))
    reports.append(validate_distribution(
        df, col="amount", expected_min=0, expected_max=1_000_000
    ))

    return reports


# ══════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    from day2.pipeline import extract_from_source, etl_transform

    print("=" * 60)
    print("DATA QUALITY & VALIDATION DEMO")
    print("=" * 60)

    # Good data
    print("\n[1] Valid dataset")
    records = extract_from_source(50)
    df      = etl_transform(records)
    reports = run_validation_suite(df)
    for r in reports:
        print(r.summary())

    # Inject bad data
    print("\n[2] Inject data quality issues")
    df_bad = df.copy()
    df_bad.loc[0, "amount"]   = -500.0          # negative amount
    df_bad.loc[1, "category"] = "unknown_cat"   # invalid category
    df_bad.loc[2, "id"]       = df_bad.loc[0, "id"]  # duplicate ID

    reports_bad = run_validation_suite(df_bad)
    for r in reports_bad:
        print(r.summary())

    # GE concepts
    print("\n[3] Great Expectations Concepts")
    import pprint
    pprint.pprint(great_expectations_concept())
