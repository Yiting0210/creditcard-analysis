"""Daily credit-risk batch: load and clean the day's applications, then score them.

Training is a separate, manual step (`python -m src.train`); this DAG only uses the current model bundle.
Each task is idempotent for its run date, so Airflow retries and backfills never duplicate rows.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


def _load(ds, **_):
    from src.score import load_applications
    print(f"loaded {load_applications(ds)} applications for {ds}")


def _score(ds, **_):
    from src.score import score_applications
    print(f"scored {score_applications(ds)} applications for {ds}")


with DAG(
    dag_id="credit_risk_pipeline",
    description="Validate, clean and score new credit applications",
    schedule="0 1 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["credit-risk"],
) as dag:
    load = PythonOperator(task_id="load_and_clean", python_callable=_load)
    score = PythonOperator(task_id="score", python_callable=_score)
    load >> score
