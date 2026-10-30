
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_DIR = Path("/workspace/NeuroVision-AI-Part-2")
PYTHON = sys.executable


def run_script(script_name):
    script_path = PROJECT_DIR / "src" / script_name

    if not script_path.is_file():
        raise FileNotFoundError(
            f"Script not found: {script_path}"
        )

    subprocess.run(
        [PYTHON, str(script_path)],
        cwd=str(PROJECT_DIR),
        check=True,
    )


with DAG(
    dag_id="neurovision_ml_pipeline",
    description="MobileNetV2 training and MLflow champion promotion",
    default_args={
        "owner": "neurovision",
        "retries": 2,
        "retry_delay": timedelta(minutes=2),
    },
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["NeuroVision", "MLflow", "MobileNetV2"],
) as dag:

    prepare_data = PythonOperator(
        task_id="prepare_data",
        python_callable=lambda: run_script("prepare_data.py"),
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=lambda: run_script("train.py"),
    )

    evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=lambda: run_script("evaluate.py"),
    )

    register = PythonOperator(
        task_id="register_model",
        python_callable=lambda: run_script("register_model.py"),
    )

    promote = PythonOperator(
        task_id="compare_and_promote_champion",
        python_callable=lambda: run_script("promote_champion.py"),
    )

    prepare_data >> train >> evaluate >> register >> promote
