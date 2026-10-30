
from pathlib import Path
import json
import os

import mlflow
import tensorflow as tf

PROJECT_DIR = Path("/workspace/NeuroVision-AI-Part-2")
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "best_model.keras"
METRICS_PATH = ARTIFACTS_DIR / "evaluation_metrics.json"
CANDIDATE_PATH = ARTIFACTS_DIR / "candidate_version.json"

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "https://andrea-innocent-officer-divisions.trycloudflare.com",
)

EXPERIMENT_NAME = "MobileNelV2"
MODEL_NAME = "mobileNetV2_IMR"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found.")

    if not METRICS_PATH.exists():
        raise FileNotFoundError("Evaluation metrics not found.")

    with open(METRICS_PATH, encoding="utf-8") as file:
        metrics = json.load(file)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_registry_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    model = tf.keras.models.load_model(MODEL_PATH)

    with mlflow.start_run(
        run_name="airflow_finetuning"
    ) as run:

        mlflow.log_params({
            "type_model": "fine_tuning",
            "architecture": "MobileNetV2",
            "image_size": "224x224",
            "batch_size": 42,
        })

        mlflow.log_metrics({
            f"test_{name}": float(value)
            for name, value in metrics.items()
        })

        model_info = mlflow.tensorflow.log_model(
            model=model,
            name="best_model",
        )

        registered_model = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=MODEL_NAME,
        )

        candidate = {
            "model_name": MODEL_NAME,
            "version": str(registered_model.version),
            "run_id": run.info.run_id,
            "test_accuracy": float(metrics["accuracy"]),
        }

    with open(CANDIDATE_PATH, "w", encoding="utf-8") as file:
        json.dump(candidate, file, indent=2)

    print("Candidate registered:", candidate)


if __name__ == "__main__":
    main()
