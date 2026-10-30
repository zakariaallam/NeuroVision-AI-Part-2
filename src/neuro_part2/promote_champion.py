
from pathlib import Path
import json
import os

import mlflow
from mlflow import MlflowClient

PROJECT_DIR = Path("/workspace/NeuroVision-AI-Part-2")
CANDIDATE_PATH = PROJECT_DIR / "artifacts" / "candidate_version.json"

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "https://andrea-innocent-officer-divisions.trycloudflare.com",
)

MODEL_NAME = "mobileNetV2_IMR"
ALIAS = "champion"


def main():
    if not CANDIDATE_PATH.exists():
        raise FileNotFoundError("Candidate information not found.")

    with open(CANDIDATE_PATH, encoding="utf-8") as file:
        candidate = json.load(file)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_registry_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient()

    candidate_version = candidate["version"]
    candidate_accuracy = float(candidate["test_accuracy"])

    # Check whether a champion already exists.
    try:
        current = client.get_model_version_by_alias(
            MODEL_NAME,
            ALIAS,
        )
    except Exception:
        # First model: assign champion only if no alias exists.
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias=ALIAS,
            version=candidate_version,
        )
        print(
            "No existing champion found. "
            f"Version {candidate_version} is now champion."
        )
        return

    if str(current.version) == str(candidate_version):
        print("Candidate is already champion.")
        return

    current_run = client.get_run(current.run_id)

    # Metrics are logged with the test_ prefix in register_model.py.
    current_accuracy = current_run.data.metrics.get(
        "test_accuracy"
    )

    if current_accuracy is None:
        raise ValueError(
            "Cannot compare: current champion has no test_accuracy. "
            "Champion alias was not changed."
        )

    print("Current champion accuracy:", current_accuracy)
    print("Candidate accuracy:", candidate_accuracy)

    if candidate_accuracy > current_accuracy:
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias=ALIAS,
            version=candidate_version,
        )
        print(
            f"Champion updated to version {candidate_version}."
        )
    else:
        print(
            "Candidate is not better. "
            "Champion alias remains unchanged."
        )


if __name__ == "__main__":
    main()
