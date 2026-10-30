
from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

PROJECT_DIR = Path("/workspace/NeuroVision-AI-Part-2")
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

DATA_PATH = ARTIFACTS_DIR / "dataset.npz"
MODEL_PATH = ARTIFACTS_DIR / "best_model.keras"
METRICS_PATH = ARTIFACTS_DIR / "evaluation_metrics.json"

BATCH_SIZE = 42


def prepare_data(image, label):
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    return image, label


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Dataset splits not found.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Best model not found.")

    data = np.load(DATA_PATH)

    dataset_test = tf.data.Dataset.from_tensor_slices(
        (data["x_test"], data["y_test"])
    )

    dataset_test = (
        dataset_test
        .map(
            prepare_data,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = tf.keras.models.load_model(MODEL_PATH)

    results = model.evaluate(
        dataset_test,
        verbose=1,
        return_dict=True,
    )

    results = {
        name: float(value)
        for name, value in results.items()
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("Test results:", results)


if __name__ == "__main__":
    main()
