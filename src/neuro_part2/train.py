
from pathlib import Path
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

PROJECT_DIR = Path("/workspace/NeuroVision-AI-Part-2")
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
DATA_PATH = ARTIFACTS_DIR / "dataset.npz"

MODEL_PATH = ARTIFACTS_DIR / "best_model.keras"
BATCH_SIZE = 42
EPOCHS = 15


def prepare_data(image, label):
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    return image, label

def make_dataset(images, labels, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices(
        (images, labels)
    )

    if shuffle:
        dataset = dataset.shuffle(1000, seed=42)

    return (
        dataset
        .map(
            prepare_data,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


def create_model(base_model):
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(4, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Run prepare_data.py first."
        )

    data = np.load(DATA_PATH)

    dataset_train = make_dataset(
        data["x_train"], data["y_train"], shuffle=True
    )
    dataset_val = make_dataset(
        data["x_val"], data["y_val"]
    )

    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )

    base_model.trainable = False
    model = create_model(base_model)

    # Transfer Learning
    checkpoint_transfer = (
        ARTIFACTS_DIR / "model_transfer.keras"
    )

    callbacks_transfer = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_transfer),
            monitor="val_loss",
            save_best_only=False,
        ),
    ]

    start_time = time.time()

    model.fit(
        dataset_train,
        epochs=EPOCHS,
        validation_data=dataset_val,
        callbacks=callbacks_transfer,
    )

    print(
        "Transfer Learning duration:",
        time.time() - start_time,
    )

    # Fine-tuning: unfreeze last 30 layers
    base_model.trainable = True

    for layer in base_model.layers[:-30]:
        layer.trainable = False

    for layer in base_model.layers[-30:]:
        layer.trainable = True

    print(
        "Trainable base layers:",
        sum(layer.trainable for layer in base_model.layers),
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-5
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_fine_tuning = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
        ),
    ]

    start_time = time.time()

    model.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=EPOCHS,
        callbacks=callbacks_fine_tuning,
    )

    print(
        "Fine-tuning duration:",
        time.time() - start_time,
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not created: {MODEL_PATH}"
        )

    print("Best model saved:", MODEL_PATH)


if __name__ == "__main__":
    main()
