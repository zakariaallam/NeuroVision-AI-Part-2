
from pathlib import Path
import zipfile

import cv2
import gdown
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_DIR = Path("/workspace/NeuroVision-AI-Part-2")
DATA_DIR = PROJECT_DIR / "data"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

FILE_ID = "1a4w4HfwvyzmdIF56TmWL5-z8of0ghE3m"
ZIP_PATH = DATA_DIR / "data.zip"

VALID_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp"}
IMAGE_SIZE = (224, 224)


def load_data(data_dir):
    data_dir = Path(data_dir)

    if data_dir.is_file() and data_dir.suffix.lower() == ".zip":
        extract_dir = data_dir.with_suffix("")

        with zipfile.ZipFile(data_dir, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        data_dir = extract_dir / "Data"

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {data_dir}"
        )

    images = []
    labels = []

    class_names = sorted(
        p.name for p in data_dir.iterdir() if p.is_dir()
    )

    if not class_names:
        raise ValueError("No classes found.")

    encoder = LabelEncoder()
    encoder.fit(class_names)

    for class_name in class_names:
        class_dir = data_dir / class_name

        for path in class_dir.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            try:
                image = cv2.imread(str(path))

                if image is None:
                    raise ValueError("Invalid image")

                image = cv2.cvtColor(
                    image, cv2.COLOR_BGR2RGB
                )
                image = cv2.resize(image, IMAGE_SIZE)

                images.append(image)
                labels.append(class_name)

            except Exception as error:
                print(f"Skipping {path}: {error}")

    if not images:
        raise ValueError("No valid images found.")

    x = np.asarray(images)
    y = encoder.transform(labels)

    print("Valid images:", len(x))
    print("Classes:", class_names)

    if len(class_names) != 4:
        raise ValueError(
            f"Model expects 4 classes; found {len(class_names)}."
        )

    return x, y, class_names


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if not ZIP_PATH.exists():
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, str(ZIP_PATH), quiet=False)

    x, y, class_names = load_data(ZIP_PATH)

    x_train, x_val_test, y_train, y_val_test = (
        train_test_split(
            x,
            y,
            test_size=0.30,
            random_state=42,
            stratify=y,
        )
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_val_test,
        y_val_test,
        test_size=0.50,
        random_state=42,
        stratify=y_val_test,
    )

    np.savez_compressed(
        ARTIFACTS_DIR / "dataset.npz",
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )

    np.save(
        ARTIFACTS_DIR / "class_names.npy",
        np.asarray(class_names),
    )

    print("Train:", len(x_train))
    print("Validation:", len(x_val))
    print("Test:", len(x_test))
    print("Dataset saved successfully.")


if __name__ == "__main__":
    main()
