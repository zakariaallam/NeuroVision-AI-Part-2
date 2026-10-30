
import os

import mlflow
import mlflow.tensorflow
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError



MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "https://andrea-innocent-officer-divisions.trycloudflare.com"
)

MODEL_NAME = "mobileNetV2_IMR"
MODEL_ALIAS = "champion"

IMG_SIZE = (224, 224)


CLASS_NAMES = [
    "glioma",
    "meningioma",
    "meningioma",
    "pituitary",
]



st.set_page_config(
    page_title="IRM Classification",
    page_icon="🧠",
    layout="centered"
)

st.title("IRM Classification")
st.write("Test du modèle champion enregistré dans MLflow.")

st.warning(
    "Prototype pédagogique uniquement. "
    "Ce résultat ne constitue pas un diagnostic médical."
)


@st.cache_resource
def load_champion():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

    return mlflow.tensorflow.load_model(model_uri)


def predict_image(image, model):
    try:
        image = image.convert("RGB")
        image = image.resize(IMG_SIZE)

        array = np.asarray(image, dtype=np.float32)
        array = np.expand_dims(array, axis=0)

        predictions = np.asarray(
            model.predict(array, verbose=0)
        )

        if predictions.ndim != 2 or predictions.shape[0] != 1:
            return {"error": "Format de sortie du modèle invalide."}

        scores = predictions[0]

        if len(scores) != len(CLASS_NAMES):
            return {
                "error": (
                    "Le nombre de classes ne correspond pas "
                    "à CLASS_NAMES."
                )
            }

        if not np.all(np.isfinite(scores)):
            return {"error": "Le modèle a retourné des scores invalides."}

        index = int(np.argmax(scores))

        return {
            "predicted_class": CLASS_NAMES[index],
            "confidence": float(scores[index]),
            "scores": {
                name: float(score)
                for name, score in zip(CLASS_NAMES, scores)
            }
        }

    except Exception as error:
        return {"error": f"Échec de la prédiction : {error}"}


uploaded_file = st.file_uploader(
    "Choisir une image IRM",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image.verify()

        uploaded_file.seek(0)
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Image chargée",
            width=400
        )

        if st.button("Prédire", type="primary"):
            try:
                with st.spinner("Chargement du champion depuis MLflow..."):
                    model = load_champion()

                with st.spinner("Analyse de l'image..."):
                    result = predict_image(image, model)

                if "error" in result:
                    st.error(result["error"])
                else:
                    st.subheader("Résultat")
                    st.write(
                        f"Classe prédite : **{result['predicted_class']}**"
                    )
                    st.metric(
                        "Score du modèle",
                        f"{result['confidence']:.2%}"
                    )

                    st.subheader("Scores par classe")
                    st.bar_chart(result["scores"])

            except Exception as error:
                st.error(
                    "Impossible de charger le modèle champion depuis "
                    f"MLflow. Vérifie l'accès au serveur et au Registry. "
                    f"Détail : {error}"
                )

    except (UnidentifiedImageError, OSError, ValueError):
        st.error("Le fichier est invalide ou ne peut pas être lu.")
