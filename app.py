import json
from io import BytesIO
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from tensorflow import keras

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "plant_disease_detector_model.keras"
LABELS_PATH = BASE_DIR / "labels.json"
IMAGE_SIZE = (224, 224)


def load_labels(labels_path: Path) -> list[str] | None:
    if not labels_path.exists():
        return None

    with labels_path.open("r", encoding="utf-8") as f:
        labels = json.load(f)

    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        raise ValueError("labels.json must contain a JSON array of strings")

    return labels


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = keras.models.load_model(MODEL_PATH)
labels = load_labels(LABELS_PATH)

app = Flask(__name__)
CORS(app)


@app.get("/")
def home():
        return """
        <html>
            <head>
                <title>Sarv Sampoorna Kisan Mitra API</title>
            </head>
            <body>
                <h1>Sarv Sampoorna Kisan Mitra API</h1>
                <p>The backend is running successfully.</p>
                <ul>
                    <li><a href="/health">Health check</a></li>
                    <li>POST /predict for image prediction</li>
                </ul>
            </body>
        </html>
        """


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": True,
            "input_shape": model.input_shape,
            "output_shape": model.output_shape,
            "labels_loaded": labels is not None,
            "num_classes": int(model.output_shape[-1]) if model.output_shape else None,
        }
    )


@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"error": "Uploaded file is empty"}), 400

    top_k = request.args.get("top_k", default=3, type=int)
    top_k = max(1, top_k)

    try:
        x = preprocess_image(image_bytes)
        probs = model.predict(x, verbose=0)[0]
        if probs.ndim != 1:
            return jsonify({"error": "Unexpected model output shape"}), 500

        top_k = min(top_k, int(probs.shape[0]))
        sorted_indices = np.argsort(probs)[::-1][:top_k]

        predictions = []
        for index in sorted_indices:
            class_index = int(index)
            label = labels[class_index] if labels and class_index < len(labels) else f"class_{class_index}"
            predictions.append(
                {
                    "index": class_index,
                    "label": label,
                    "confidence": float(probs[class_index]),
                }
            )

        return jsonify(
            {
                "num_classes": int(probs.shape[0]),
                "confidence_sum": float(np.sum(probs)),
                "top_prediction": predictions[0],
                "top_k": predictions,
            }
        )
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
