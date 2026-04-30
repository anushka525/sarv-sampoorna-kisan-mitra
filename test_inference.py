import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras


def load_labels(labels_path: Path) -> list[str] | None:
    if not labels_path.exists():
        return None

    with labels_path.open("r", encoding="utf-8") as f:
        labels = json.load(f)

    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        raise ValueError("labels.json must contain a JSON array of strings")

    return labels


def preprocess_image(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single-image inference test")
    parser.add_argument("--model", default="plant_disease_detector_model.keras")
    parser.add_argument("--image", default="abr.jpeg")
    parser.add_argument("--labels", default="labels.json")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    model_path = Path(args.model)
    image_path = Path(args.image)
    labels_path = Path(args.labels)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = keras.models.load_model(model_path)
    labels = load_labels(labels_path)

    x = preprocess_image(image_path)
    probs = model.predict(x, verbose=0)[0]
    indices = np.argsort(probs)[::-1]
    top_k = max(1, min(args.top_k, int(probs.shape[0])))

    print(f"Model: {model_path}")
    print(f"Image: {image_path}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output classes: {probs.shape[0]}")
    print(f"Probability sum: {float(np.sum(probs)):.6f}")
    print("Top predictions:")

    for rank, index in enumerate(indices[:top_k], start=1):
        class_index = int(index)
        label = labels[class_index] if labels and class_index < len(labels) else f"class_{class_index}"
        confidence = float(probs[class_index])
        print(f"{rank}. idx={class_index} label={label} confidence={confidence:.6f}")


if __name__ == "__main__":
    main()
