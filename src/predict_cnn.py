"""Run the trained CNN on one face image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "cnn" / "best_model.keras"
LABELS_PATH = ROOT / "models" / "cnn" / "labels.json"
IMAGE_SIZE = (96, 96)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Optional provisional rejection threshold. Calibrate it before production use.",
    )
    args = parser.parse_args()
    if not 0 <= args.min_confidence <= 1:
        raise ValueError("--min-confidence must be between 0 and 1")
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise FileNotFoundError("Train the CNN first with: python src/train_cnn.py")

    model = tf.keras.models.load_model(MODEL_PATH)
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    with Image.open(args.image) as image:
        resized = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    batch = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
    probabilities = model.predict(batch, verbose=0)[0]
    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index])
    identity = labels[str(index)] if confidence >= args.min_confidence else "Unknown"
    print(json.dumps({"identity": identity, "confidence": confidence}, indent=2))


if __name__ == "__main__":
    main()
