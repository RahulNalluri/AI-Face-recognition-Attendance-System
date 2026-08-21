"""Recognize one face using the locally enrolled SFace classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from train_sface_classifier import (
    SFACE_MODEL,
    YUNET_MODEL,
    extract_feature,
    load_bgr,
    predict,
)


ROOT = Path(__file__).resolve().parent.parent
CLASSIFIER = ROOT / "models" / "sface" / "classifier.npz"
LABELS = ROOT / "models" / "sface" / "labels.json"
CONFIG = ROOT / "models" / "sface" / "classifier_config.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.0,
        help="Optional unknown-person threshold. Calibrate with unenrolled faces before deployment.",
    )
    args = parser.parse_args()
    if not -1 <= args.min_similarity <= 1:
        raise ValueError("--min-similarity must be between -1 and 1")
    for required in (args.image, YUNET_MODEL, SFACE_MODEL, CLASSIFIER, LABELS, CONFIG):
        if not required.exists():
            raise FileNotFoundError(f"Required file is missing: {required}")

    detector = cv2.FaceDetectorYN.create(
        str(YUNET_MODEL), "", (320, 320), score_threshold=0.55, nms_threshold=0.3, top_k=5000
    )
    recognizer = cv2.FaceRecognizerSF.create(str(SFACE_MODEL), "")
    feature, detection_confidence = extract_feature(load_bgr(args.image), detector, recognizer)
    if feature is None:
        print(json.dumps({"identity": "No face detected", "confidence": 0.0}, indent=2))
        return

    stored = np.load(CLASSIFIER)
    train_features = stored["train_features"]
    train_labels = stored["train_labels"]
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    predicted = int(
        predict(
            feature[None, :],
            train_features,
            train_labels,
            config["method"],
            config["class_count"],
        )[0]
    )
    class_similarities = feature @ train_features[train_labels == predicted].T
    similarity = float(class_similarities.max())
    identity = labels[str(predicted)] if similarity >= args.min_similarity else "Unknown"
    print(
        json.dumps(
            {
                "identity": identity,
                "similarity": similarity,
                "face_detection_confidence": detection_confidence,
                "classifier": config["method"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
