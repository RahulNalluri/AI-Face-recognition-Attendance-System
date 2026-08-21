"""Create a few-shot face classifier with OpenCV's pretrained SFace network.

YuNet supplies five facial landmarks for alignment. SFace converts each aligned
face to a 128-dimensional neural embedding. A small cosine classifier is chosen
using validation images only and is then evaluated once on the held-out test
split. The downloaded public ONNX weights and all private embeddings remain in
the ignored ``models/`` directory.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts" / ".matplotlib"))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps


RAW_ROOT = ROOT / "data" / "raw"
SPLIT_MANIFEST = ROOT / "data" / "leakage_free" / "manifest.csv"
CROPPED_ROOT = ROOT / "data" / "face_cropped" / "original_splits"
YUNET_MODEL = ROOT / "models" / "pretrained" / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = ROOT / "models" / "pretrained" / "face_recognition_sface_2021dec.onnx"
MODEL_DIR = ROOT / "models" / "sface"
EVALUATION_DIR = ROOT / "artifacts" / "evaluation" / "sface"


def assignments() -> list[dict[str, str]]:
    rows = []
    with SPLIT_MANIFEST.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["status"] == "original":
                rows.append(
                    {"label": row["label"], "split": row["split"], "source": row["source"]}
                )
    return sorted(rows, key=lambda row: (row["split"], row["label"], row["source"]))


def load_bgr(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        rgb = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def extract_feature(
    image: np.ndarray,
    detector: cv2.FaceDetectorYN,
    recognizer: cv2.FaceRecognizerSF,
) -> tuple[np.ndarray | None, float | None]:
    scale = min(1.0, 1000.0 / max(image.shape[:2]))
    working = (
        cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1
        else image
    )
    detector.setInputSize((working.shape[1], working.shape[0]))
    _, faces = detector.detect(working)
    if faces is None or len(faces) == 0:
        return None, None
    face = max(faces, key=lambda candidate: candidate[2] * candidate[3])
    aligned = recognizer.alignCrop(working, face)
    feature = recognizer.feature(aligned).reshape(-1).astype(np.float32)
    feature /= max(float(np.linalg.norm(feature)), 1e-12)
    return feature, float(face[-1])


def fallback_feature(
    path: Path, recognizer: cv2.FaceRecognizerSF
) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Fallback crop is missing: {path}")
    resized = cv2.resize(image, (112, 112), interpolation=cv2.INTER_AREA)
    feature = recognizer.feature(resized).reshape(-1).astype(np.float32)
    feature /= max(float(np.linalg.norm(feature)), 1e-12)
    return feature


def confusion(actual: np.ndarray, predicted: np.ndarray, count: int) -> np.ndarray:
    matrix = np.zeros((count, count), dtype=np.int64)
    for truth, prediction in zip(actual, predicted):
        matrix[int(truth), int(prediction)] += 1
    return matrix


def metrics(matrix: np.ndarray, names: list[str]) -> dict:
    per_class = []
    for index, name in enumerate(names):
        true_positive = float(matrix[index, index])
        false_positive = float(matrix[:, index].sum() - true_positive)
        false_negative = float(matrix[index].sum() - true_positive)
        precision = true_positive / max(true_positive + false_positive, 1.0)
        recall = true_positive / max(true_positive + false_negative, 1.0)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class.append(
            {
                "class": name,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": int(matrix[index].sum()),
            }
        )
    return {
        "accuracy": float(np.trace(matrix) / max(matrix.sum(), 1)),
        "macro_f1": float(np.mean([row["f1_score"] for row in per_class])),
        "per_class": per_class,
    }


def predict(
    query: np.ndarray,
    train: np.ndarray,
    train_labels: np.ndarray,
    method: str,
    class_count: int,
) -> np.ndarray:
    if method == "centroid":
        centroids = np.stack(
            [train[train_labels == index].mean(axis=0) for index in range(class_count)]
        )
        centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
        return (query @ centroids.T).argmax(axis=1)
    neighbours = int(method.removeprefix("knn_"))
    similarities = query @ train.T
    indices = np.argpartition(-similarities, neighbours - 1, axis=1)[:, :neighbours]
    predictions = []
    for row, nearest in zip(similarities, indices):
        scores = np.zeros(class_count, dtype=np.float64)
        for index in nearest:
            scores[train_labels[index]] += max(float(row[index]), 0.0) ** 4
        predictions.append(int(scores.argmax()))
    return np.asarray(predictions)


def save_confusion(matrix: np.ndarray, names: list[str]) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    visual = axis.imshow(matrix, cmap="Purples")
    figure.colorbar(visual, ax=axis)
    axis.set(
        title="SFace Test Confusion Matrix",
        xlabel="Predicted identity",
        ylabel="Actual identity",
        xticks=range(len(names)),
        yticks=range(len(names)),
        xticklabels=names,
        yticklabels=names,
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(len(names)):
        for column in range(len(names)):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(EVALUATION_DIR / "confusion_matrix.png", dpi=180)
    plt.close(figure)


def main() -> None:
    for required in (SPLIT_MANIFEST, YUNET_MODEL, SFACE_MODEL):
        if not required.exists():
            raise FileNotFoundError(f"Required file is missing: {required}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    detector = cv2.FaceDetectorYN.create(
        str(YUNET_MODEL), "", (320, 320), score_threshold=0.55, nms_threshold=0.3, top_k=5000
    )
    recognizer = cv2.FaceRecognizerSF.create(str(SFACE_MODEL), "")
    rows = assignments()
    names = sorted({row["label"] for row in rows})
    label_to_index = {name: index for index, name in enumerate(names)}
    features: dict[str, list[np.ndarray]] = {split: [] for split in ("train", "validation", "test")}
    labels: dict[str, list[int]] = {split: [] for split in features}
    extraction_log = []
    extraction_counts: Counter[str] = Counter()

    for row in rows:
        raw_path = RAW_ROOT / row["label"] / row["source"]
        feature, confidence = extract_feature(load_bgr(raw_path), detector, recognizer)
        status = "yunet_aligned"
        if feature is None:
            fallback = CROPPED_ROOT / row["split"] / row["label"] / f"{Path(row['source']).stem}.jpg"
            feature = fallback_feature(fallback, recognizer)
            status = "haar_crop_fallback"
        features[row["split"]].append(feature)
        labels[row["split"]].append(label_to_index[row["label"]])
        extraction_counts[status] += 1
        extraction_log.append({**row, "status": status, "detection_confidence": confidence})

    arrays = {split: np.stack(values) for split, values in features.items()}
    targets = {split: np.asarray(values, dtype=np.int64) for split, values in labels.items()}
    candidates = ["centroid", "knn_1", "knn_3", "knn_5", "knn_7"]
    validation_results = {}
    for method in candidates:
        predicted = predict(
            arrays["validation"], arrays["train"], targets["train"], method, len(names)
        )
        result = metrics(confusion(targets["validation"], predicted, len(names)), names)
        validation_results[method] = {
            "accuracy": result["accuracy"], "macro_f1": result["macro_f1"]
        }
    selected = max(
        candidates,
        key=lambda method: (
            validation_results[method]["accuracy"],
            validation_results[method]["macro_f1"],
            -candidates.index(method),
        ),
    )
    test_predicted = predict(
        arrays["test"], arrays["train"], targets["train"], selected, len(names)
    )
    matrix = confusion(targets["test"], test_predicted, len(names))
    report = metrics(matrix, names)
    report.update(
        {
            "selected_classifier": selected,
            "selection_policy": "highest validation accuracy, then validation macro F1",
            "validation_candidates": validation_results,
            "test_samples": int(len(targets["test"])),
            "confusion_matrix": matrix.tolist(),
            "feature_extraction": dict(extraction_counts),
            "model": "OpenCV SFace 2021dec",
        }
    )
    np.savez_compressed(
        MODEL_DIR / "classifier.npz",
        train_features=arrays["train"],
        train_labels=targets["train"],
    )
    (MODEL_DIR / "labels.json").write_text(
        json.dumps({str(index): name for index, name in enumerate(names)}, indent=2),
        encoding="utf-8",
    )
    (MODEL_DIR / "classifier_config.json").write_text(
        json.dumps({"method": selected, "class_count": len(names)}, indent=2),
        encoding="utf-8",
    )
    (EVALUATION_DIR / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (EVALUATION_DIR / "extraction_log.json").write_text(
        json.dumps(extraction_log, indent=2), encoding="utf-8"
    )
    save_confusion(matrix, names)
    print(f"Extraction: {dict(extraction_counts)}")
    print(f"Validation candidates: {validation_results}")
    print(f"Selected classifier: {selected}")
    print(f"Test accuracy: {report['accuracy'] * 100:.2f}%")
    print(f"Macro F1: {report['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
