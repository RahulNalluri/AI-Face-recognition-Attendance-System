"""Calibrate SFace unknown-person rejection without using test labels for tuning.

For every validation face, the strongest same-identity similarity is a genuine
score. The strongest different-identity similarity simulates that identity not
being enrolled and is therefore an impostor score. A security-first threshold
is selected at the requested maximum observed false-accept rate, then evaluated
unchanged on the held-out test split.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts" / ".matplotlib"))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train_sface_classifier import (
    CROPPED_ROOT,
    SFACE_MODEL,
    YUNET_MODEL,
    assignments,
    extract_feature,
    fallback_feature,
    load_bgr,
)


CLASSIFIER = ROOT / "models" / "sface" / "classifier.npz"
LABELS = ROOT / "models" / "sface" / "labels.json"
THRESHOLD_FILE = ROOT / "models" / "sface" / "unknown_threshold.json"
EVALUATION_DIR = ROOT / "artifacts" / "evaluation" / "unknown_rejection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-far",
        type=float,
        default=0.0,
        help="Maximum observed validation false-accept rate (default: 0).",
    )
    return parser.parse_args()


def feature_for(
    row: dict[str, str],
    detector: cv2.FaceDetectorYN,
    recognizer: cv2.FaceRecognizerSF,
) -> np.ndarray:
    raw = ROOT / "data" / "raw" / row["label"] / row["source"]
    feature, _ = extract_feature(load_bgr(raw), detector, recognizer)
    if feature is not None:
        return feature
    fallback = (
        CROPPED_ROOT
        / row["split"]
        / row["label"]
        / f"{Path(row['source']).stem}.jpg"
    )
    return fallback_feature(fallback, recognizer)


def score_split(
    split: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    label_to_index: dict[str, int],
    detector: cv2.FaceDetectorYN,
    recognizer: cv2.FaceRecognizerSF,
) -> dict[str, np.ndarray]:
    genuine = []
    impostor = []
    predicted = []
    actual = []
    for row in assignments():
        if row["split"] != split:
            continue
        feature = feature_for(row, detector, recognizer)
        label = label_to_index[row["label"]]
        similarities = feature @ train_features.T
        genuine.append(float(similarities[train_labels == label].max()))
        impostor.append(float(similarities[train_labels != label].max()))
        nearest = int(np.argmax(similarities))
        predicted.append(int(train_labels[nearest]))
        actual.append(label)
    return {
        "genuine": np.asarray(genuine),
        "impostor": np.asarray(impostor),
        "predicted": np.asarray(predicted),
        "actual": np.asarray(actual),
    }


def rates(genuine: np.ndarray, impostor: np.ndarray, threshold: float) -> dict[str, float]:
    return {
        "false_accept_rate": float(np.mean(impostor >= threshold)),
        "false_reject_rate": float(np.mean(genuine < threshold)),
        "genuine_accept_rate": float(np.mean(genuine >= threshold)),
        "unknown_reject_rate": float(np.mean(impostor < threshold)),
    }


def select_threshold(
    genuine: np.ndarray, impostor: np.ndarray, target_far: float
) -> tuple[float, dict[str, float]]:
    values = np.unique(np.concatenate([genuine, impostor]))
    candidates = [float(values[0] - 1e-6), float(values[-1] + 1e-6)]
    candidates.extend(float((left + right) / 2) for left, right in zip(values[:-1], values[1:]))
    evaluated = [(threshold, rates(genuine, impostor, threshold)) for threshold in candidates]
    eligible = [item for item in evaluated if item[1]["false_accept_rate"] <= target_far]
    if not eligible:
        minimum_far = min(item[1]["false_accept_rate"] for item in evaluated)
        eligible = [item for item in evaluated if item[1]["false_accept_rate"] == minimum_far]
    minimum_frr = min(item[1]["false_reject_rate"] for item in eligible)
    best = [item for item in eligible if item[1]["false_reject_rate"] == minimum_frr]
    # Midpoint candidates already sit between adjacent observed scores. Prefer
    # the lowest tied threshold to avoid unnecessarily rejecting genuine users.
    return min(best, key=lambda item: item[0])


def summary(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(values.min()),
        "median": float(np.median(values)),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
    }


def roc_statistics(genuine: np.ndarray, impostor: np.ndarray) -> dict:
    values = sorted(set(np.concatenate([genuine, impostor]).tolist()), reverse=True)
    thresholds = [float("inf"), *values, float("-inf")]
    false_accept_rates = np.asarray([np.mean(impostor >= value) for value in thresholds])
    true_accept_rates = np.asarray([np.mean(genuine >= value) for value in thresholds])
    false_reject_rates = 1.0 - true_accept_rates
    eer_index = int(np.argmin(np.abs(false_accept_rates - false_reject_rates)))
    auc = float(np.trapezoid(true_accept_rates, false_accept_rates))
    return {
        "thresholds": thresholds,
        "false_accept_rates": false_accept_rates.tolist(),
        "true_accept_rates": true_accept_rates.tolist(),
        "auc": auc,
        "equal_error_rate": float(
            (false_accept_rates[eer_index] + false_reject_rates[eer_index]) / 2
        ),
        "equal_error_threshold": float(thresholds[eer_index]),
    }


def save_plot(genuine: np.ndarray, impostor: np.ndarray, threshold: float) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.8))
    bins = np.linspace(min(genuine.min(), impostor.min()) - 0.02, 1.0, 24)
    axis.hist(impostor, bins=bins, alpha=0.7, label="Simulated unknown", color="#c44e52")
    axis.hist(genuine, bins=bins, alpha=0.7, label="Genuine enrolled", color="#4c72b0")
    axis.axvline(threshold, color="black", linestyle="--", label=f"Threshold {threshold:.3f}")
    axis.set(
        title="Validation Similarity and Unknown-Rejection Threshold",
        xlabel="Cosine similarity",
        ylabel="Number of validation faces",
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(EVALUATION_DIR / "threshold_distribution.png", dpi=180)
    plt.close(figure)


def save_roc(roc: dict) -> None:
    figure, axis = plt.subplots(figsize=(5.8, 5.4))
    axis.plot(
        roc["false_accept_rates"],
        roc["true_accept_rates"],
        color="#4c72b0",
        linewidth=2,
        label=f"ROC (AUC={roc['auc']:.3f})",
    )
    axis.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Chance")
    axis.set(
        title="Validation Unknown-Rejection ROC",
        xlabel="False accept rate",
        ylabel="Genuine accept rate",
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(EVALUATION_DIR / "roc_curve.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not 0 <= args.target_far <= 1:
        raise ValueError("--target-far must be between 0 and 1")
    for required in (CLASSIFIER, LABELS, YUNET_MODEL, SFACE_MODEL):
        if not required.exists():
            raise FileNotFoundError(f"Required file is missing: {required}")
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    stored = np.load(CLASSIFIER)
    train_features = stored["train_features"]
    train_labels = stored["train_labels"]
    index_to_label = json.loads(LABELS.read_text(encoding="utf-8"))
    label_to_index = {label: int(index) for index, label in index_to_label.items()}
    detector = cv2.FaceDetectorYN.create(
        str(YUNET_MODEL), "", (320, 320), score_threshold=0.55, nms_threshold=0.3, top_k=5000
    )
    recognizer = cv2.FaceRecognizerSF.create(str(SFACE_MODEL), "")

    validation = score_split(
        "validation", train_features, train_labels, label_to_index, detector, recognizer
    )
    threshold, validation_rates = select_threshold(
        validation["genuine"], validation["impostor"], args.target_far
    )
    validation_roc = roc_statistics(validation["genuine"], validation["impostor"])
    test = score_split("test", train_features, train_labels, label_to_index, detector, recognizer)
    test_rates = rates(test["genuine"], test["impostor"], threshold)
    accepted = test["genuine"] >= threshold
    correct = test["predicted"] == test["actual"]
    test_rates["correct_known_accept_rate"] = float(np.mean(accepted & correct))
    test_rates["closed_set_identification_accuracy"] = float(np.mean(correct))

    configuration = {
        "threshold": threshold,
        "similarity": "cosine",
        "selection_split": "validation",
        "target_false_accept_rate": args.target_far,
        "policy": "minimize false rejects subject to target observed FAR",
        "validation_rates": validation_rates,
        "validation_auc": validation_roc["auc"],
        "validation_equal_error_rate": validation_roc["equal_error_rate"],
        "validation_equal_error_threshold": validation_roc["equal_error_threshold"],
        "warning": "Preliminary threshold; verify with genuinely unenrolled people before deployment.",
    }
    THRESHOLD_FILE.write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    report = {
        **configuration,
        "validation_samples": int(len(validation["genuine"])),
        "validation_genuine_scores": summary(validation["genuine"]),
        "validation_impostor_scores": summary(validation["impostor"]),
        "test_samples": int(len(test["genuine"])),
        "test_rates_at_locked_threshold": test_rates,
        "test_genuine_scores": summary(test["genuine"]),
        "test_impostor_scores": summary(test["impostor"]),
    }
    (EVALUATION_DIR / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    save_plot(validation["genuine"], validation["impostor"], threshold)
    save_roc(validation_roc)
    print(f"Calibrated cosine threshold: {threshold:.4f}")
    print(f"Validation FAR: {validation_rates['false_accept_rate'] * 100:.2f}%")
    print(f"Validation FRR: {validation_rates['false_reject_rate'] * 100:.2f}%")
    print(f"Test known accept + correct: {test_rates['correct_known_accept_rate'] * 100:.2f}%")
    print(f"Test simulated unknown rejection: {test_rates['unknown_reject_rate'] * 100:.2f}%")
    print(f"Threshold file: {THRESHOLD_FILE}")


if __name__ == "__main__":
    main()
