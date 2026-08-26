"""Validate the deployed SFace recognizer with a locked unknown threshold."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from recognition_validation import DEFAULT_GATES, summarize_records
from train_sface_classifier import (
    RAW_ROOT,
    SFACE_MODEL,
    YUNET_MODEL,
    assignments,
    extract_feature,
    load_bgr,
    predict,
)


ROOT = Path(__file__).resolve().parent.parent
CLASSIFIER = ROOT / "models" / "sface" / "classifier.npz"
LABELS = ROOT / "models" / "sface" / "labels.json"
CONFIG = ROOT / "models" / "sface" / "classifier_config.json"
THRESHOLD = ROOT / "models" / "sface" / "unknown_threshold.json"
DEFAULT_UNKNOWN_DIR = ROOT / "data" / "recognition_validation" / "unknown"
DEFAULT_OUTPUT = ROOT / "artifacts" / "evaluation" / "recognition_validation" / "report.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--enrolled-dir", type=Path,
        help="Optional folder with one subfolder per enrolled identity; defaults to held-out originals.",
    )
    parser.add_argument("--unknown-dir", type=Path, default=DEFAULT_UNKNOWN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-unknown", action="store_true")
    parser.add_argument("--min-known-detection", type=float, default=DEFAULT_GATES["known_detection_rate"])
    parser.add_argument("--min-known-identification", type=float, default=DEFAULT_GATES["known_identification_rate"])
    parser.add_argument("--min-unknown-detection", type=float, default=DEFAULT_GATES["unknown_detection_rate"])
    parser.add_argument("--min-unknown-rejection", type=float, default=DEFAULT_GATES["unknown_rejection_rate"])
    return parser.parse_args()


def images_under(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)


def enrolled_cases(directory: Path | None) -> list[tuple[str, Path]]:
    if directory is not None:
        return [(path.parent.name, path) for path in images_under(directory)]
    return [
        (row["label"], RAW_ROOT / row["label"] / row["source"])
        for row in assignments() if row["split"] == "test"
    ]


def main() -> None:
    args = parse_args()
    gates = {
        "known_detection_rate": args.min_known_detection,
        "known_identification_rate": args.min_known_identification,
        "unknown_detection_rate": args.min_unknown_detection,
        "unknown_rejection_rate": args.min_unknown_rejection,
    }
    if any(not 0 <= value <= 1 for value in gates.values()):
        raise ValueError("All validation gates must be between 0 and 1")
    for required in (YUNET_MODEL, SFACE_MODEL, CLASSIFIER, LABELS, CONFIG, THRESHOLD):
        if not required.exists():
            raise FileNotFoundError(f"Required file is missing: {required}")

    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    enrolled_labels = [labels[str(index)] for index in range(len(labels))]
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    threshold = float(json.loads(THRESHOLD.read_text(encoding="utf-8"))["threshold"])
    stored = np.load(CLASSIFIER)
    train_features, train_labels = stored["train_features"], stored["train_labels"]
    detector = cv2.FaceDetectorYN.create(
        str(YUNET_MODEL), "", (320, 320), score_threshold=0.55,
        nms_threshold=0.3, top_k=5000,
    )
    recognizer = cv2.FaceRecognizerSF.create(str(SFACE_MODEL), "")
    cases = enrolled_cases(args.enrolled_dir) + [
        ("Unknown", path) for path in images_under(args.unknown_dir)
    ]
    unknown_labels = sorted({label for label, _ in cases if label not in {*enrolled_labels, "Unknown"}})
    if unknown_labels:
        raise ValueError(f"Validation folders do not match enrolled labels: {', '.join(unknown_labels)}")

    records = []
    for expected, path in cases:
        feature, detection_confidence = extract_feature(load_bgr(path), detector, recognizer)
        if feature is None:
            records.append({
                "path": str(path), "expected": expected, "predicted": "No face",
                "detected": False, "similarity": None, "detection_confidence": None,
            })
            continue
        predicted_index = int(predict(
            feature[None, :], train_features, train_labels,
            config["method"], config["class_count"],
        )[0])
        similarity = float((feature @ train_features[train_labels == predicted_index].T).max())
        predicted = labels[str(predicted_index)] if similarity >= threshold else "Unknown"
        records.append({
            "path": str(path), "expected": expected, "predicted": predicted,
            "detected": True, "similarity": round(similarity, 6),
            "detection_confidence": (
                None if detection_confidence is None else round(float(detection_confidence), 6)
            ),
        })

    report = summarize_records(records, enrolled_labels, gates)
    report.update({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "OpenCV SFace 2021dec",
        "classifier": config["method"],
        "locked_unknown_threshold": threshold,
        "enrolled_source": str(args.enrolled_dir or "held-out test originals"),
        "unknown_source": str(args.unknown_dir),
        "records": records,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "path", "expected", "predicted", "detected", "similarity", "detection_confidence",
        ])
        writer.writeheader()
        writer.writerows(records)

    known = report["known"]
    unknown = report["unknown"]
    print(f"Recognition validation status: {report['status'].upper()}")
    print(f"Known detection: {(known['detection_rate'] or 0) * 100:.2f}%")
    print(f"Known identification: {(known['identification_rate'] or 0) * 100:.2f}%")
    print(
        "Unknown rejection: not measured (add consented unknown faces)"
        if unknown["samples"] == 0 else
        f"Unknown rejection: {(unknown['rejection_rate'] or 0) * 100:.2f}%"
    )
    for notice in report["notices"]:
        print(f"- {notice}")
    print(f"Report: {args.output}")
    if args.require_unknown and report["status"] == "incomplete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
