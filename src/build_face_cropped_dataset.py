"""Build a face-focused dataset while preserving the leakage-free splits.

The existing ``data/leakage_free/manifest.csv`` remains the source of truth for
train/validation/test membership. Each original photograph is EXIF-corrected,
searched for a face, padded to retain useful context, and resized. No augmented
image is used for validation or testing.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = ROOT / "data" / "raw"
SOURCE_MANIFEST = ROOT / "data" / "leakage_free" / "manifest.csv"
DESTINATION = ROOT / "data" / "face_cropped"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_SIZE = 160


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--padding", type=float, default=0.28)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_assignments() -> list[dict[str, str]]:
    if not SOURCE_MANIFEST.exists():
        raise FileNotFoundError(
            "Build data/leakage_free first with src/build_leakage_free_dataset.py"
        )
    assignments: dict[tuple[str, str], str] = {}
    with SOURCE_MANIFEST.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["status"] != "original":
                continue
            key = (row["label"], row["source"])
            previous = assignments.setdefault(key, row["split"])
            if previous != row["split"]:
                raise RuntimeError(f"Cross-split source leakage found for {key}")
    return [
        {"label": label, "source": source, "split": split}
        for (label, source), split in sorted(assignments.items())
    ]


def rotate(image: np.ndarray, angle: float) -> np.ndarray:
    if angle == 0:
        return image
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def detect_largest_face(
    image: np.ndarray, detector: cv2.CascadeClassifier
) -> tuple[np.ndarray, tuple[int, int, int, int], float] | None:
    """Return the rotated image and strongest face candidate."""
    candidates: list[tuple[float, np.ndarray, tuple[int, int, int, int], float]] = []
    for angle in (0.0, -10.0, 10.0):
        rotated = rotate(image, angle)
        gray = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY)
        variants = (gray, cv2.createCLAHE(2.0, (8, 8)).apply(gray))
        minimum = max(36, min(gray.shape) // 14)
        for variant in variants:
            faces = detector.detectMultiScale(
                variant,
                scaleFactor=1.08,
                minNeighbors=4,
                minSize=(minimum, minimum),
            )
            for x, y, width, height in faces:
                center_x = x + width / 2
                center_y = y + height / 2
                image_center_x = rotated.shape[1] / 2
                image_center_y = rotated.shape[0] / 2
                center_distance = (
                    abs(center_x - image_center_x) / rotated.shape[1]
                    + abs(center_y - image_center_y) / rotated.shape[0]
                )
                area_ratio = (width * height) / (rotated.shape[0] * rotated.shape[1])
                score = area_ratio - 0.03 * center_distance
                candidates.append(
                    (score, rotated, (int(x), int(y), int(width), int(height)), angle)
                )
    if not candidates:
        return None
    _, detected_image, box, angle = max(candidates, key=lambda item: item[0])
    return detected_image, box, angle


def square_crop(
    image: np.ndarray, box: tuple[int, int, int, int], padding: float
) -> np.ndarray:
    x, y, width, height = box
    side = int(round(max(width, height) * (1 + 2 * padding)))
    center_x = x + width // 2
    center_y = y + height // 2
    left = max(0, center_x - side // 2)
    top = max(0, center_y - side // 2)
    right = min(image.shape[1], left + side)
    bottom = min(image.shape[0], top + side)
    left = max(0, right - side)
    top = max(0, bottom - side)
    return image[top:bottom, left:right]


def center_square(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    side = min(height, width)
    left = (width - side) // 2
    top = (height - side) // 2
    return image[top : top + side, left : left + side]


def process_image(
    source: Path,
    detector: cv2.CascadeClassifier,
    padding: float,
    image_size: int,
) -> tuple[Image.Image, str, str]:
    with Image.open(source) as opened:
        rgb = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))
    detection = detect_largest_face(rgb, detector)
    if detection is None:
        crop = center_square(rgb)
        status = "center_crop_fallback"
        rotation = ""
    else:
        detected_image, box, angle = detection
        crop = square_crop(detected_image, box, padding)
        status = "face_detected"
        rotation = f"{angle:g}"
    output = Image.fromarray(crop).resize(
        (image_size, image_size), Image.Resampling.LANCZOS
    )
    return output, status, rotation


def main() -> None:
    args = parse_args()
    if args.image_size < 64:
        raise ValueError("--image-size must be at least 64")
    if not 0 <= args.padding <= 1:
        raise ValueError("--padding must be between 0 and 1")
    if DESTINATION.exists() and not args.overwrite:
        raise FileExistsError(f"{DESTINATION} exists; use --overwrite to replace it")
    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)
    DESTINATION.mkdir(parents=True)

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if detector.empty():
        raise RuntimeError("OpenCV frontal-face cascade could not be loaded")

    rows: list[dict[str, str]] = []
    counts: Counter[tuple[str, str]] = Counter()
    detection_counts: Counter[str] = Counter()
    for assignment in load_assignments():
        label = assignment["label"]
        split = assignment["split"]
        source = RAW_ROOT / label / assignment["source"]
        if not source.exists() or source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise FileNotFoundError(f"Assigned original is missing: {source}")
        face, status, rotation = process_image(
            source, detector, args.padding, args.image_size
        )
        output_dir = DESTINATION / "original_splits" / split / label
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"{source.stem}.jpg"
        face.save(output, quality=95)
        counts[(label, split)] += 1
        detection_counts[status] += 1
        rows.append(
            {
                "label": label,
                "split": split,
                "source": source.name,
                "output": str(output.relative_to(DESTINATION)),
                "status": status,
                "rotation_degrees": rotation,
            }
        )

    with (DESTINATION / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "label",
                "split",
                "source",
                "output",
                "status",
                "rotation_degrees",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    labels = sorted({row["label"] for row in rows})
    for label in labels:
        print(
            f"{label}: train={counts[(label, 'train')]}, "
            f"validation={counts[(label, 'validation')]}, test={counts[(label, 'test')]}"
        )
    print(f"Detection results: {dict(detection_counts)}")
    print(f"Face-focused dataset: {DESTINATION}")


if __name__ == "__main__":
    main()
