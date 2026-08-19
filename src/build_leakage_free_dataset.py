"""Create a reproducible, leakage-free face-recognition dataset.

The original images are split *before* augmentation.  Only the training split
is augmented, so no transformed copy of a validation/test image can enter the
training data.  Existing ``dataset`` and ``dataset_augmented`` folders are
never modified.

Usage:
    python build_leakage_free_dataset.py
    python build_leakage_free_dataset.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "raw"
DESTINATION = ROOT / "data" / "leakage_free"
LABELS = ("Rahul", "Harshit", "Sohail", "Jagadeesh")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGE_SIZE = (96, 96)
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-train-images",
        type=int,
        default=300,
        help="Total training images per identity after augmentation (default: 300).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the generated data_v2 folder. Source images are never deleted.",
    )
    return parser.parse_args()


def image_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def load_image(source: Path) -> Image.Image | None:
    try:
        with Image.open(source) as image:
            return image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    except (OSError, ValueError):
        return None


def augment(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Use mild, attendance-camera-like transformations only."""
    angle = float(rng.uniform(-12, 12))
    transformed = image.rotate(angle, resample=Image.Resampling.BICUBIC)
    if rng.random() < 0.5:
        transformed = transformed.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    transformed = ImageEnhance.Brightness(transformed).enhance(float(rng.uniform(0.85, 1.15)))
    transformed = ImageEnhance.Contrast(transformed).enhance(float(rng.uniform(0.9, 1.1)))
    if rng.random() < 0.25:
        transformed = transformed.filter(ImageFilter.GaussianBlur(radius=0.7))
    return transformed


def split_originals(files: list[Path], rng: np.random.Generator) -> tuple[list[Path], list[Path], list[Path]]:
    """Deterministic 70/15/15 split without a third-party ML dependency."""
    indices = rng.permutation(len(files))
    train_end = max(1, round(len(files) * TRAIN_RATIO))
    validation_end = min(len(files) - 1, train_end + max(1, round(len(files) * VALIDATION_RATIO)))
    train = [files[index] for index in indices[:train_end]]
    validation = [files[index] for index in indices[train_end:validation_end]]
    test = [files[index] for index in indices[validation_end:]]
    return sorted(train), sorted(validation), sorted(test)


def replace_generated_folder(destination: Path, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"{destination} already exists. Re-run with --overwrite to replace only generated leakage_free files."
        )
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)


def main() -> None:
    args = parse_args()
    if not SOURCE.exists():
        raise FileNotFoundError(f"Source dataset is missing: {SOURCE}")
    if args.target_train_images < 1:
        raise ValueError("--target-train-images must be positive")

    replace_generated_folder(DESTINATION, args.overwrite)
    rng = np.random.default_rng(SEED)
    manifest_rows: list[dict[str, str]] = []

    for label in LABELS:
        originals = image_files(SOURCE / label)
        if len(originals) < 8:
            raise ValueError(f"{label} needs at least 8 original images; found {len(originals)}")

        train, validation, test = split_originals(originals, rng)
        split_files = {"train": sorted(train), "validation": sorted(validation), "test": sorted(test)}
        train_faces: list[tuple[Path, np.ndarray]] = []

        for split, files in split_files.items():
            split_dir = DESTINATION / "original_splits" / split / label
            split_dir.mkdir(parents=True, exist_ok=True)
            for index, source in enumerate(files):
                face = load_image(source)
                if face is None:
                    manifest_rows.append({"label": label, "split": split, "source": source.name, "output": "", "status": "skipped_invalid_image"})
                    continue
                output = split_dir / f"{index:03d}_{source.stem}.jpg"
                face.save(output, quality=95)
                manifest_rows.append({"label": label, "split": split, "source": source.name, "output": str(output.relative_to(DESTINATION)), "status": "original"})
                if split == "train":
                    train_faces.append((source, face))

        if not train_faces:
            raise RuntimeError(f"No detectable training faces for {label}.")

        augmented_dir = DESTINATION / "training_augmented" / label
        augmented_dir.mkdir(parents=True, exist_ok=True)
        for index, (source, face) in enumerate(train_faces):
            output = augmented_dir / f"orig_{index:03d}_{source.stem}.jpg"
            face.save(output, quality=95)
            manifest_rows.append({"label": label, "split": "train", "source": source.name, "output": str(output.relative_to(DESTINATION)), "status": "training_original"})

        count = len(train_faces)
        while count < args.target_train_images:
            source, base = train_faces[count % len(train_faces)]
            output = augmented_dir / f"aug_{count:04d}.jpg"
            augment(base, rng).save(output, quality=95)
            manifest_rows.append({"label": label, "split": "train", "source": source.name, "output": str(output.relative_to(DESTINATION)), "status": "augmented_training_only"})
            count += 1

        print(f"{label}: originals train/val/test = {len(train)}/{len(validation)}/{len(test)}; train images after augmentation = {count}")

    with (DESTINATION / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["label", "split", "source", "output", "status"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nDone. Leakage-free dataset written to: {DESTINATION}")
    print("Validation and test folders contain only unaugmented faces from held-out original images.")


if __name__ == "__main__":
    main()
