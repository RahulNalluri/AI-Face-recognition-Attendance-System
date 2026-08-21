"""Train and evaluate a leakage-free CNN face classifier.

Training uses only ``data/leakage_free/training_augmented``. Validation and
test images are held-out originals. The test split is evaluated only after the
best validation checkpoint has been restored.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "leakage_free"
TRAIN_DIR = DATA_ROOT / "training_augmented"
VALIDATION_DIR = DATA_ROOT / "original_splits" / "validation"
TEST_DIR = DATA_ROOT / "original_splits" / "test"
MANIFEST = DATA_ROOT / "manifest.csv"
MODEL_DIR = ROOT / "models" / "cnn"
EVALUATION_DIR = ROOT / "artifacts" / "evaluation" / "cnn"
IMAGE_SIZE = (96, 96)
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def require_dataset() -> None:
    required = (TRAIN_DIR, VALIDATION_DIR, TEST_DIR)
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing leakage-free dataset folders:\n" + "\n".join(missing))


def validate_manifest() -> None:
    """Fail if one original filename was assigned to multiple data splits."""
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Dataset manifest is missing: {MANIFEST}")
    assignments: dict[tuple[str, str], set[str]] = {}
    with MANIFEST.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if not row["source"] or row["status"].startswith("skipped"):
                continue
            key = (row["label"], row["source"])
            assignments.setdefault(key, set()).add(row["split"])
    leaked = {key: splits for key, splits in assignments.items() if len(splits) > 1}
    if leaked:
        examples = list(leaked.items())[:5]
        raise RuntimeError(f"Cross-split source leakage detected: {examples}")


def dataset_from(directory: Path, batch_size: int, shuffle: bool, seed: int, class_names=None):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=IMAGE_SIZE,
        shuffle=shuffle,
        seed=seed if shuffle else None,
    )


def build_model(class_count: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="face_image")
    x = tf.keras.layers.Rescaling(1.0 / 255, name="normalize")(inputs)
    # Mild online variation is applied only while training.
    x = tf.keras.layers.RandomContrast(0.08, seed=11, name="random_contrast")(x)

    for filters, dropout in ((32, 0.10), (64, 0.15), (128, 0.20)):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="face_features")(x)
    x = tf.keras.layers.Dropout(0.40)(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax", name="identity")(x)
    model = tf.keras.Model(inputs, outputs, name="face_attendance_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_count: int) -> np.ndarray:
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    for actual, predicted in zip(y_true, y_pred):
        matrix[int(actual), int(predicted)] += 1
    return matrix


def classification_metrics(matrix: np.ndarray, names: list[str]) -> dict:
    rows = []
    for index, name in enumerate(names):
        true_positive = float(matrix[index, index])
        false_positive = float(matrix[:, index].sum() - true_positive)
        false_negative = float(matrix[index, :].sum() - true_positive)
        support = int(matrix[index, :].sum())
        precision = true_positive / max(true_positive + false_positive, 1.0)
        recall = true_positive / max(true_positive + false_negative, 1.0)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append({
            "class": name,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support,
        })
    return {
        "per_class": rows,
        "macro_precision": float(np.mean([row["precision"] for row in rows])),
        "macro_recall": float(np.mean([row["recall"] for row in rows])),
        "macro_f1": float(np.mean([row["f1_score"] for row in rows])),
    }


def save_history(history: tf.keras.callbacks.History) -> None:
    values = history.history
    epochs = range(1, len(values["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, values["accuracy"], label="Training")
    axes[0].plot(epochs, values["val_accuracy"], label="Validation")
    axes[0].set(title="CNN Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, values["loss"], label="Training")
    axes[1].plot(epochs, values["val_loss"], label="Validation")
    axes[1].set(title="CNN Loss", xlabel="Epoch", ylabel="Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(EVALUATION_DIR / "training_history.png", dpi=180)
    plt.close(figure)
    with (EVALUATION_DIR / "training_history.json").open("w", encoding="utf-8") as file:
        json.dump({key: [float(value) for value in series] for key, series in values.items()}, file, indent=2)


def save_confusion(matrix: np.ndarray, names: list[str]) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        title="CNN Test Confusion Matrix",
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
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center",
                      color="white" if matrix[row, column] > threshold else "black")
    figure.tight_layout()
    figure.savefig(EVALUATION_DIR / "confusion_matrix.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    require_dataset()
    validate_manifest()
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    train = dataset_from(TRAIN_DIR, args.batch_size, True, args.seed)
    class_names = list(train.class_names)
    validation = dataset_from(VALIDATION_DIR, args.batch_size, False, args.seed, class_names)
    test = dataset_from(TEST_DIR, args.batch_size, False, args.seed, class_names)
    autotune = tf.data.AUTOTUNE
    train = train.prefetch(autotune)
    validation = validation.cache().prefetch(autotune)
    test = test.cache().prefetch(autotune)

    model = build_model(len(class_names))
    best_model = MODEL_DIR / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(best_model, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.4, patience=3, min_lr=1e-6),
        tf.keras.callbacks.CSVLogger(EVALUATION_DIR / "epoch_log.csv"),
    ]
    history = model.fit(train, validation_data=validation, epochs=args.epochs, callbacks=callbacks)
    save_history(history)

    model = tf.keras.models.load_model(best_model)
    test_loss, test_accuracy = model.evaluate(test, verbose=0)
    probabilities = model.predict(test, verbose=0)
    predicted = probabilities.argmax(axis=1)
    actual = np.concatenate([labels.numpy() for _, labels in test], axis=0)
    matrix = confusion_matrix(actual, predicted, len(class_names))
    metrics = classification_metrics(matrix, class_names)
    metrics.update({
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "test_samples": int(len(actual)),
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "class_names": class_names,
        "confusion_matrix": matrix.tolist(),
    })

    save_confusion(matrix, class_names)
    with (EVALUATION_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    with (MODEL_DIR / "labels.json").open("w", encoding="utf-8") as file:
        json.dump({str(index): name for index, name in enumerate(class_names)}, file, indent=2)
    with (MODEL_DIR / "training_config.json").open("w", encoding="utf-8") as file:
        json.dump({
            "image_size": list(IMAGE_SIZE),
            "batch_size": args.batch_size,
            "maximum_epochs": args.epochs,
            "seed": args.seed,
            "training_data": str(TRAIN_DIR.relative_to(ROOT)),
            "validation_data": str(VALIDATION_DIR.relative_to(ROOT)),
            "test_data": str(TEST_DIR.relative_to(ROOT)),
        }, file, indent=2)

    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Best model: {best_model}")
    print(f"Evaluation: {EVALUATION_DIR}")


if __name__ == "__main__":
    main()
