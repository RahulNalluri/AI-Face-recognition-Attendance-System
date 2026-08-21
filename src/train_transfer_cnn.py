"""Train and evaluate a face-focused MobileNetV2 transfer-learning model."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


DATA_ROOT = ROOT / "data" / "face_cropped" / "original_splits"
TRAIN_DIR = DATA_ROOT / "train"
VALIDATION_DIR = DATA_ROOT / "validation"
TEST_DIR = DATA_ROOT / "test"
MODEL_DIR = ROOT / "models" / "transfer_cnn"
EVALUATION_DIR = ROOT / "artifacts" / "evaluation" / "transfer_cnn"
IMAGE_SIZE = (160, 160)
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-epochs", type=int, default=30)
    parser.add_argument("--fine-tune-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def dataset_from(
    directory: Path,
    batch_size: int,
    shuffle: bool,
    seed: int,
    class_names: list[str] | None = None,
) -> tf.data.Dataset:
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


def build_model(class_count: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=101),
            tf.keras.layers.RandomRotation(0.045, fill_mode="reflect", seed=102),
            tf.keras.layers.RandomTranslation(0.07, 0.07, fill_mode="reflect", seed=103),
            tf.keras.layers.RandomZoom((-0.10, 0.14), fill_mode="reflect", seed=104),
            tf.keras.layers.RandomContrast(0.15, seed=105),
            tf.keras.layers.RandomBrightness(0.12, value_range=(0, 255), seed=106),
            tf.keras.layers.GaussianNoise(0.8, seed=107),
        ],
        name="attendance_camera_augmentation",
    )
    inputs = tf.keras.Input((*IMAGE_SIZE, 3), name="face_image")
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
        alpha=1.0,
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="face_embedding")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="identity_features",
    )(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(
        class_count, activation="softmax", name="identity"
    )(x)
    return tf.keras.Model(inputs, outputs, name="face_mobilenet_v2"), backbone


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )


def callbacks(checkpoint: Path, log: Path, patience: int) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint, monitor="val_loss", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=4, min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(log),
    ]


def class_weights(train_directory: Path, names: list[str]) -> dict[int, float]:
    counts = np.asarray(
        [sum(path.is_file() for path in (train_directory / name).iterdir()) for name in names],
        dtype=np.float64,
    )
    total = float(counts.sum())
    return {index: total / (len(names) * count) for index, count in enumerate(counts)}


def merge_histories(
    head: tf.keras.callbacks.History, fine: tf.keras.callbacks.History
) -> dict[str, list[float]]:
    keys = set(head.history) | set(fine.history)
    return {
        key: [float(value) for value in head.history.get(key, [])]
        + [float(value) for value in fine.history.get(key, [])]
        for key in keys
    }


def predict_probabilities(model: tf.keras.Model, dataset: tf.data.Dataset, flip: bool) -> np.ndarray:
    batches = []
    for images, _ in dataset:
        normal = model(images, training=False).numpy()
        if flip:
            mirrored = model(tf.image.flip_left_right(images), training=False).numpy()
            normal = (normal + mirrored) / 2.0
        batches.append(normal)
    return np.concatenate(batches, axis=0)


def actual_labels(dataset: tf.data.Dataset) -> np.ndarray:
    return np.concatenate([labels.numpy() for _, labels in dataset], axis=0)


def confusion_matrix(actual: np.ndarray, predicted: np.ndarray, count: int) -> np.ndarray:
    matrix = np.zeros((count, count), dtype=np.int64)
    for truth, prediction in zip(actual, predicted):
        matrix[int(truth), int(prediction)] += 1
    return matrix


def classification_metrics(matrix: np.ndarray, names: list[str]) -> dict:
    rows = []
    for index, name in enumerate(names):
        true_positive = float(matrix[index, index])
        false_positive = float(matrix[:, index].sum() - true_positive)
        false_negative = float(matrix[index, :].sum() - true_positive)
        precision = true_positive / max(true_positive + false_positive, 1.0)
        recall = true_positive / max(true_positive + false_negative, 1.0)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append(
            {
                "class": name,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": int(matrix[index].sum()),
            }
        )
    return {
        "per_class": rows,
        "macro_precision": float(np.mean([row["precision"] for row in rows])),
        "macro_recall": float(np.mean([row["recall"] for row in rows])),
        "macro_f1": float(np.mean([row["f1_score"] for row in rows])),
    }


def save_history(values: dict[str, list[float]], fine_tune_start: int) -> None:
    epochs = np.arange(1, len(values["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, values["accuracy"], label="Training")
    axes[0].plot(epochs, values["val_accuracy"], label="Validation")
    axes[0].set(title="Transfer CNN Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].plot(epochs, values["loss"], label="Training")
    axes[1].plot(epochs, values["val_loss"], label="Validation")
    axes[1].set(title="Transfer CNN Loss", xlabel="Epoch", ylabel="Loss")
    for axis in axes:
        axis.axvline(fine_tune_start + 0.5, color="gray", linestyle="--", label="Fine-tuning")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(EVALUATION_DIR / "training_history.png", dpi=180)
    plt.close(figure)
    (EVALUATION_DIR / "training_history.json").write_text(
        json.dumps(values, indent=2), encoding="utf-8"
    )


def save_confusion(matrix: np.ndarray, names: list[str]) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(matrix, cmap="Greens")
    figure.colorbar(image, ax=axis)
    axis.set(
        title="Transfer CNN Test Confusion Matrix",
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
    args = parse_args()
    for directory in (TRAIN_DIR, VALIDATION_DIR, TEST_DIR):
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Missing {directory}; run src/build_face_cropped_dataset.py first"
            )
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    train = dataset_from(TRAIN_DIR, args.batch_size, True, args.seed)
    names = list(train.class_names)
    validation = dataset_from(
        VALIDATION_DIR, args.batch_size, False, args.seed, names
    ).cache().prefetch(tf.data.AUTOTUNE)
    test = dataset_from(TEST_DIR, args.batch_size, False, args.seed, names).cache().prefetch(
        tf.data.AUTOTUNE
    )
    train = train.cache().shuffle(256, seed=args.seed).prefetch(tf.data.AUTOTUNE)
    weights = class_weights(TRAIN_DIR, names)

    model, backbone = build_model(len(names))
    head_checkpoint = MODEL_DIR / "head_best.keras"
    compile_model(model, 3e-4)
    head_history = model.fit(
        train,
        validation_data=validation,
        epochs=args.head_epochs,
        class_weight=weights,
        callbacks=callbacks(
            head_checkpoint, EVALUATION_DIR / "head_epoch_log.csv", patience=8
        ),
        verbose=2,
    )

    model = tf.keras.models.load_model(head_checkpoint)
    backbone = next(
        layer for layer in model.layers if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name
    )
    backbone.trainable = True
    for layer in backbone.layers[:-35]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    compile_model(model, 1e-5)
    fine_checkpoint = MODEL_DIR / "fine_tuned_best.keras"
    fine_history = model.fit(
        train,
        validation_data=validation,
        epochs=args.fine_tune_epochs,
        class_weight=weights,
        callbacks=callbacks(
            fine_checkpoint, EVALUATION_DIR / "fine_tune_epoch_log.csv", patience=8
        ),
        verbose=2,
    )

    head_loss = min(head_history.history["val_loss"])
    fine_loss = min(fine_history.history["val_loss"])
    selected_stage = "fine_tuned" if fine_loss < head_loss else "frozen_backbone"
    selected_checkpoint = fine_checkpoint if fine_loss < head_loss else head_checkpoint
    best_model = MODEL_DIR / "best_model.keras"
    shutil.copy2(selected_checkpoint, best_model)
    model = tf.keras.models.load_model(best_model)

    validation_actual = actual_labels(validation)
    validation_plain = predict_probabilities(model, validation, flip=False)
    validation_tta = predict_probabilities(model, validation, flip=True)
    plain_accuracy = float(np.mean(validation_plain.argmax(axis=1) == validation_actual))
    tta_accuracy = float(np.mean(validation_tta.argmax(axis=1) == validation_actual))
    use_flip_tta = tta_accuracy > plain_accuracy

    test_actual = actual_labels(test)
    test_probabilities = predict_probabilities(model, test, flip=use_flip_tta)
    test_predicted = test_probabilities.argmax(axis=1)
    matrix = confusion_matrix(test_actual, test_predicted, len(names))
    metrics = classification_metrics(matrix, names)
    test_accuracy = float(np.mean(test_predicted == test_actual))
    clipped = np.clip(test_probabilities[np.arange(len(test_actual)), test_actual], 1e-7, 1.0)
    metrics.update(
        {
            "test_accuracy": test_accuracy,
            "test_loss": float(-np.mean(np.log(clipped))),
            "test_samples": int(len(test_actual)),
            "class_names": names,
            "confusion_matrix": matrix.tolist(),
            "selected_training_stage": selected_stage,
            "validation_plain_accuracy": plain_accuracy,
            "validation_flip_tta_accuracy": tta_accuracy,
            "flip_tta_selected": use_flip_tta,
            "head_best_validation_loss": float(head_loss),
            "fine_tuned_best_validation_loss": float(fine_loss),
        }
    )

    histories = merge_histories(head_history, fine_history)
    save_history(histories, len(head_history.history["loss"]))
    save_confusion(matrix, names)
    (EVALUATION_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (MODEL_DIR / "labels.json").write_text(
        json.dumps({str(index): name for index, name in enumerate(names)}, indent=2),
        encoding="utf-8",
    )
    (MODEL_DIR / "training_config.json").write_text(
        json.dumps(
            {
                "image_size": list(IMAGE_SIZE),
                "batch_size": args.batch_size,
                "head_epochs": args.head_epochs,
                "fine_tune_epochs": args.fine_tune_epochs,
                "seed": args.seed,
                "backbone": "MobileNetV2 ImageNet",
                "selected_training_stage": selected_stage,
                "flip_tta_selected_on_validation": use_flip_tta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Selected stage: {selected_stage}")
    print(f"Flip TTA selected from validation: {use_flip_tta}")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Best model: {best_model}")


if __name__ == "__main__":
    main()
