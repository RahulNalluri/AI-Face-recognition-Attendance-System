import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

# ── CONFIGURATION ──────────────────────────────────────────
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = str(ROOT / "data" / "legacy_augmented")
SAVE_PATH    = str(ROOT / "artifacts" / "processed")
IMG_SIZE     = (96, 96)              # ← changed to 96x96 to match train.py
LABELS = ["Rahul", "Harshit", "Sohail", "Jagadeesh"]
# ───────────────────────────────────────────────────────────


def load_and_preprocess():

    images = []
    labels = []

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    for label_index, person_name in enumerate(LABELS):
        person_folder = os.path.join(DATASET_PATH, person_name)

        if not os.path.exists(person_folder):
            print(f"[WARNING] Folder not found: {person_folder}")
            continue

        print(f"\n[INFO] Processing: {person_name} (label={label_index})")
        count = 0

        for img_file in os.listdir(person_folder):

            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(person_folder, img_file)
            img = cv2.imread(img_path)

            if img is None:
                print(f"  [SKIP] Could not read: {img_file}")
                continue

            # Since augment.py already cropped and resized faces to 96x96
            # we just normalize directly without detecting again
            face_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, IMG_SIZE)
            face_normalized = face_resized / 255.0

            images.append(face_normalized)
            labels.append(label_index)
            count += 1

        print(f"  [OK] {count} images loaded from {person_name}")

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    print(f"\n[INFO] Total images: {len(images)}")
    print(f"[INFO] Label distribution: { {LABELS[i]: int(np.sum(labels==i)) for i in range(len(LABELS))} }")

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    print(f"\n[INFO] Training samples : {len(X_train)}")
    print(f"[INFO] Testing  samples : {len(X_test)}")

    os.makedirs(SAVE_PATH, exist_ok=True)
    np.save(os.path.join(SAVE_PATH, "X_train.npy"), X_train)
    np.save(os.path.join(SAVE_PATH, "X_test.npy"),  X_test)
    np.save(os.path.join(SAVE_PATH, "y_train.npy"), y_train)
    np.save(os.path.join(SAVE_PATH, "y_test.npy"),  y_test)

    label_map = {i: name for i, name in enumerate(LABELS)}
    np.save(os.path.join(SAVE_PATH, "label_map.npy"), label_map)

    print(f"\n[DONE] All arrays saved to '{SAVE_PATH}/' folder")
    print(f"[DONE] Label map: {label_map}")


if __name__ == "__main__":
    load_and_preprocess()
