import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img

# ── CONFIG ─────────────────────────────────────────────────
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH   = str(ROOT / "data" / "raw")
AUGMENTED_PATH = str(ROOT / "data" / "legacy_augmented")
TARGET         = 300
LABELS         = ["Rahul", "Harshit", "Sohail", "Jagadeesh"]
IMG_SIZE       = (96, 96)
# ───────────────────────────────────────────────────────────

augmentor = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.75, 1.25],
    fill_mode="nearest"
)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

for person_name in LABELS:
    src = os.path.join(DATASET_PATH,   person_name)
    dst = os.path.join(AUGMENTED_PATH, person_name)

    # Clear existing augmented folder
    if os.path.exists(dst):
        for f in os.listdir(dst):
            os.remove(os.path.join(dst, f))
    os.makedirs(dst, exist_ok=True)

    print(f"\n[INFO] Processing: {person_name}")

    # ── Step 1: collect clean face crops from original dataset ──
    good_faces = []

    for img_file in os.listdir(src):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img = cv2.imread(os.path.join(src, img_file))
        if img is None:
            continue

        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Try with relaxed parameters first
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40)
        )

        if len(faces) == 0:
            # Try even more relaxed
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30)
            )

        if len(faces) == 0:
            # No face detected — use center crop of the image
            # Better than skipping — original dataset images should have faces
            h, w = img.shape[:2]
            margin = int(min(h, w) * 0.1)
            face_crop = img[margin:h-margin, margin:w-margin]
        else:
            # Take largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            # Add 10% padding around face
            pad = int(min(w, h) * 0.1)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(img.shape[1], x + w + pad)
            y2 = min(img.shape[0], y + h + pad)
            face_crop = img[y1:y2, x1:x2]

        # Resize and convert to RGB
        face_resized = cv2.resize(face_crop, IMG_SIZE)
        face_rgb     = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        good_faces.append(face_rgb)

    print(f"  [INFO] {len(good_faces)} clean crops from original dataset")

    if len(good_faces) == 0:
        print(f"  [ERROR] No images found for {person_name}")
        continue

    # ── Step 2: save originals ──────────────────────────────────
    for i, face in enumerate(good_faces):
        bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(dst, f"orig_{i:03d}.jpg"), bgr)

    # ── Step 3: augment until TARGET reached ───────────────────
    count    = len(good_faces)
    idx      = 0

    while count < TARGET:
        base = good_faces[idx % len(good_faces)]
        idx += 1

        arr = img_to_array(base).reshape((1, 96, 96, 3))

        for batch in augmentor.flow(arr, batch_size=1):
            aug = array_to_img(batch[0])
            aug_np  = np.array(aug)

            # Verify the augmented image is not blank
            if aug_np.mean() < 20 or aug_np.mean() > 235:
                break  # skip this bad augmentation

            aug_bgr = cv2.cvtColor(aug_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(dst, f"aug_{count:04d}.jpg"), aug_bgr)
            count += 1
            break

    print(f"  [DONE] {count} total images saved → {dst}")

print(f"\n[ALL DONE] Clean augmented dataset ready in '{AUGMENTED_PATH}/'")
