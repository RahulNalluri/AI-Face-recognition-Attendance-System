import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

# ── CONFIG ─────────────────────────────────────────────────
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = str(ROOT / "data" / "raw")
MODEL_PATH   = str(ROOT / "models" / "knn")
LABELS       = ["Rahul", "Harshit", "Sohail", "Jagadeesh"]
IMG_SIZE     = (100, 100)
# ───────────────────────────────────────────────────────────

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

os.makedirs(MODEL_PATH, exist_ok=True)

# ── LOAD AND EXTRACT FEATURES ──────────────────────────────
print("[INFO] Extracting face features...")

features = []
labels   = []

for person in LABELS:
    folder = os.path.join(DATASET_PATH, person)
    count  = 0

    for img_file in os.listdir(folder):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img = cv2.imread(os.path.join(folder, img_file))
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Try to detect face
        face_found = False
        for scale in [1.05, 1.1, 1.2]:
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=scale,
                minNeighbors=3, minSize=(30, 30)
            )
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
                # Add padding
                pad = int(min(w, h) * 0.1)
                x1  = max(0, x-pad)
                y1  = max(0, y-pad)
                x2  = min(img.shape[1], x+w+pad)
                y2  = min(img.shape[0], y+h+pad)
                face = gray[y1:y2, x1:x2]
                face_found = True
                break

        if not face_found:
            # Center crop fallback
            h, w   = gray.shape
            size   = min(h, w)
            y0     = (h-size)//2
            x0     = (w-size)//2
            face   = gray[y0:y0+size, x0:x0+size]

        # Resize and normalize
        face_resized = cv2.resize(face, IMG_SIZE)
        face_eq      = cv2.equalizeHist(face_resized)  # histogram equalization
        face_flat    = face_eq.flatten() / 255.0        # flatten to 1D vector

        features.append(face_flat)
        labels.append(person)
        count += 1

    print(f"  {person}: {count} images")

features = np.array(features)
labels   = np.array(labels)

print(f"\n[INFO] Total: {len(features)} samples")
print(f"[INFO] Feature vector size: {features.shape[1]}")

# ── ENCODE LABELS ──────────────────────────────────────────
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# ── SPLIT ──────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    features, labels_encoded,
    test_size=0.2,
    random_state=42,
    stratify=labels_encoded
)

print(f"[INFO] Train: {len(X_train)} | Test: {len(X_test)}")

# ── TRAIN KNN ──────────────────────────────────────────────
# K=3 means it looks at 3 nearest neighbors and votes
print("\n[INFO] Training KNN classifier...")

knn = KNeighborsClassifier(
    n_neighbors=3,
    metric="euclidean",
    weights="distance"   # closer neighbors get more vote weight
)
knn.fit(X_train, y_train)

# ── EVALUATE ───────────────────────────────────────────────
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*55}")
print(f"  Test Accuracy : {accuracy * 100:.2f}%")
print(f"{'='*55}")
print("\nPer-person breakdown:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── SAVE ───────────────────────────────────────────────────
# Save KNN model
with open(os.path.join(MODEL_PATH, "knn_model.pkl"), "wb") as f:
    pickle.dump(knn, f)

# Save label encoder
with open(os.path.join(MODEL_PATH, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

# Save label map for app.py compatibility
label_map = {i: name for i, name in enumerate(le.classes_)}
np.save(os.path.join(MODEL_PATH, "label_map.npy"), label_map)

print(f"\n[DONE] model/knn_model.pkl")
print(f"[DONE] model/label_encoder.pkl")
print(f"[DONE] model/label_map.npy → {label_map}")
