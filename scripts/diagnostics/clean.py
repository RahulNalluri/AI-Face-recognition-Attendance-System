import cv2
import os
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUGMENTED_PATH = str(ROOT / "data" / "legacy_augmented")
LABELS = ["Rahul", "Harshit", "Sohail", "Jagadeesh"]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

total_removed = 0

for person_name in LABELS:
    folder = os.path.join(AUGMENTED_PATH, person_name)
    files  = [f for f in os.listdir(folder)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    removed = 0
    kept    = 0

    for img_file in files:
        img_path = os.path.join(folder, img_file)
        img = cv2.imread(img_path)

        if img is None:
            os.remove(img_path)
            removed += 1
            continue

        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Check 1 — is the image too bright or too dark (blank/background)
        mean_val = gray.mean()
        if mean_val < 30 or mean_val > 230:
            os.remove(img_path)
            removed += 1
            continue

        # Check 2 — does it still have a detectable face
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20)
        )

        if len(faces) == 0:
            os.remove(img_path)
            removed += 1
            continue

        kept += 1

    print(f"{person_name}: kept {kept}, removed {removed}")
    total_removed += removed

print(f"\nTotal removed : {total_removed}")
print("Now run preprocess.py and train.py again")
