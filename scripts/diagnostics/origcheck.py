import cv2
import os
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = str(ROOT / "data" / "raw")
LABELS = ["Rahul", "Harshit", "Sohail", "Jagadeesh"]

fig, axes = plt.subplots(4, 8, figsize=(20, 10))

for row, person in enumerate(LABELS):
    folder = os.path.join(DATASET_PATH, person)
    files  = sorted(os.listdir(folder))[:8]
    for col, f in enumerate(files):
        img = cv2.imread(os.path.join(folder, f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[row][col].imshow(img)
        axes[row][col].set_title(f"{person}", fontsize=7)
        axes[row][col].axis("off")

plt.tight_layout()
plt.savefig(ROOT / "artifacts" / "diagnostics" / "original_check.png")
plt.show()
