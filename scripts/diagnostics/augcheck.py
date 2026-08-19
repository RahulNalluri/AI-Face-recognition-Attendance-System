import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

LABELS = ["Rahul", "Harshit", "Sohail", "Jagadeesh"]
ROOT = Path(__file__).resolve().parents[2]
AUGMENTED_PATH = str(ROOT / "data" / "legacy_augmented")

fig, axes = plt.subplots(4, 8, figsize=(20, 10))

for row, person in enumerate(LABELS):
    folder = os.path.join(AUGMENTED_PATH, person)
    files  = sorted(os.listdir(folder))[:8]
    
    for col, img_file in enumerate(files):
        img = cv2.imread(os.path.join(folder, img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[row][col].imshow(img)
        axes[row][col].set_title(f"{person}\n{img_file[:8]}", fontsize=7)
        axes[row][col].axis("off")

plt.tight_layout()
plt.savefig(ROOT / "artifacts" / "diagnostics" / "aug_check.png")
plt.show()
