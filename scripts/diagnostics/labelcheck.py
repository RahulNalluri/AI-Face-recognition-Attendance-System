import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "artifacts" / "processed"
X_train = np.load(PROCESSED / "X_train.npy")
y_train = np.load(PROCESSED / "y_train.npy")
label_map = {0:"Rahul", 1:"Harshit", 2:"Sohail", 3:"Jagadeesh"}

# Show 5 images per person
fig, axes = plt.subplots(4, 5, figsize=(15, 12))

for person_idx in range(4):
    # Get all images for this person
    person_images = X_train[y_train == person_idx]
    print(f"{label_map[person_idx]}: {len(person_images)} images")
    
    for j in range(5):
        ax = axes[person_idx][j]
        if j < len(person_images):
            ax.imshow(person_images[j])
        ax.set_title(f"{label_map[person_idx]}", fontsize=9)
        ax.axis("off")

plt.tight_layout()
plt.savefig(ROOT / "artifacts" / "diagnostics" / "label_check.png")
plt.show()
