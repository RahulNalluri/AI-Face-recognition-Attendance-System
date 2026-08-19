import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "artifacts" / "processed"
X_train = np.load(PROCESSED / "X_train.npy")
y_train = np.load(PROCESSED / "y_train.npy")

label_map = {0:"Rahul", 1:"Harshit", 2:"Sohail", 3:"Jagadeesh"}

fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i])
    ax.set_title(label_map[y_train[i]], fontsize=8)
    ax.axis("off")

plt.tight_layout()
plt.savefig(ROOT / "artifacts" / "diagnostics" / "sample_check.png")
plt.show()
print("Saved sample_check.png")
