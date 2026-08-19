import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "artifacts" / "processed"
X_train = np.load(PROCESSED / "X_train.npy")
X_test  = np.load(PROCESSED / "X_test.npy")
y_train = np.load(PROCESSED / "y_train.npy")
y_test  = np.load(PROCESSED / "y_test.npy")

print("X_train shape  :", X_train.shape)
print("X_test shape   :", X_test.shape)
print("y_train shape  :", y_train.shape)
print("y_test shape   :", y_test.shape)

print("\nX_train min    :", X_train.min())
print("X_train max    :", X_train.max())
print("X_train mean   :", X_train.mean())

print("\ny_train unique :", np.unique(y_train, return_counts=True))
print("y_test  unique :", np.unique(y_test,  return_counts=True))

# Check if all images are actually different
print("\nFirst image sum  :", X_train[0].sum())
print("Second image sum :", X_train[1].sum())
print("Third image sum  :", X_train[2].sum())

print("\nSample y_train:", y_train[:20])
print("Sample y_test :", y_test[:20])
