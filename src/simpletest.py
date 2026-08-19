import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "artifacts" / "processed"
X_train = np.load(PROCESSED / "X_train.npy")
y_train = np.load(PROCESSED / "y_train.npy")
X_test  = np.load(PROCESSED / "X_test.npy")
y_test  = np.load(PROCESSED / "y_test.npy")

y_train_cat = to_categorical(y_train, 4)
y_test_cat  = to_categorical(y_test,  4)

# Simplest possible model — just flatten and classify
model = Sequential([
    Flatten(input_shape=(96, 96, 3)),
    Dense(64, activation="relu"),
    Dense(4,  activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    X_train, y_train_cat,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test_cat),
    verbose=1
)

loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nAccuracy: {acc*100:.2f}%")
print(f"Loss    : {loss:.4f}")
