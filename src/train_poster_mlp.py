"""Train a small NumPy MLP solely to produce reproducible poster metrics."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
rng = np.random.default_rng(42)


def prepare(path):
    x = np.load(path).astype(np.float32)
    # 96x96 RGB -> 24x24 grayscale using 4x4 average pooling.
    x = x.mean(axis=3)
    x = x.reshape(len(x), 24, 4, 24, 4).mean(axis=(2, 4))
    return x.reshape(len(x), -1)


X_train = prepare(ROOT / "artifacts" / "processed" / "X_train.npy")
X_test = prepare(ROOT / "artifacts" / "processed" / "X_test.npy")
y_train = np.load(ROOT / "artifacts" / "processed" / "y_train.npy").astype(np.int64)
y_test = np.load(ROOT / "artifacts" / "processed" / "y_test.npy").astype(np.int64)

mean = X_train.mean(axis=0, keepdims=True)
std = X_train.std(axis=0, keepdims=True) + 1e-5
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

n_in, n_hidden, n_out = X_train.shape[1], 128, 4
W1 = rng.normal(0, np.sqrt(2 / n_in), (n_in, n_hidden)).astype(np.float32)
b1 = np.zeros((1, n_hidden), np.float32)
W2 = rng.normal(0, np.sqrt(2 / n_hidden), (n_hidden, n_out)).astype(np.float32)
b2 = np.zeros((1, n_out), np.float32)
params = [W1, b1, W2, b2]
m = [np.zeros_like(p) for p in params]
v = [np.zeros_like(p) for p in params]


def forward(x):
    z1 = x @ W1 + b1
    h = np.maximum(z1, 0)
    logits = h @ W2 + b2
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return z1, h, probs


def metrics(x, y):
    _, _, probs = forward(x)
    loss = -np.log(probs[np.arange(len(y)), y] + 1e-9).mean()
    pred = probs.argmax(axis=1)
    return float(loss), float((pred == y).mean()), pred


epochs, batch_size, lr = 30, 64, 0.0015
train_acc, val_acc, train_loss, val_loss = [], [], [], []
step = 0
for epoch in range(epochs):
    order = rng.permutation(len(X_train))
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        xb, yb = X_train[idx], y_train[idx]
        z1, h, probs = forward(xb)
        dlogits = probs
        dlogits[np.arange(len(yb)), yb] -= 1
        dlogits /= len(yb)
        gW2 = h.T @ dlogits
        gb2 = dlogits.sum(axis=0, keepdims=True)
        dh = dlogits @ W2.T
        dz1 = dh * (z1 > 0)
        gW1 = xb.T @ dz1
        gb1 = dz1.sum(axis=0, keepdims=True)
        grads = [gW1, gb1, gW2, gb2]
        step += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * (g * g)
            mh = m[i] / (1 - 0.9 ** step)
            vh = v[i] / (1 - 0.999 ** step)
            p -= lr * mh / (np.sqrt(vh) + 1e-8)
    tl, ta, _ = metrics(X_train, y_train)
    vl, va, _ = metrics(X_test, y_test)
    train_loss.append(tl); train_acc.append(ta)
    val_loss.append(vl); val_acc.append(va)
    print(f"epoch={epoch+1:02d} train_acc={ta:.4f} val_acc={va:.4f} train_loss={tl:.4f} val_loss={vl:.4f}")

_, final_acc, pred = metrics(X_test, y_test)
confusion = np.zeros((4, 4), dtype=np.int32)
for true, guessed in zip(y_test, pred):
    confusion[true, guessed] += 1
per_class = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)

out = ROOT / "artifacts" / "metrics" / "model_metrics.npz"
out.parent.mkdir(parents=True, exist_ok=True)
np.savez(
    out,
    train_acc=np.array(train_acc), val_acc=np.array(val_acc),
    train_loss=np.array(train_loss), val_loss=np.array(val_loss),
    confusion=confusion, per_class=per_class, final_acc=final_acc,
)
print(f"saved={out}")
