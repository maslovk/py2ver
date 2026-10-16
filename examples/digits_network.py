"""
A deeper neural network that recognises handwritten digits (0-9).

Step up from the XOR example:
  XOR network : 2 -> 4 -> 1          tanh + sigmoid, 4 examples, full batch
  This network: 64 -> 128 -> 64 -> 10  ReLU + softmax, ~1,800 real images,
                mini-batches, Adam optimiser, train/test split, accuracy.

Everything (forward pass, backprop, Adam) is written by hand in NumPy.
scikit-learn is used only to load the built-in 8x8 digits dataset.
"""
import numpy as np
from sklearn.datasets import load_digits

rng = np.random.default_rng(0)

# ---------------------------------------------------------------- data
digits = load_digits()                      # 1,797 images of 8x8 pixels
X = digits.data / 16.0                      # pixel values 0..16 -> 0..1
y = digits.target                           # labels 0..9

idx = rng.permutation(len(X))
split = int(0.8 * len(X))
train_idx, test_idx = idx[:split], idx[split:]
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]


def one_hot(labels, n_classes=10):
    out = np.zeros((len(labels), n_classes))
    out[np.arange(len(labels)), labels] = 1
    return out


# ---------------------------------------------------------------- layers
class Dense:
    """Fully connected layer: out = x @ W + b"""

    def __init__(self, n_in, n_out):
        # He initialisation works well with ReLU
        self.W = rng.normal(0, np.sqrt(2 / n_in), (n_in, n_out))
        self.b = np.zeros(n_out)
        # Adam optimiser state
        self.mW, self.vW = np.zeros_like(self.W), np.zeros_like(self.W)
        self.mb, self.vb = np.zeros_like(self.b), np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad):
        self.dW = self.x.T @ grad
        self.db = grad.sum(axis=0)
        return grad @ self.W.T              # gradient for the previous layer

    def adam_step(self, lr, t, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-4):
        self.dW += weight_decay * self.W    # L2 regularisation
        for p, g, m, v in ((self.W, self.dW, self.mW, self.vW),
                           (self.b, self.db, self.mb, self.vb)):
            m[:] = beta1 * m + (1 - beta1) * g
            v[:] = beta2 * v + (1 - beta2) * g ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)    # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- network
class DigitNetwork:
    def __init__(self, sizes=(64, 128, 64, 10)):
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Dense(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:          # no ReLU after the last layer
                self.layers.append(ReLU())
        self.t = 0                          # Adam step counter

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return softmax(x)                   # class probabilities

    def train_batch(self, x, y_onehot, lr):
        probs = self.forward(x)
        loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-9), axis=1))

        grad = (probs - y_onehot) / len(x)  # softmax + cross-entropy gradient
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        self.t += 1
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.adam_step(lr, self.t)
        return loss

    def predict(self, x):
        return self.forward(np.atleast_2d(x)).argmax(axis=1)

    def accuracy(self, x, labels):
        return np.mean(self.predict(x) == labels)

    def fit(self, X, y, epochs=30, batch_size=32, lr=0.001):
        Y = one_hot(y)
        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(X))
            losses = []
            for start in range(0, len(X), batch_size):
                b = order[start:start + batch_size]
                losses.append(self.train_batch(X[b], Y[b], lr))
            if epoch == 1 or epoch % 5 == 0:
                print(f"epoch {epoch:3d}   loss {np.mean(losses):.4f}   "
                      f"train acc {self.accuracy(X, y):.3f}   "
                      f"test acc {self.accuracy(X_test, y_test):.3f}")


# ---------------------------------------------------------------- helpers
def show_digit(pixels):
    shades = " .:-=+*#%@"
    img = pixels.reshape(8, 8)
    return "\n".join("".join(shades[int(v * 9)] * 2 for v in row) for row in img)


def confusion_matrix(true, pred, n=10):
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(true, pred):
        m[t, p] += 1
    return m


# ---------------------------------------------------------------- run
if __name__ == "__main__":
    print(f"Training on {len(X_train)} images, testing on {len(X_test)}\n")
    net = DigitNetwork()
    n_params = sum(l.W.size + l.b.size for l in net.layers if isinstance(l, Dense))
    print(f"Network 64 -> 128 -> 64 -> 10, {n_params:,} trainable parameters\n")

    net.fit(X_train, y_train, epochs=30)

    preds = net.predict(X_test)
    print(f"\nFinal test accuracy: {np.mean(preds == y_test):.1%}")

    print("\nConfusion matrix (rows = true digit, columns = predicted)")
    print("     " + " ".join(f"{d:3d}" for d in range(10)))
    for d, row in enumerate(confusion_matrix(y_test, preds)):
        print(f"{d:3d}  " + " ".join(f"{v:3d}" for v in row))

    print("\nA few test images:")
    probs = net.forward(X_test[:3])
    for i in range(3):
        print(f"\nTrue: {y_test[i]}   Predicted: {probs[i].argmax()} "
              f"({probs[i].max():.1%} confident)")
        print(show_digit(X_test[i]))

    wrong = np.where(preds != y_test)[0]
    if len(wrong):
        i = wrong[0]
        print(f"\nOne it got wrong. True: {y_test[i]}, predicted: {preds[i]}")
        print(show_digit(X_test[i]))
