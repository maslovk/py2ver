import numpy as np

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y = np.array([[0], [1], [1], [0]], dtype=float)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class XORNetwork:
    def __init__(self, n_in=2, n_hidden=4, n_out=1, seed=2026):
        rng = np.random.default_rng(seed)
        self.W1 = rng.uniform(-1.5, 1.5, (n_in, n_hidden))   # input  -> hidden
        self.b1 = rng.uniform(-0.5, 0.5, (1, n_hidden))
        self.W2 = rng.uniform(-1.5, 1.5, (n_hidden, n_out))  # hidden -> output
        self.b2 = np.zeros((1, n_out))

    def forward(self, x):
        self.x = x
        self.h = np.tanh(x @ self.W1 + self.b1)       # hidden activations
        self.o = sigmoid(self.h @ self.W2 + self.b2)  # output probability
        return self.o

    def loss(self, y):
        eps = 1e-9
        return -np.mean(y * np.log(self.o + eps) + (1 - y) * np.log(1 - self.o + eps))

    def backward(self, y, lr):
        n = len(y)
        dz2 = (self.o - y) / n                         # sigmoid + cross-entropy
        dW2 = self.h.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)

        dz1 = (dz2 @ self.W2.T) * (1 - self.h ** 2)    # back through tanh
        dW1 = self.x.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def train(self, X, Y, lr=0.5, max_steps=10_000, target_loss=0.01, log_every=200):
        for step in range(1, max_steps + 1):
            self.forward(X)
            L = self.loss(Y)
            self.backward(Y, lr)
            if step % log_every == 0 or L < target_loss:
                print(f"step {step:5d}   loss {L:.4f}")
            if L < target_loss:
                break
        return L

    def predict(self, x):
        return self.forward(np.atleast_2d(x))


if __name__ == "__main__":
    net = XORNetwork()
    net.train(X, Y, lr=0.5)

    print("\n x1  x2 | target  prediction  class")
    print("--------+---------------------------")
    for x, y in zip(X, Y):
        p = net.predict(x)[0, 0]
        print(f" {int(x[0]):2d}  {int(x[1]):2d} |   {int(y[0])}      {p:.3f}      {int(p >= 0.5)}")

    print(f"\nInput (0.8, 0.3) -> {net.predict([0.8, 0.3])[0, 0]:.3f}")
