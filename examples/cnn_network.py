"""
A convolutional neural network (CNN) built from scratch in NumPy.

Step up from the previous examples:
  1. XOR network   : 2 -> 4 -> 1               (dense layers, 17 params)
  2. Digit MLP     : 64 -> 128 -> 64 -> 10     (dense layers, 17k params)
  3. This CNN      : conv -> pool -> conv -> pool -> dense -> dense

New ideas in this file:
  * Convolution layers: small 3x3 filters slide over the image and look for
    local patterns (edges, curves), sharing the same weights everywhere.
  * Max pooling: shrinks each feature map by keeping the strongest response
    in every 2x2 block, so the network cares less about exact position.
  * Dropout: randomly switches off neurons during training to reduce overfitting.
  * Data augmentation: shifts training images by a pixel so the network
    sees more variety than the raw dataset contains.
  * A generic Adam optimiser that works on any layer's parameters.

Image shapes follow the usual (batch, channels, height, width) layout.
scikit-learn is used only to load the built-in 8x8 digits dataset.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.datasets import load_digits

rng = np.random.default_rng(0)


# ====================================================================== layers
class Conv2D:
    """3x3 convolution, stride 1, zero padding 1 (output keeps the same size)."""

    def __init__(self, in_ch, out_ch, k=3):
        self.k = k
        fan_in = in_ch * k * k
        self.W = rng.normal(0, np.sqrt(2 / fan_in), (out_ch, in_ch, k, k))
        self.b = np.zeros(out_ch)

    def forward(self, x, training=True):
        N, C, H, W = x.shape
        self.x_shape = x.shape
        p = self.k // 2
        xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
        # Every 3x3 patch as a row ("im2col"), so convolution becomes one matmul
        win = sliding_window_view(xp, (self.k, self.k), axis=(2, 3))  # N,C,H,W,k,k
        self.cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(N * H * W, -1)
        out = self.cols @ self.W.reshape(len(self.W), -1).T + self.b  # (N*H*W, F)
        return out.reshape(N, H, W, -1).transpose(0, 3, 1, 2)

    def backward(self, grad):
        N, C, H, W = self.x_shape
        F, k, p = len(self.W), self.k, self.k // 2
        g = grad.transpose(0, 2, 3, 1).reshape(-1, F)                 # (N*H*W, F)
        self.dW = (g.T @ self.cols).reshape(self.W.shape)
        self.db = g.sum(axis=0)
        dcols = (g @ self.W.reshape(F, -1)).reshape(N, H, W, C, k, k)
        # "col2im": add each patch's gradient back to where it came from
        dxp = np.zeros((N, C, H + 2 * p, W + 2 * p))
        for i in range(k):
            for j in range(k):
                dxp[:, :, i:i + H, j:j + W] += dcols[..., i, j].transpose(0, 3, 1, 2)
        return dxp[:, :, p:p + H, p:p + W]

    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]


class MaxPool2x2:
    def forward(self, x, training=True):
        N, C, H, W = x.shape
        blocks = x.reshape(N, C, H // 2, 2, W // 2, 2)
        out = blocks.max(axis=(3, 5))
        self.mask = blocks == out[:, :, :, None, :, None]  # where the max was
        self.x_shape = x.shape
        return out

    def backward(self, grad):
        # Only the winning pixel in each 2x2 block receives the gradient
        g = self.mask * grad[:, :, :, None, :, None]
        return g.reshape(self.x_shape)


class ReLU:
    def forward(self, x, training=True):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask


class Flatten:
    def forward(self, x, training=True):
        self.shape = x.shape
        return x.reshape(len(x), -1)

    def backward(self, grad):
        return grad.reshape(self.shape)


class Dropout:
    def __init__(self, rate=0.3):
        self.rate = rate

    def forward(self, x, training=True):
        if not training:
            return x
        self.mask = (rng.random(x.shape) > self.rate) / (1 - self.rate)
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask


class Dense:
    def __init__(self, n_in, n_out):
        self.W = rng.normal(0, np.sqrt(2 / n_in), (n_in, n_out))
        self.b = np.zeros(n_out)

    def forward(self, x, training=True):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad):
        self.dW = self.x.T @ grad
        self.db = grad.sum(axis=0)
        return grad @ self.W.T

    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]


# =================================================================== optimiser
class Adam:
    def __init__(self, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-4):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, beta1, beta2, eps, weight_decay
        self.state, self.t = {}, 0

    def step(self, layers):
        self.t += 1
        for layer in layers:
            if not hasattr(layer, "params"):
                continue
            for p, g in layer.params():
                if p.ndim > 1:
                    g = g + self.wd * p                  # weight decay on weights only
                m, v = self.state.setdefault(id(p), (np.zeros_like(p), np.zeros_like(p)))
                m[:] = self.b1 * m + (1 - self.b1) * g
                v[:] = self.b2 * v + (1 - self.b2) * g * g
                m_hat = m / (1 - self.b1 ** self.t)
                v_hat = v / (1 - self.b2 ** self.t)
                p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ===================================================================== network
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class CNN:
    def __init__(self):
        self.layers = [
            Conv2D(1, 16),  ReLU(), MaxPool2x2(),   # 1x8x8   -> 16x8x8 -> 16x4x4
            Conv2D(16, 32), ReLU(), MaxPool2x2(),   # 16x4x4  -> 32x4x4 -> 32x2x2
            Flatten(),                               # 32x2x2  -> 128
            Dense(128, 64), ReLU(), Dropout(0.2),    # 128     -> 64
            Dense(64, 10),                           # 64      -> 10 class scores
        ]
        self.opt = Adam(lr=0.002)

    def forward(self, x, training=False):
        for layer in self.layers:
            x = layer.forward(x, training)
        return softmax(x)

    def train_batch(self, x, labels):
        probs = self.forward(x, training=True)
        loss = -np.mean(np.log(probs[np.arange(len(x)), labels] + 1e-9))
        grad = probs.copy()
        grad[np.arange(len(x)), labels] -= 1     # softmax + cross-entropy gradient
        grad /= len(x)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        self.opt.step(self.layers)
        return loss

    def predict(self, x, batch=256):
        return np.concatenate([self.forward(x[i:i + batch]).argmax(1)
                               for i in range(0, len(x), batch)])

    def n_params(self):
        return sum(p.size for l in self.layers if hasattr(l, "W") for p in (l.W, l.b))


# ======================================================================== data
def random_shift(images, max_shift=1):
    """Data augmentation: move each image up to 1 pixel in any direction."""
    N, C, H, W = images.shape
    padded = np.pad(images, ((0, 0), (0, 0), (max_shift,) * 2, (max_shift,) * 2))
    out = np.empty_like(images)
    dy, dx = rng.integers(0, 2 * max_shift + 1, (2, N))
    for i in range(N):
        out[i] = padded[i, :, dy[i]:dy[i] + H, dx[i]:dx[i] + W]
    return out


def show(img, shades=" .:-=+*#%@"):
    img = (img - img.min()) / (np.ptp(img) + 1e-9)
    return ["".join(shades[int(v * 9)] * 2 for v in row) for row in img]


# ========================================================================= run
if __name__ == "__main__":
    digits = load_digits()
    X = (digits.images / 16.0)[:, None, :, :]    # (1797, 1, 8, 8)
    y = digits.target
    idx = rng.permutation(len(X))
    split = int(0.8 * len(X))
    X_train, y_train = X[idx[:split]], y[idx[:split]]
    X_test, y_test = X[idx[split:]], y[idx[split:]]

    net = CNN()
    print(f"Training on {len(X_train)} images, testing on {len(X_test)}")
    print(f"CNN with {net.n_params():,} trainable parameters\n")

    epochs, batch_size = 40, 32
    for epoch in range(1, epochs + 1):
        if epoch == 30:
            net.opt.lr /= 4                       # lower learning rate to fine-tune
        order = rng.permutation(len(X_train))
        losses = []
        for s in range(0, len(order), batch_size):
            b = order[s:s + batch_size]
            losses.append(net.train_batch(random_shift(X_train[b]), y_train[b]))
        if epoch == 1 or epoch % 5 == 0:
            tr = np.mean(net.predict(X_train) == y_train)
            te = np.mean(net.predict(X_test) == y_test)
            print(f"epoch {epoch:3d}   loss {np.mean(losses):.4f}   "
                  f"train acc {tr:.3f}   test acc {te:.3f}")

    preds = net.predict(X_test)
    print(f"\nFinal test accuracy: {np.mean(preds == y_test):.1%}")

    # What does the network "see"? Show the input and 4 first-layer feature maps.
    conv1 = net.layers[0]
    sample = X_test[:1]
    fmap = np.maximum(conv1.forward(sample, training=False), 0)[0]
    strongest = np.argsort(-fmap.reshape(16, -1).max(1))[:4]
    print(f"\nTest image (true {y_test[0]}, predicted {preds[0]}) and what 4 of the "
          f"16 first-layer filters respond to:")
    panels = [show(sample[0, 0])] + [show(fmap[f]) for f in strongest]
    print("   ".join(f"{t:<16}" for t in ["input"] + [f"filter {f}" for f in strongest]))
    for row in zip(*panels):
        print("   ".join(row))
