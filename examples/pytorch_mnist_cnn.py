"""
A convolutional neural network in PyTorch, trained on MNIST (28x28 digits).

This is the PyTorch version of cnn_digits.py, scaled up to real-size images:
  * PyTorch computes every backward pass for you (autograd), so there is no
    hand-written backprop, im2col or Adam code.
  * MNIST has 60,000 training images and 10,000 test images, each 28x28.
    torchvision downloads it (about 12 MB) the first time you run this.
  * Uses a GPU automatically if one is available (CUDA or Apple Silicon).

New ideas compared with the NumPy CNN:
  * nn.Module / nn.Sequential to define the model
  * Dataset + DataLoader for batching and shuffling
  * BatchNorm to keep activations well scaled and speed up training
  * A one-cycle learning-rate schedule
  * Saving the best model to disk and loading it again for predictions

Install:  pip install torch torchvision
Run:      python pytorch_mnist_cnn.py            (5 epochs)
          python pytorch_mnist_cnn.py --epochs 10
"""
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ======================================================================= model
class DigitCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()

        def block(c_in, c_out):
            # Two 3x3 convolutions, then halve the image size
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(1, 32),     # 1x28x28  -> 32x14x14
            block(32, 64),    # 32x14x14 -> 64x7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                 # 64x7x7 -> 3136
            nn.Dropout(0.3),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),    # raw scores (logits)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ======================================================================== data
MEAN, STD = 0.1307, 0.3081  # MNIST pixel mean and standard deviation


def get_loaders(batch_size, data_dir="./data", use_cuda=False):
    train_tf = transforms.Compose([
        # Data augmentation: small random rotations and shifts
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),                 # uint8 0..255 -> float 0..1
        transforms.Normalize((MEAN,), (STD,)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,)),
    ])
    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=train_tf)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=test_tf)

    kw = {"num_workers": 2, "pin_memory": use_cuda}
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **kw)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False, **kw)
    return train_loader, test_loader, test_set


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================ train / evaluate
def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()                        # turns dropout and BatchNorm updates on
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)                       # forward pass
        loss = F.cross_entropy(logits, labels)       # softmax + cross-entropy

        optimizer.zero_grad()
        loss.backward()                              # autograd does backprop
        optimizer.step()                             # update the weights
        scheduler.step()                             # adjust the learning rate

        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        seen += len(labels)
    return total_loss / seen, correct / seen


@torch.no_grad()                         # no gradients needed for evaluation
def evaluate(model, loader, device):
    model.eval()                         # dropout off, BatchNorm uses saved stats
    total_loss, correct, seen = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        seen += len(labels)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
    return total_loss / seen, correct / seen, torch.cat(all_preds), torch.cat(all_labels)


# ===================================================================== helpers
def confusion_matrix(preds, labels, n=10):
    m = torch.zeros(n, n, dtype=torch.long)
    for t, p in zip(labels, preds):
        m[t, p] += 1
    return m


def ascii_digit(img_uint8):
    """Draw a 28x28 image as 14 lines of text (every second pixel)."""
    shades = " .:-=+*#%@"
    small = img_uint8[::2, ::2].float() / 255
    return "\n".join("".join(shades[int(v * 9)] * 2 for v in row) for row in small)


# ======================================================================== main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--save", default="mnist_cnn.pt")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = pick_device()
    print(f"Using device: {device}")

    train_loader, test_loader, test_set = get_loaders(
        args.batch_size, use_cuda=device.type == "cuda")

    model = DigitCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params:,} trainable parameters\n")
    print(model, "\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # One-cycle: warm the learning rate up, then anneal it down to near zero
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        te_loss, te_acc, _, _ = evaluate(model, test_loader, device)
        print(f"epoch {epoch:2d}  train loss {tr_loss:.4f} acc {tr_acc:.4f}  |  "
              f"test loss {te_loss:.4f} acc {te_acc:.4f}  ({time.time() - start:.0f}s)")
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), args.save)

    print(f"\nBest test accuracy: {best_acc:.2%}  (weights saved to {args.save})")

    # ---- Load the saved model and use it, as you would in a separate program
    model = DigitCNN().to(device)
    model.load_state_dict(torch.load(args.save, map_location=device))
    _, acc, preds, labels = evaluate(model, test_loader, device)

    print("\nConfusion matrix (rows = true digit, columns = predicted)")
    print("      " + " ".join(f"{d:4d}" for d in range(10)))
    for d, row in enumerate(confusion_matrix(preds, labels)):
        print(f"{d:4d}  " + " ".join(f"{v:4d}" for v in row.tolist()))

    wrong = (preds != labels).nonzero().flatten()
    print(f"\n{len(wrong)} of {len(labels)} test images misclassified.")
    for i in wrong[:2].tolist():
        image, label = test_set[i]
        with torch.no_grad():
            probs = F.softmax(model(image.unsqueeze(0).to(device)), dim=1)[0]
        top2 = probs.topk(2)
        print(f"\nTrue {label}, predicted {top2.indices[0].item()} "
              f"({top2.values[0].item():.0%}), second guess "
              f"{top2.indices[1].item()} ({top2.values[1].item():.0%})")
        print(ascii_digit(test_set.data[i]))


if __name__ == "__main__":   # required for DataLoader workers on Windows/macOS
    main()
