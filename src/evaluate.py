"""Post-training evaluation: curves, confusion matrix, classification report, sample grid."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # never pop a blocking window; always save to file
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from src.data import IMAGENET_MEAN, IMAGENET_STD


def plot_training_curves(results: dict, out_path: Path) -> None:
    epochs = range(1, len(results["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, results["train_loss"], label="train")
    axes[0].plot(epochs, results["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, results["train_acc"], label="train")
    axes[1].plot(epochs, results["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@torch.inference_mode()
def evaluate_on_test(model: nn.Module, test_loader: DataLoader, device: str):
    """Single pass over the held-out test set. Returns (y_true, y_pred, acc)."""
    model.eval()
    y_true, y_pred = [], []
    correct = 0

    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    acc = correct / len(y_true)
    return np.array(y_true), np.array(y_pred), acc


def plot_confusion_matrix(y_true, y_pred, class_names, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (normalized)")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                     color="white" if cm[i, j] > 0.5 else "black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_classification_report(y_true, y_pred, class_names, out_path: Path) -> str:
    report = classification_report(y_true, y_pred, target_names=class_names)
    Path(out_path).write_text(report)
    return report


def _denormalize(img_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (img_tensor.cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


@torch.inference_mode()
def plot_sample_predictions(model: nn.Module, test_loader: DataLoader, class_names,
                             device: str, out_path: Path, n: int = 12) -> None:
    """Grid of test images with predicted vs actual label + confidence.

    Pulls batches until it has at least one misclassified example (when one exists),
    so the grid isn't all easy correct cases.
    """
    model.eval()
    samples = []
    misclassified_indices = []

    for X, y in test_loader:
        logits = model(X.to(device))
        probs = torch.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)
        for i in range(len(y)):
            samples.append((X[i], y[i].item(), preds[i].item(), confs[i].item()))
            if preds[i].item() != y[i].item():
                misclassified_indices.append(len(samples) - 1)

    # test_loader is unshuffled (ImageFolder groups by class), so shuffle our own
    # picks here -- otherwise the grid would only ever show the first class.
    rng = np.random.RandomState(42)
    rng.shuffle(misclassified_indices)
    remaining_indices = [i for i in range(len(samples)) if i not in set(misclassified_indices)]
    rng.shuffle(remaining_indices)

    chosen_indices = misclassified_indices[: n // 3]
    chosen_indices += remaining_indices[: n - len(chosen_indices)]
    chosen = [samples[i] for i in chosen_indices]

    cols = 4
    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)

    for ax, (img, true_idx, pred_idx, conf) in zip(axes, chosen):
        ax.imshow(_denormalize(img))
        correct = true_idx == pred_idx
        ax.set_title(
            f"pred: {class_names[pred_idx]} ({conf:.0%})\nactual: {class_names[true_idx]}",
            color="green" if correct else "red",
            fontsize=9,
        )
        ax.axis("off")

    for ax in axes[len(chosen):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
