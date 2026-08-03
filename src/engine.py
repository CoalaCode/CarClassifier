"""Training and evaluation step functions."""
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def train_step(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module,
                optimizer: torch.optim.Optimizer, device: str) -> tuple[float, float]:
    model.train()
    train_loss, train_acc = 0.0, 0.0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(y_pred, dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    return train_loss / len(dataloader), train_acc / len(dataloader)


def eval_step(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, device: str) -> tuple[float, float]:
    """Runs a forward-only pass; used for both validation (during training) and test (once, at the end)."""
    model.eval()
    eval_loss, eval_acc = 0.0, 0.0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred_logits = model(X)
            loss = loss_fn(y_pred_logits, y)
            eval_loss += loss.item()

            y_pred_labels = y_pred_logits.argmax(dim=1)
            eval_acc += (y_pred_labels == y).sum().item() / len(y_pred_labels)

    return eval_loss / len(dataloader), eval_acc / len(dataloader)


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    loss_fn: nn.Module = None,
    scheduler=None,
    epochs: int = 15,
):
    """Trains for `epochs`, validating (never testing) each epoch. Returns per-epoch history."""
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state_dict = None

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        val_loss, val_acc = eval_step(model, val_dataloader, loss_fn, device)

        print(
            f"Epoch: {epoch + 1} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if scheduler is not None:
            old_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]["lr"]
            if old_lr != new_lr:
                print(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")

    return results, best_state_dict
