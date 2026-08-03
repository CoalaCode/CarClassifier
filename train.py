"""Train a car-brand classifier and produce evaluation artifacts.

Examples:
    python train.py --model custom --epochs 15
    python train.py --model resnet18 --epochs 10
"""
import argparse
import multiprocessing
import warnings
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import torch
from torch import nn

from src.data import get_dataloaders
from src.engine import train
from src.evaluate import (evaluate_on_test, plot_confusion_matrix,
                            plot_sample_predictions, plot_training_curves,
                            write_classification_report)
from src.model import build_model

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["custom", "resnet18"], default="custom")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--freeze-backbone", action="store_true",
                    help="resnet18 only: freeze pretrained conv layers, train just the new head")
    p.add_argument("--data-dir", type=Path, default=Path("Cars"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return p.parse_args()


def class_weights_from_loader(train_loader, num_classes, device):
    counts = np.zeros(num_classes)
    for _, y in train_loader:
        for label in y.tolist():
            counts[label] += 1
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if not (args.data_dir / "train").exists() or not (args.data_dir / "test").exists():
        print(f"Error: expected '{args.data_dir}/train' and '{args.data_dir}/test'. "
              f"Run crawler.py first to build the dataset (see README).")
        return

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        cars_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
    )
    print(f"Classes: {class_names}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    model = build_model(args.model, num_classes=len(class_names), input_size=args.image_size,
                        freeze_backbone=args.freeze_backbone).to(device)

    try:
        from torchinfo import summary
        summary(model, input_size=(1, 3, args.image_size, args.image_size))
    except ImportError:
        pass
    except UnicodeEncodeError:
        # Some Windows terminals (cp1252) can't render torchinfo's box-drawing characters.
        print(f"Model: {args.model} ({sum(p.numel() for p in model.parameters()):,} parameters)")

    # Mild class imbalance across brands (see README) -> weight the loss instead of resampling.
    weights = class_weights_from_loader(train_loader, len(class_names), device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    start_time = timer()
    results, best_state_dict = train(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        device=device,
        loss_fn=loss_fn,
        scheduler=scheduler,
        epochs=args.epochs,
    )
    print(f"Total training time: {timer() - start_time:.1f}s")

    # Evaluate the best-val-loss checkpoint (not necessarily the last epoch) on the held-out test set.
    model.load_state_dict(best_state_dict)
    checkpoint_path = args.output_dir / f"{args.model}_best.pth"
    torch.save(best_state_dict, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

    y_true, y_pred, test_acc = evaluate_on_test(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.4f}")

    plot_training_curves(results, args.output_dir / f"{args.model}_curves.png")
    plot_confusion_matrix(y_true, y_pred, class_names, args.output_dir / f"{args.model}_confusion_matrix.png")
    plot_sample_predictions(model, test_loader, class_names, device, args.output_dir / f"{args.model}_sample_predictions.png")
    report = write_classification_report(y_true, y_pred, class_names, args.output_dir / f"{args.model}_classification_report.txt")
    print(report)
    print(f"All evaluation artifacts written to {args.output_dir}/")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
