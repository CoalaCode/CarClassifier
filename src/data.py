"""Dataset loading, transforms, and train/val/test dataloaders."""
import platform
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def safe_convert_to_rgb(img: Image.Image) -> Image.Image:
    """Convert image to RGB, compositing transparency onto a white background."""
    if img.mode in ("P", "PA"):
        img = img.convert("RGBA")
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        return background
    return img.convert("RGB")


def preserve_aspect_ratio_resize(img: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """Resize preserving aspect ratio, then pad (white) to target_size."""
    width, height = img.size
    target_width, target_height = target_size
    scale = min(target_width / width, target_height / height)
    new_width, new_height = int(width * scale), int(height * scale)

    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    img_final = Image.new("RGB", target_size, (255, 255, 255))
    left = (target_width - new_width) // 2
    top = (target_height - new_height) // 2
    img_final.paste(img_resized, (left, top))
    return img_final


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    """Single source of truth for preprocessing. train=True adds augmentation."""
    ops = [
        transforms.Lambda(safe_convert_to_rgb),
        transforms.Lambda(lambda img: preserve_aspect_ratio_resize(img, (image_size, image_size))),
    ]
    if train:
        ops.append(transforms.TrivialAugmentWide(num_magnitude_bins=31))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(ops)


def _num_workers() -> int:
    # Windows can't fork worker processes for DataLoader; keep it single-process there.
    if platform.system() == "Windows":
        return 0
    import os
    return min(4, os.cpu_count() or 1)


def get_dataloaders(
    cars_dir: Path,
    image_size: int = 128,
    batch_size: int = 32,
    val_split: float = 0.15,
    seed: int = 42,
):
    """Load Cars/train (split into train+val) and Cars/test (held out, untouched).

    Returns (train_loader, val_loader, test_loader, class_names).
    """
    cars_dir = Path(cars_dir)
    train_dir, test_dir = cars_dir / "train", cars_dir / "test"

    train_transform = build_transforms(image_size, train=True)
    eval_transform = build_transforms(image_size, train=False)

    # ImageFolder classes = brand subfolders; images are found recursively within
    # each brand folder (so brand/model/*.jpg still gets labeled by brand).
    full_train = datasets.ImageFolder(root=train_dir, transform=train_transform)
    # A second view of the same files with eval-time transforms, so the validation
    # split doesn't get train-time augmentation.
    full_train_eval = datasets.ImageFolder(root=train_dir, transform=eval_transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=eval_transform)

    class_names = full_train.classes
    n_val = int(len(full_train) * val_split)
    n_train = len(full_train) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(
        range(len(full_train)), [n_train, n_val], generator=generator
    )

    train_data = torch.utils.data.Subset(full_train, train_indices.indices)
    val_data = torch.utils.data.Subset(full_train_eval, val_indices.indices)

    num_workers = _num_workers()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_names
