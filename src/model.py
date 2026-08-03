"""Model definitions: a from-scratch CNN and a fine-tuned ResNet18 baseline."""
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ImprovedCarClassifier(nn.Module):
    """From-scratch CNN: 3 conv blocks (BatchNorm + ReLU + MaxPool) then a small MLP head."""

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, input_size: int = 128) -> None:
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units * 2),
            nn.ReLU(),
            nn.Conv2d(hidden_units * 2, hidden_units * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(hidden_units * 2, hidden_units * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units * 4),
            nn.ReLU(),
            nn.Conv2d(hidden_units * 4, hidden_units * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units * 4),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        conv_output_size = input_size // 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units * 4 * conv_output_size * conv_output_size, hidden_units * 8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_units * 8, output_shape),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        return self.classifier(x)


def build_resnet18(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """ImageNet-pretrained ResNet18 with a new classification head.

    freeze_backbone=True only trains the new final layer (fast, good for a small
    dataset); set False to fine-tune the whole network.
    """
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model(name: str, num_classes: int, input_size: int = 128, freeze_backbone: bool = False) -> nn.Module:
    if name == "custom":
        torch.manual_seed(42)
        return ImprovedCarClassifier(input_shape=3, hidden_units=32, output_shape=num_classes, input_size=input_size)
    if name == "resnet18":
        return build_resnet18(num_classes=num_classes, freeze_backbone=freeze_backbone)
    raise ValueError(f"Unknown model name: {name!r} (expected 'custom' or 'resnet18')")
