"""
Lightweight CNN model for disaster detection in resource-constrained environments.

Architecture uses depthwise separable convolutions (MobileNet-style) to keep
parameter count low while maintaining sufficient accuracy for disaster classification.
"""

from typing import Optional

import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution block: depthwise conv + pointwise conv."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, stride=stride, padding=1,
            groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x


class LightweightDisasterCNN(nn.Module):
    """
    Lightweight CNN for disaster scene classification.

    Classifies images into one of the following categories:
        0 - Cyclone
        1 - Earthquake
        2 - Flood
        3 - Wildfire
        4 - No Disaster

    Designed for resource-constrained devices (edge devices, embedded systems).
    Total parameters: ~320K (configurable via width_multiplier).
    """

    CLASS_NAMES = ["Cyclone", "Earthquake", "Flood", "Wildfire", "No Disaster"]

    def __init__(self, num_classes: int = 5, width_multiplier: float = 1.0, dropout: float = 0.2):
        """
        Args:
            num_classes: Number of output classes.
            width_multiplier: Scale factor for the number of channels (use <1.0 to
                              further reduce model size).
            dropout: Dropout probability before the final classifier.
        """
        super().__init__()

        def ch(n):
            return max(1, int(n * width_multiplier))

        # Initial standard convolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, ch(32), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch(32)),
            nn.ReLU6(inplace=True),
        )

        # Depthwise separable convolution stages
        self.features = nn.Sequential(
            DepthwiseSeparableConv(ch(32),  ch(64),  stride=1),
            DepthwiseSeparableConv(ch(64),  ch(128), stride=2),
            DepthwiseSeparableConv(ch(128), ch(128), stride=1),
            DepthwiseSeparableConv(ch(128), ch(256), stride=2),
            DepthwiseSeparableConv(ch(256), ch(256), stride=1),
            DepthwiseSeparableConv(ch(256), ch(512), stride=2),
            # Lightweight "wide" stage (fewer repetitions than MobileNetV1)
            DepthwiseSeparableConv(ch(512), ch(512), stride=1),
            DepthwiseSeparableConv(ch(512), ch(512), stride=1),
            DepthwiseSeparableConv(ch(512), ch(256), stride=2),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(ch(256), num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(num_classes: int = 5, width_multiplier: float = 1.0,
                checkpoint_path: Optional[str] = None,
                device: str = "cpu") -> LightweightDisasterCNN:
    """
    Convenience factory that builds the model and optionally loads a checkpoint.

    Args:
        num_classes: Number of output classes.
        width_multiplier: Channel scale factor.
        checkpoint_path: Path to a saved ``state_dict`` (``.pth`` file). When
                         provided, weights are loaded into the model.
        device: Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).

    Returns:
        LightweightDisasterCNN ready for inference or continued training.
    """
    model = LightweightDisasterCNN(num_classes=num_classes, width_multiplier=width_multiplier)
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device)
        # Support both raw state_dict and checkpoint dicts saved by train.py
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
    model.to(device)
    return model
