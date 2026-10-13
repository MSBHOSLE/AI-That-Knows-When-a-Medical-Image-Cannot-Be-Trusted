"""
Model architecture for Medical AI Reliability.
Deep Residual Classifier with Monte Carlo Dropout and deep feature extraction.
"""

from typing import Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Standard residual convolutional block with batch normalization and dropout.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_rate)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class TrustworthyMedicalCNN(nn.Module):
    """
    Trustworthy Medical CNN classifier designed for small/medium medical patch images.
    Features:
    1. ResNet backbone with calibrated capacity.
    2. Monte Carlo Dropout enabled at intermediate and linear layers.
    3. Explicit latent feature extraction for Out-of-Distribution (OOD) scoring (Mahalanobis / Energy).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 9,
        base_channels: int = 64,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 28x28 -> 28x28
        self.layer1 = nn.Sequential(
            ResidualBlock(base_channels, base_channels, stride=1, dropout_rate=dropout_rate),
            ResidualBlock(base_channels, base_channels, stride=1, dropout_rate=dropout_rate),
        )

        # Stage 2: 28x28 -> 14x14
        self.layer2 = nn.Sequential(
            ResidualBlock(base_channels, base_channels * 2, stride=2, dropout_rate=dropout_rate),
            ResidualBlock(base_channels * 2, base_channels * 2, stride=1, dropout_rate=dropout_rate),
        )

        # Stage 3: 14x14 -> 7x7
        self.layer3 = nn.Sequential(
            ResidualBlock(base_channels * 2, base_channels * 4, stride=2, dropout_rate=dropout_rate),
            ResidualBlock(base_channels * 4, base_channels * 4, stride=1, dropout_rate=dropout_rate),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = base_channels * 4

        self.fc_dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract latent feature representation before final linear classification head.
        Used for Mahalanobis distance, distance-based OOD, and representation diagnostics.
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)
        features = torch.flatten(x, 1)
        return features

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Standard forward pass.

        Args:
            x: Input tensor (B, C, H, W).
            return_features: If True, returns (logits, features).
        """
        features = self.extract_features(x)
        dropped_features = self.fc_dropout(features)
        logits = self.classifier(dropped_features)

        if return_features:
            return logits, features
        return logits

    def forward_mc_dropout(
        self, x: torch.Tensor, num_samples: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform Monte Carlo Dropout sampling during inference.
        Dropout layers remain active even when in evaluation mode.

        Args:
            x: Input tensor (B, C, H, W).
            num_samples: Number of stochastic forward passes T.

        Returns:
            Tuple of (mean_probs [B, C], variance_probs [B, C]).
        """
        # Force dropout layers into training mode while keeping batchnorm in eval mode
        self.eval()
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()

        batch_size = x.size(0)
        sample_probs = []

        with torch.no_grad():
            for _ in range(num_samples):
                logits = self.forward(x, return_features=False)
                probs = F.softmax(logits, dim=-1)
                sample_probs.append(probs)

        # Shape: (T, B, NumClasses)
        stacked_probs = torch.stack(sample_probs, dim=0)
        mean_probs = torch.mean(stacked_probs, dim=0)
        var_probs = torch.var(stacked_probs, dim=0)

        # Restore evaluation mode
        self.eval()
        return mean_probs, var_probs


def build_model(
    num_classes: int = 9,
    in_channels: int = 3,
    dropout_rate: float = 0.2,
    base_channels: int = 64,
) -> TrustworthyMedicalCNN:
    """
    Factory function for TrustworthyMedicalCNN.
    """
    return TrustworthyMedicalCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        dropout_rate=dropout_rate,
    )
