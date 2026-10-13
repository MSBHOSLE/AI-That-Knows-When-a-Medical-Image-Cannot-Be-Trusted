"""
Probability calibration module for Medical AI Reliability.
Implements post-hoc Temperature Scaling fitted strictly on validation data,
along with ECE (Expected Calibration Error), MCE, and Brier Score computation.
"""

from typing import Dict, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    """
    Applies post-hoc Temperature Scaling to model logits.
    The scalar temperature parameter T is optimized via L-BFGS to minimize
    Negative Log-Likelihood (NLL) strictly on the held-out validation set.
    """

    def __init__(self, init_temperature: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Scale logits by temperature parameter.
        """
        temperature = self.temperature.clamp(min=1e-3)
        return logits / temperature

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        max_iter: int = 50,
        lr: float = 0.01,
    ) -> float:
        """
        Fit temperature parameter T on validation set logits and labels.

        Args:
            val_logits: Tensor of shape (N, NumClasses).
            val_labels: Tensor of shape (N,) containing true ground-truth targets.

        Returns:
            Optimal temperature value (float).
        """
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(val_logits)
            loss = nll_criterion(scaled_logits, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        optimal_t = float(self.temperature.detach().item())
        return optimal_t

    def calibrate_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Return calibrated softmax probabilities.
        """
        with torch.no_grad():
            scaled = self.forward(logits)
            return F.softmax(scaled, dim=-1)


def compute_calibration_metrics(
    probs: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 15,
) -> Dict[str, Any]:
    """
    Compute calibration metrics: Expected Calibration Error (ECE),
    Maximum Calibration Error (MCE), and Brier Score.

    Args:
        probs: Predicted probability tensor (N, NumClasses).
        labels: True integer labels tensor (N,).
        num_bins: Number of confidence bins.

    Returns:
        Dictionary with ECE, MCE, Brier Score, and reliability diagram bin stats.
    """
    if isinstance(probs, np.ndarray):
        probs = torch.tensor(probs)
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels)

    confidences, predictions = torch.max(probs, dim=-1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean().item()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean().item()
            avg_confidence_in_bin = confidences[in_bin].mean().item()
            gap = abs(avg_confidence_in_bin - accuracy_in_bin)

            ece += gap * prop_in_bin
            mce = max(mce, gap)

            bin_confidences.append(avg_confidence_in_bin)
            bin_accuracies.append(accuracy_in_bin)
            bin_counts.append(int(in_bin.sum().item()))
        else:
            bin_confidences.append(float((bin_lower + bin_upper) / 2))
            bin_accuracies.append(0.0)
            bin_counts.append(0)

    # Compute Brier Score
    num_classes = probs.size(-1)
    one_hot_labels = F.one_hot(labels, num_classes=num_classes).float()
    brier_score = float(torch.mean(torch.sum((probs - one_hot_labels) ** 2, dim=-1)).item())

    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier_score": float(brier_score),
        "bin_confidences": bin_confidences,
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
        "bin_boundaries": bin_boundaries.tolist(),
    }
