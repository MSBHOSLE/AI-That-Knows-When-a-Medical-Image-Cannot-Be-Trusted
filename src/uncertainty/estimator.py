"""
Uncertainty quantification module for Medical AI Reliability.
Implements Monte Carlo Dropout uncertainty estimation and information-theoretic
decomposition into Aleatoric (data) and Epistemic (model) uncertainty.
"""

from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyEstimator:
    """
    Estimates predictive uncertainty using Monte Carlo Dropout.
    Decomposes total uncertainty into Epistemic (Mutual Information)
    and Aleatoric (Expected Entropy) components.
    """

    def __init__(self, model: nn.Module, num_samples: int = 20, eps: float = 1e-8):
        """
        Args:
            model: PyTorch classifier module.
            num_samples: Number of stochastic forward passes T.
            eps: Small numerical stability constant for log calculations.
        """
        self.model = model
        self.num_samples = num_samples
        self.eps = eps

    def predict_with_uncertainty(
        self, x: torch.Tensor, device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Perform stochastic forward passes with dropout enabled.

        Args:
            x: Input tensor batch of shape (B, C, H, W).
            device: Target execution device.

        Returns:
            Dictionary containing:
                - 'mean_probs': Mean probability distribution (B, NumClasses).
                - 'predicted_class': Argmax class prediction (B,).
                - 'confidence': Max mean softmax probability (B,).
                - 'predictive_entropy': Total uncertainty H(p_bar) (B,).
                - 'aleatoric_uncertainty': Expected sample entropy E[H(p)] (B,).
                - 'epistemic_uncertainty': Mutual information I(y, W | x) (B,).
                - 'variation_ratio': Dispersion of predictions across passes (B,).
                - 'variance': Mean variance across classes (B,).
        """
        if device is not None:
            x = x.to(device)

        # Set dropout layers into active training mode, keep batchnorm frozen in eval mode
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()

        sample_probs = []
        sample_preds = []

        with torch.no_grad():
            for _ in range(self.num_samples):
                logits = self.model(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
                probs = F.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                sample_probs.append(probs)
                sample_preds.append(preds)

        # Re-freeze model
        self.model.eval()

        # Stacked shapes: (T, B, C) and (T, B)
        stacked_probs = torch.stack(sample_probs, dim=0)
        stacked_preds = torch.stack(sample_preds, dim=0)

        # 1. Mean Predictive Probability
        mean_probs = torch.mean(stacked_probs, dim=0)  # (B, C)
        confidences, predicted_classes = torch.max(mean_probs, dim=-1)

        # 2. Total Uncertainty: Predictive Entropy H(p_bar)
        predictive_entropy = -torch.sum(
            mean_probs * torch.log(mean_probs + self.eps), dim=-1
        )  # (B,)

        # 3. Aleatoric Uncertainty: Expected Entropy E_t[H(p_t)]
        sample_entropies = -torch.sum(
            stacked_probs * torch.log(stacked_probs + self.eps), dim=-1
        )  # (T, B)
        aleatoric_uncertainty = torch.mean(sample_entropies, dim=0)  # (B,)

        # 4. Epistemic Uncertainty: Mutual Information I(y, W | x) = H(p_bar) - E[H(p)]
        epistemic_uncertainty = torch.clamp(
            predictive_entropy - aleatoric_uncertainty, min=0.0
        )  # (B,)

        # 5. Variation Ratio: 1 - fraction of modal vote
        mode_counts = []
        for b in range(x.size(0)):
            preds_b = stacked_preds[:, b]
            mode_count = torch.max(torch.bincount(preds_b, minlength=mean_probs.size(-1)))
            mode_counts.append(mode_count)
        mode_counts_tensor = torch.tensor(mode_counts, device=x.device, dtype=torch.float32)
        variation_ratio = 1.0 - (mode_counts_tensor / self.num_samples)

        # 6. Variance across MC passes
        variance = torch.mean(torch.var(stacked_probs, dim=0), dim=-1)  # (B,)

        return {
            "mean_probs": mean_probs,
            "predicted_class": predicted_classes,
            "confidence": confidences,
            "predictive_entropy": predictive_entropy,
            "aleatoric_uncertainty": aleatoric_uncertainty,
            "epistemic_uncertainty": epistemic_uncertainty,
            "variation_ratio": variation_ratio,
            "variance": variance,
        }
