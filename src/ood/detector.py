"""
Out-of-Distribution (OOD) detection module for Medical AI Reliability.
Implements:
1. Maximum Softmax Probability (MSP baseline)
2. Energy-based OOD scoring (Liu et al., NeurIPS 2020)
3. Deep Feature Mahalanobis distance (Lee et al., NeurIPS 2018)
Along with AUROC, AUPRC, and FPR95 evaluation routines.
"""

from typing import Dict, Tuple, Optional, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


class OODDetector:
    """
    Multi-method Out-of-Distribution detector.
    Computes scores indicating how likely an input comes from the in-distribution training domain.
    Convention: HIGHER score = MORE IN-DISTRIBUTION (more trusted).
    """

    def __init__(self, method: str = "energy", temperature: float = 1.0):
        """
        Args:
            method: One of ['msp', 'energy', 'mahalanobis'].
            temperature: Softmax/Energy temperature scaling parameter.
        """
        self.method = method.lower()
        self.temperature = temperature

        # Mahalanobis parameters
        self.class_means = None  # (C, D)
        self.precision_matrix = None  # (D, D)
        self.num_classes = None

    def fit_mahalanobis(
        self, features: torch.Tensor, labels: torch.Tensor, eps: float = 1e-4
    ):
        """
        Fit class-conditional Gaussian distributions on deep feature representations
        from the training or validation partition.

        Args:
            features: Tensor of shape (N, FeatureDim).
            labels: Tensor of shape (N,) with ground-truth class indices.
            eps: Regularization constant for covariance inversion.
        """
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=torch.float32)
        if isinstance(labels, np.ndarray):
            labels = torch.tensor(labels, dtype=torch.int64)

        features = features.cpu().float()
        labels = labels.cpu().long()

        num_classes = int(labels.max().item()) + 1
        feature_dim = features.size(1)
        self.num_classes = num_classes

        # Compute empirical class means
        means = []
        for c in range(num_classes):
            c_mask = labels == c
            if c_mask.sum() > 0:
                mean_c = features[c_mask].mean(dim=0)
            else:
                mean_c = torch.zeros(feature_dim)
            means.append(mean_c)
        self.class_means = torch.stack(means, dim=0)  # (C, D)

        # Compute tied empirical covariance matrix
        centered_features = torch.zeros_like(features)
        for c in range(num_classes):
            c_mask = labels == c
            if c_mask.sum() > 0:
                centered_features[c_mask] = features[c_mask] - self.class_means[c]

        cov = torch.mm(centered_features.t(), centered_features) / features.size(0)
        # Add diagonal shrinkage for stability
        cov += torch.eye(feature_dim) * eps
        self.precision_matrix = torch.linalg.pinv(cov)

    def compute_score(
        self,
        logits: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
        method: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Compute in-distribution trust score for a batch of samples.
        Higher score = more in-distribution.
        """
        target_method = (method or self.method).lower()

        if target_method == "msp":
            if logits is None:
                raise ValueError("Logits required for MSP scoring.")
            probs = F.softmax(logits, dim=-1)
            scores, _ = torch.max(probs, dim=-1)
            return scores

        elif target_method == "energy":
            if logits is None:
                raise ValueError("Logits required for Energy scoring.")
            # Free energy: S(x) = T * log(sum(exp(logits / T)))
            scaled_logits = logits / self.temperature
            energy_scores = self.temperature * torch.logsumexp(scaled_logits, dim=-1)
            return energy_scores

        elif target_method == "mahalanobis":
            if features is None:
                raise ValueError("Features required for Mahalanobis scoring.")
            if self.class_means is None or self.precision_matrix is None:
                raise RuntimeError("Mahalanobis detector has not been fitted yet! Call fit_mahalanobis first.")

            # Compute min distance to any class centroid: -min_c (f - mu_c)^T Sigma^{-1} (f - mu_c)
            # Higher (less negative) = closer to distribution
            batch_size = features.size(0)
            dist_list = []
            features_cpu = features.cpu()
            means_cpu = self.class_means.cpu()
            precision_cpu = self.precision_matrix.cpu()

            for c in range(self.num_classes):
                diff = features_cpu - means_cpu[c]  # (B, D)
                term = torch.mm(diff, precision_cpu)  # (B, D)
                dist_c = torch.sum(term * diff, dim=-1)  # (B,)
                dist_list.append(dist_c)

            stacked_dists = torch.stack(dist_list, dim=1)  # (B, C)
            min_dists, _ = torch.min(stacked_dists, dim=1)  # (B,)
            scores = -min_dists  # Negative distance so higher is more in-distribution
            return scores.to(features.device)

        else:
            raise ValueError(f"Unknown OOD method: {target_method}")

    @staticmethod
    def evaluate_ood(
        id_scores: np.ndarray,
        ood_scores: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate OOD detection performance distinguishing ID from OOD samples.
        ID samples have positive label 1, OOD samples have label 0.

        Args:
            id_scores: Array of trust scores for in-distribution test samples.
            ood_scores: Array of trust scores for out-of-distribution test samples.

        Returns:
            Dictionary with AUROC, AUPRC, and FPR@95% TPR.
        """
        labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
        scores = np.concatenate([id_scores, ood_scores])

        # Filter NaNs or Infs
        valid = np.isfinite(scores)
        labels = labels[valid]
        scores = scores[valid]

        auroc = float(roc_auc_score(labels, scores))
        auprc = float(average_precision_score(labels, scores))

        # Compute FPR at 95% TPR
        fpr, tpr, thresholds = roc_curve(labels, scores)
        # Find threshold where TPR >= 0.95
        idx = np.where(tpr >= 0.95)[0]
        if len(idx) > 0:
            fpr95 = float(fpr[idx[0]])
        else:
            fpr95 = 1.0

        return {
            "auroc": auroc,
            "auprc": auprc,
            "fpr95": fpr95,
        }
