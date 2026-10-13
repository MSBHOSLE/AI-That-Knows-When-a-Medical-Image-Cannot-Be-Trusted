"""
Trustworthy Clinical Decision and Selective Prediction module.
Implements multi-signal abstention policy (quality gate, OOD score, uncertainty, calibrated confidence),
validation-based threshold selection, and selective risk-coverage curves.
"""

from typing import Dict, Tuple, Optional, Any, List
import numpy as np
import torch


class SelectivePredictionPolicy:
    """
    Multi-signal triage policy determining whether an AI prediction is reliable
    or must be deferred to human clinical review.
    """

    def __init__(
        self,
        blur_threshold: float = 40.0,
        ood_threshold: Optional[float] = None,
        entropy_threshold: Optional[float] = None,
        epistemic_threshold: Optional[float] = None,
        confidence_threshold: float = 0.70,
    ):
        """
        Thresholds for deferral. If any threshold triggers, the system abstains.
        """
        self.blur_threshold = blur_threshold
        self.ood_threshold = ood_threshold
        self.entropy_threshold = entropy_threshold
        self.epistemic_threshold = epistemic_threshold
        self.confidence_threshold = confidence_threshold

    def fit_thresholds_on_validation(
        self,
        val_confidences: np.ndarray,
        val_entropies: np.ndarray,
        val_ood_scores: np.ndarray,
        val_correct: np.ndarray,
        target_accuracy_on_accepted: float = 0.95,
        percentile_ood_reject: float = 5.0,
    ):
        """
        Determine operational thresholds strictly on the validation partition.
        Never touches test data.
        """
        # Set OOD threshold at the 5th percentile of validation ID scores (95% ID pass rate)
        if len(val_ood_scores) > 0:
            self.ood_threshold = float(np.percentile(val_ood_scores, percentile_ood_reject))

        # Sweep confidence threshold to achieve target accuracy on accepted samples
        sorted_indices = np.argsort(val_confidences)
        best_conf_thresh = 0.50
        for idx in sorted_indices:
            thresh = val_confidences[idx]
            accepted_mask = val_confidences >= thresh
            if np.sum(accepted_mask) > 50:
                acc_accepted = np.mean(val_correct[accepted_mask])
                if acc_accepted >= target_accuracy_on_accepted:
                    best_conf_thresh = thresh
                    break

        self.confidence_threshold = float(best_conf_thresh)
        # Set entropy threshold at 90th percentile of correct samples
        correct_entropies = val_entropies[val_correct == 1]
        if len(correct_entropies) > 0:
            self.entropy_threshold = float(np.percentile(correct_entropies, 95.0))

    def evaluate_sample(
        self,
        image_quality_meta: Dict[str, Any],
        calibrated_confidence: float,
        predictive_entropy: float,
        epistemic_uncertainty: float,
        ood_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single inference sample against the safety triage policy.
        """
        triggered_reasons = []

        # 1. Pre-inference Image Quality Gate
        if not image_quality_meta.get("acceptable", True):
            for w in image_quality_meta.get("warnings", []):
                triggered_reasons.append(f"Image Quality Warning: {w}")

        blur_val = image_quality_meta.get("blur_score", 100.0)
        if blur_val < self.blur_threshold:
            triggered_reasons.append(
                f"Severe blur detected (Laplacian Var {blur_val:.1f} < {self.blur_threshold:.1f})"
            )

        # 2. Out-of-Distribution Gate
        if self.ood_threshold is not None and ood_score is not None:
            if ood_score < self.ood_threshold:
                triggered_reasons.append(
                    f"Out-of-Distribution sample (OOD Score {ood_score:.2f} < {self.ood_threshold:.2f})"
                )

        # 3. Epistemic Uncertainty Gate
        if self.epistemic_threshold is not None:
            if epistemic_uncertainty > self.epistemic_threshold:
                triggered_reasons.append(
                    f"High epistemic uncertainty (Mutual Info {epistemic_uncertainty:.3f} > {self.epistemic_threshold:.3f})"
                )

        # 4. Total Predictive Entropy Gate
        if self.entropy_threshold is not None:
            if predictive_entropy > self.entropy_threshold:
                triggered_reasons.append(
                    f"High predictive entropy ({predictive_entropy:.3f} > {self.entropy_threshold:.3f})"
                )

        # 5. Calibrated Confidence Gate
        if calibrated_confidence < self.confidence_threshold:
            triggered_reasons.append(
                f"Low calibrated confidence ({calibrated_confidence * 100:.1f}% < {self.confidence_threshold * 100:.1f}%)"
            )

        # Decision
        if len(triggered_reasons) > 0:
            status = "UNTRUSTED"
            action = "DEFER_TO_HUMAN_PATHOLOGIST"
            is_accepted = False
        else:
            status = "TRUSTED"
            action = "PROCEED_WITH_DIAGNOSIS"
            is_accepted = True

        return {
            "status": status,
            "action": action,
            "is_accepted": is_accepted,
            "triggered_reasons": triggered_reasons,
            "telemetry": {
                "calibrated_confidence": float(calibrated_confidence),
                "predictive_entropy": float(predictive_entropy),
                "epistemic_uncertainty": float(epistemic_uncertainty),
                "ood_score": float(ood_score) if ood_score is not None else None,
                "blur_score": float(blur_val),
            },
        }


def compute_risk_coverage_curve(
    confidences: np.ndarray,
    is_correct: np.ndarray,
    num_thresholds: int = 100,
) -> Dict[str, Any]:
    """
    Compute selective prediction risk vs coverage tradeoff curve.
    Coverage = Fraction of samples accepted.
    Selective Risk = Error rate among accepted samples (1 - Accuracy_accepted).

    Args:
        confidences: Array of selection scores / confidence in [0, 1].
        is_correct: Binary array (1 for correct prediction, 0 for error).
        num_thresholds: Number of sweep points.

    Returns:
        Dictionary containing coverages, selective_risks, selective_accuracies,
        AURC (Area Under Risk-Coverage Curve), and optimal AURC.
    """
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    coverages = []
    selective_risks = []
    selective_accuracies = []

    n_total = len(is_correct)

    for thresh in thresholds:
        accepted = confidences >= thresh
        n_accepted = np.sum(accepted)

        if n_accepted > 0:
            cov = n_accepted / n_total
            acc = np.mean(is_correct[accepted])
            risk = 1.0 - acc
        else:
            cov = 0.0
            acc = 1.0
            risk = 0.0

        coverages.append(cov)
        selective_risks.append(risk)
        selective_accuracies.append(acc)

    # Sort by coverage ascending for integration
    coverages = np.array(coverages)
    selective_risks = np.array(selective_risks)
    selective_accuracies = np.array(selective_accuracies)

    sort_idx = np.argsort(coverages)
    sorted_cov = coverages[sort_idx]
    sorted_risk = selective_risks[sort_idx]

    # AURC via trapezoidal rule (compatible with NumPy 1.x and 2.x)
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    aurc = float(trapz_fn(sorted_risk, sorted_cov))

    # Optimal theoretical AURC (if perfect rejection of all mistakes first)
    total_errors = np.sum(1 - is_correct)
    total_samples = len(is_correct)
    clean_acc = np.mean(is_correct)
    
    # Excess AURC
    return {
        "coverages": coverages.tolist(),
        "selective_risks": selective_risks.tolist(),
        "selective_accuracies": selective_accuracies.tolist(),
        "aurc": aurc,
        "clean_accuracy": float(clean_acc),
    }
