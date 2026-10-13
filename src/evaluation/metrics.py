"""
Scientific evaluation and failure-analysis metrics module.
Computes accuracy, macro-F1, error-detection AUROC, silent-failure analysis,
and selective prediction risk metrics.
"""

from typing import Dict, Tuple, Optional, Any, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
)


class ComprehensiveMetrics:
    """
    Computes diagnostic performance, uncertainty-error alignment,
    and silent failure quantification.
    """

    @staticmethod
    def classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute standard multiclass performance metrics.
        """
        acc = float(accuracy_score(y_true, y_pred))
        bacc = float(balanced_accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

        return {
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "macro_f1": macro_f1,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
        }

    @staticmethod
    def error_detection_auroc(
        is_correct: np.ndarray,
        uncertainty_scores: np.ndarray,
    ) -> float:
        """
        Evaluate how well an uncertainty metric identifies mistakes (AUROC of error detection).
        Error label: 1 if prediction is wrong (is_correct == 0), 0 if correct.
        Score: Uncertainty (higher score should ideally predict an error).
        """
        errors = (is_correct == 0).astype(int)
        if len(np.unique(errors)) < 2:
            return 0.5  # Only one class present (e.g. 100% correct or 0% correct)

        valid = np.isfinite(uncertainty_scores)
        try:
            auroc = float(roc_auc_score(errors[valid], uncertainty_scores[valid]))
        except Exception:
            auroc = 0.5
        return auroc

    @staticmethod
    def identify_silent_failures(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidences: np.ndarray,
        entropies: np.ndarray,
        confidence_cutoff: float = 0.85,
        entropy_cutoff: float = 0.35,
    ) -> Dict[str, Any]:
        """
        Identify hazardous 'silent failures':
        Samples where the model is INCORRECT, but expresses HIGH confidence and LOW entropy.
        """
        is_error = y_true != y_pred
        confident_mask = confidences >= confidence_cutoff
        low_entropy_mask = entropies <= entropy_cutoff

        silent_failure_mask = is_error & confident_mask & low_entropy_mask
        silent_count = int(np.sum(silent_failure_mask))
        total_errors = int(np.sum(is_error))
        total_samples = len(y_true)

        fraction_of_errors = float(silent_count / max(total_errors, 1))
        fraction_of_total = float(silent_count / max(total_samples, 1))

        silent_indices = np.where(silent_failure_mask)[0].tolist()

        return {
            "silent_failure_count": silent_count,
            "total_errors": total_errors,
            "silent_fraction_of_errors": fraction_of_errors,
            "silent_fraction_of_dataset": fraction_of_total,
            "silent_sample_indices": silent_indices[:50],  # sample list
        }
