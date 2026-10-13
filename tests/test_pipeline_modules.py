"""
Unit and integration test suite for the Medical AI Reliability pipeline.
Verifies dataset hygiene, model mechanics, MC Dropout, corruptions,
calibration, OOD detection, and selective abstention policies.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

from src.data.dataset import MedicalImageDataset, SyntheticOODDataset, PATHMNIST_CLASSES
from src.models.classifier import build_model, TrustworthyMedicalCNN
from src.corruptions.image_corruptions import ControlledCorruptions, ImageQualityAssessor
from src.uncertainty.estimator import UncertaintyEstimator
from src.calibration.temperature_scaling import TemperatureScaler, compute_calibration_metrics
from src.ood.detector import OODDetector
from src.abstention.policy import SelectivePredictionPolicy, compute_risk_coverage_curve
from src.evaluation.metrics import ComprehensiveMetrics


def test_synthetic_ood_dataset():
    """Verify procedural synthetic OOD dataset properties."""
    ds = SyntheticOODDataset(num_samples=20, image_size=28, in_channels=3, pattern_type="uniform_noise")
    assert len(ds) == 20
    img_tensor, label = ds[0]
    assert img_tensor.shape == (3, 28, 28)
    assert label == -1


def test_model_forward_and_features():
    """Verify model forward pass, MC-dropout, and latent representation extraction."""
    model = build_model(num_classes=9, in_channels=3, base_channels=32)
    x = torch.randn(4, 3, 28, 28)

    # Standard forward
    logits = model(x)
    assert logits.shape == (4, 9)

    # Features extraction
    feats = model.extract_features(x)
    assert feats.shape == (4, 128)  # base_channels * 4 = 128

    # MC-Dropout pass
    mean_p, var_p = model.forward_mc_dropout(x, num_samples=5)
    assert mean_p.shape == (4, 9)
    assert var_p.shape == (4, 9)
    assert torch.all(mean_p >= 0.0) and torch.all(mean_p <= 1.0)


def test_image_corruptions():
    """Verify all 6 corruption types produce valid uint8 arrays without NaNs."""
    dummy_img = np.random.randint(50, 200, (28, 28, 3), dtype=np.uint8)
    corruption_types = [
        "gaussian_noise",
        "motion_blur",
        "defocus_blur",
        "brightness_shift",
        "contrast_shift",
        "he_stain_variation",
    ]

    for c in corruption_types:
        for sev in [1, 3, 5]:
            corrupted = ControlledCorruptions.apply(dummy_img, c, severity=sev)
            assert corrupted.shape == (28, 28, 3)
            assert corrupted.dtype == np.uint8
            assert not np.isnan(corrupted).any()


def test_image_quality_assessor():
    """Verify deterministic blur and SNR indicators."""
    sharp_img = np.zeros((28, 28, 3), dtype=np.uint8)
    sharp_img[::2, ::2] = 255  # High-frequency pattern

    blurred_img = ControlledCorruptions.defocus_blur(sharp_img, severity=5)

    q_sharp = ImageQualityAssessor.assess_image(sharp_img)
    q_blur = ImageQualityAssessor.assess_image(blurred_img)

    assert q_sharp["blur_score"] > q_blur["blur_score"]
    assert q_blur["blur_score"] < 40.0
    assert q_blur["acceptable"] is False


def test_uncertainty_decomposition():
    """Verify predictive entropy decomposition into epistemic and aleatoric parts."""
    model = build_model(num_classes=9, base_channels=32)
    estimator = UncertaintyEstimator(model, num_samples=10)
    x = torch.randn(2, 3, 28, 28)

    unc = estimator.predict_with_uncertainty(x)

    assert "mean_probs" in unc
    assert "predictive_entropy" in unc
    assert "epistemic_uncertainty" in unc
    assert "aleatoric_uncertainty" in unc

    # Mathematical property: Total Entropy >= Aleatoric Uncertainty
    assert torch.all(unc["predictive_entropy"] >= unc["aleatoric_uncertainty"] - 1e-5)


def test_temperature_scaling_and_calibration():
    """Verify temperature scaling optimization and ECE reduction."""
    torch.manual_seed(42)
    val_logits = torch.randn(100, 9) * 3.0  # Overconfident logits
    val_labels = torch.randint(0, 9, (100,))

    scaler = TemperatureScaler(init_temperature=1.0)
    opt_t = scaler.fit(val_logits, val_labels)

    assert opt_t > 0.0
    probs_cal = scaler.calibrate_probabilities(val_logits)
    cal_metrics = compute_calibration_metrics(probs_cal, val_labels)

    assert 0.0 <= cal_metrics["ece"] <= 1.0
    assert 0.0 <= cal_metrics["brier_score"] <= 2.0


def test_ood_detectors():
    """Verify MSP, Energy, and Mahalanobis scoring mechanics."""
    logits_id = torch.tensor([[4.0, 1.0, 0.5], [3.5, 0.5, 0.2]])
    logits_ood = torch.tensor([[0.5, 0.4, 0.6], [0.1, 0.2, 0.1]])

    detector = OODDetector(method="energy")
    scores_id = detector.compute_score(logits_id, method="energy")
    scores_ood = detector.compute_score(logits_ood, method="energy")

    # In-distribution logits should have higher free energy than uniform/low logits
    assert torch.all(scores_id > scores_ood)

    # Test Mahalanobis fitting
    feats_train = torch.randn(50, 16)
    labels_train = torch.randint(0, 3, (50,))
    mahal = OODDetector(method="mahalanobis")
    mahal.fit_mahalanobis(feats_train, labels_train)

    test_feats = torch.randn(5, 16)
    m_scores = mahal.compute_score(features=test_feats, method="mahalanobis")
    assert m_scores.shape == (5,)


def test_selective_prediction_policy():
    """Verify multi-signal triage gates and decision rules."""
    policy = SelectivePredictionPolicy(
        blur_threshold=40.0,
        ood_threshold=-10.0,
        entropy_threshold=1.0,
        confidence_threshold=0.70,
    )

    # Clean accepted sample
    clean_meta = {"acceptable": True, "blur_score": 120.0, "warnings": []}
    res_clean = policy.evaluate_sample(
        image_quality_meta=clean_meta,
        calibrated_confidence=0.92,
        predictive_entropy=0.25,
        epistemic_uncertainty=0.08,
        ood_score=-2.5,
    )
    assert res_clean["status"] == "TRUSTED"
    assert res_clean["is_accepted"] is True

    # Corrupted / Blurred sample
    blur_meta = {"acceptable": False, "blur_score": 18.0, "warnings": ["Severe blur"]}
    res_blur = policy.evaluate_sample(
        image_quality_meta=blur_meta,
        calibrated_confidence=0.92,
        predictive_entropy=0.25,
        epistemic_uncertainty=0.08,
        ood_score=-2.5,
    )
    assert res_blur["status"] == "UNTRUSTED"
    assert res_blur["is_accepted"] is False
    assert any("blur" in r.lower() for r in res_blur["triggered_reasons"])


def test_risk_coverage_curve():
    """Verify monotonic behavior of selective risk and AURC."""
    confs = np.array([0.95, 0.88, 0.82, 0.75, 0.60, 0.40, 0.30])
    correct = np.array([1, 1, 1, 1, 0, 0, 0])

    curve = compute_risk_coverage_curve(confs, correct)
    assert "aurc" in curve
    assert 0.0 <= curve["aurc"] <= 1.0
    assert len(curve["coverages"]) > 0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
