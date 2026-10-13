#!/usr/bin/env python3
"""
Baseline evaluation and calibration script.
Fits temperature scaling and selective thresholds strictly on validation data,
then evaluates calibration, MC-Dropout uncertainty, and risk-coverage on locked test data.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Ensure src modules are resolvable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import load_pathmnist_datasets
from src.models.classifier import build_model
from src.calibration.temperature_scaling import TemperatureScaler, compute_calibration_metrics
from src.uncertainty.estimator import UncertaintyEstimator
from src.abstention.policy import SelectivePredictionPolicy, compute_risk_coverage_curve
from src.evaluation.metrics import ComprehensiveMetrics
from src.evaluation.visualizer import ReliabilityVisualizer


def run_evaluation(
    checkpoint_path: str = "experiments/checkpoints/best_model.pt",
    data_dir: str = "data/raw",
    results_dir: str = "experiments/results",
    reports_dir: str = "reports/figures",
    num_mc_samples: int = 15,
    device_name: str = "auto",
):
    print("=" * 70)
    print("  [Step 3/5] Validation Calibration & Locked Test Evaluation")
    print("=" * 70)

    if device_name == "auto":
        device = torch.device(
            "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        )
    else:
        device = torch.device(device_name)
    print(f"[*] Execution device: {device}")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Load trained checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}. Run scripts/train.py first!")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(
        num_classes=ckpt.get("num_classes", 9),
        dropout_rate=ckpt.get("dropout_rate", 0.2),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[✓] Loaded model checkpoint from: {checkpoint_path} (Val Acc: {ckpt.get('val_acc', 0.0):.2f}%)")

    # 2. Load Datasets
    print("[*] Loading PathMNIST validation and locked test partitions...")
    datasets = load_pathmnist_datasets(data_dir=data_dir, download=True)
    val_loader = DataLoader(datasets["val"], batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(datasets["test"], batch_size=128, shuffle=False, num_workers=0)

    # 3. Validation Phase: Fit Temperature Scaler
    print("\n[*] Processing validation partition (fitting calibration temperature T)...")
    val_logits_list = []
    val_labels_list = []
    val_features_list = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits, feats = model(images, return_features=True)
            val_logits_list.append(logits.cpu())
            val_features_list.append(feats.cpu())
            val_labels_list.append(labels)

    val_logits = torch.cat(val_logits_list, dim=0)
    val_features = torch.cat(val_features_list, dim=0)
    val_labels = torch.cat(val_labels_list, dim=0)

    scaler = TemperatureScaler().to(device)
    optimal_t = scaler.fit(val_logits.to(device), val_labels.to(device))
    print(f"    - Optimized Temperature T: {optimal_t:.4f}")

    # Compute uncalibrated vs calibrated validation ECE
    val_probs_uncal = torch.softmax(val_logits, dim=-1)
    val_probs_cal = scaler.calibrate_probabilities(val_logits.to(device)).cpu()

    val_ece_uncal = compute_calibration_metrics(val_probs_uncal, val_labels)
    val_ece_cal = compute_calibration_metrics(val_probs_cal, val_labels)
    print(f"    - Validation ECE: {val_ece_uncal['ece'] * 100:.2f}% -> {val_ece_cal['ece'] * 100:.2f}% (Calibrated)")

    # 4. Locked Test Set Evaluation
    print("\n[*] Evaluating on LOCKED UNTOUCHED TEST SET (7,180 images)...")
    uncertainty_estimator = UncertaintyEstimator(model, num_samples=num_mc_samples)

    test_raw_logits = []
    test_labels_list = []
    test_preds_list = []
    test_confs_list = []
    test_entropies_list = []
    test_epistemic_list = []

    for images, labels in test_loader:
        images = images.to(device)
        # 1. Forward raw logits for calibration
        with torch.no_grad():
            logits = model(images)
            test_raw_logits.append(logits.cpu())

        # 2. MC Dropout Uncertainty
        unc_res = uncertainty_estimator.predict_with_uncertainty(images)
        test_labels_list.append(labels.numpy())
        test_preds_list.append(unc_res["predicted_class"].cpu().numpy())
        test_confs_list.append(unc_res["confidence"].cpu().numpy())
        test_entropies_list.append(unc_res["predictive_entropy"].cpu().numpy())
        test_epistemic_list.append(unc_res["epistemic_uncertainty"].cpu().numpy())

    all_test_logits = torch.cat(test_raw_logits, dim=0)
    y_true = np.concatenate(test_labels_list)
    y_pred = np.concatenate(test_preds_list)
    confidences = np.concatenate(test_confs_list)
    entropies = np.concatenate(test_entropies_list)
    epistemic_unc = np.concatenate(test_epistemic_list)
    is_correct = (y_true == y_pred).astype(int)

    # Calibrated probabilities on test set
    test_probs_cal = scaler.calibrate_probabilities(all_test_logits.to(device)).cpu()
    test_probs_uncal = torch.softmax(all_test_logits, dim=-1)

    test_cal_metrics_uncal = compute_calibration_metrics(test_probs_uncal, torch.tensor(y_true))
    test_cal_metrics_cal = compute_calibration_metrics(test_probs_cal, torch.tensor(y_true))

    # 5. Scientific Metrics
    clf_metrics = ComprehensiveMetrics.classification_metrics(y_true, y_pred)
    error_auroc = ComprehensiveMetrics.error_detection_auroc(is_correct, entropies)
    silent_failures = ComprehensiveMetrics.identify_silent_failures(
        y_true, y_pred, confidences, entropies
    )
    risk_cov_data = compute_risk_coverage_curve(confidences, is_correct)

    print("\n" + "=" * 70)
    print("  LOCKED TEST SET BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  • Test Accuracy:               {clf_metrics['accuracy'] * 100:.2f}%")
    print(f"  • Balanced Accuracy:           {clf_metrics['balanced_accuracy'] * 100:.2f}%")
    print(f"  • Macro-F1 Score:              {clf_metrics['macro_f1']:.4f}")
    print(f"  • Pre-calibration ECE:         {test_cal_metrics_uncal['ece'] * 100:.2f}%")
    print(f"  • Calibrated ECE (T={optimal_t:.2f}):      {test_cal_metrics_cal['ece'] * 100:.2f}%")
    print(f"  • Error Detection AUROC:       {error_auroc:.4f}")
    print(f"  • Area Under Risk-Coverage:    {risk_cov_data['aurc'] * 100:.2f}")
    print(f"  • Silent Failures (Unwarned):  {silent_failures['silent_failure_count']} ({silent_failures['silent_fraction_of_errors'] * 100:.1f}% of errors)")
    print("=" * 70)

    # 6. Render Publication Figures
    visualizer = ReliabilityVisualizer(output_dir=reports_dir)
    fig1 = visualizer.plot_calibration_comparison(test_cal_metrics_uncal, test_cal_metrics_cal)
    fig2 = visualizer.plot_risk_coverage(risk_cov_data)
    print(f"[✓] Saved Calibration Reliability Figure to: {fig1}")
    print(f"[✓] Saved Risk-Coverage Curve to:           {fig2}")

    # 7. Save serialized evaluation metrics
    results_payload = {
        "classification": clf_metrics,
        "calibration": {
            "optimal_temperature": optimal_t,
            "uncalibrated_ece": test_cal_metrics_uncal["ece"],
            "calibrated_ece": test_cal_metrics_cal["ece"],
            "calibrated_brier": test_cal_metrics_cal["brier_score"],
        },
        "uncertainty": {
            "error_detection_auroc": error_auroc,
            "mean_predictive_entropy": float(np.mean(entropies)),
            "mean_epistemic_uncertainty": float(np.mean(epistemic_unc)),
            "silent_failures": silent_failures,
        },
        "selective_prediction": {
            "aurc": risk_cov_data["aurc"],
            "clean_accuracy": risk_cov_data["clean_accuracy"],
        },
    }

    out_json = os.path.join(results_dir, "clean_test_metrics.json")
    with open(out_json, "w") as f:
        json.dump(results_payload, f, indent=2)

    # Save fitted scaler and thresholds for inference engine
    calib_checkpoint = os.path.join(results_dir, "calibration_and_policy.pt")
    torch.save(
        {
            "temperature": optimal_t,
            "val_features": val_features,
            "val_labels": val_labels,
            "val_confidences": val_probs_cal.max(dim=-1)[0].numpy(),
            "val_entropies": (-val_probs_cal * torch.log(val_probs_cal + 1e-8)).sum(dim=-1).numpy(),
            "val_correct": (val_probs_cal.argmax(dim=-1) == val_labels).numpy(),
        },
        calib_checkpoint,
    )
    print(f"[✓] Saved calibration & policy state to:     {calib_checkpoint}")
    print(f"[✓] Saved test metrics JSON to:              {out_json}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trustworthy Medical Classifier")
    parser.add_argument("--checkpoint", type=str, default="experiments/checkpoints/best_model.pt")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--results-dir", type=str, default="experiments/results")
    parser.add_argument("--reports-dir", type=str, default="reports/figures")
    parser.add_argument("--mc-samples", type=int, default=15)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        reports_dir=args.reports_dir,
        num_mc_samples=args.mc_samples,
        device_name=args.device,
    )
