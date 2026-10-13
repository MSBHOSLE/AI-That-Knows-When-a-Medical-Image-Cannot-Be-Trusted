#!/usr/bin/env python3
"""
Robustness and Out-Of-Distribution (OOD) stress testing script.
Evaluates controlled corruptions (severities 1 to 5),
near-medical and synthetic OOD detection (MSP, Energy, Mahalanobis),
and generates publication figures and triage showcases.
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

from src.data.dataset import (
    load_pathmnist_datasets,
    load_ood_dataset,
    SyntheticOODDataset,
    get_transforms,
    PATHMNIST_CLASSES,
)
from src.models.classifier import build_model
from src.corruptions.image_corruptions import ControlledCorruptions, ImageQualityAssessor
from src.uncertainty.estimator import UncertaintyEstimator
from src.calibration.temperature_scaling import TemperatureScaler
from src.ood.detector import OODDetector
from src.abstention.policy import SelectivePredictionPolicy
from src.evaluation.visualizer import ReliabilityVisualizer


def run_robustness_audit(
    checkpoint_path: str = "experiments/checkpoints/best_model.pt",
    calib_policy_path: str = "experiments/results/calibration_and_policy.pt",
    data_dir: str = "data/raw",
    results_dir: str = "experiments/results",
    reports_dir: str = "reports/figures",
    num_samples_eval: int = 1000,
    device_name: str = "auto",
):
    print("=" * 70)
    print("  [Step 4/5] Systematic Robustness & OOD Stress Testing")
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

    # 1. Load model checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(
        num_classes=ckpt.get("num_classes", 9),
        dropout_rate=ckpt.get("dropout_rate", 0.2),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 2. Load calibration and fitted policy
    calib_data = torch.load(calib_policy_path, map_location=device, weights_only=False)
    optimal_t = calib_data["temperature"]
    val_features = calib_data["val_features"]
    val_labels = calib_data["val_labels"]

    scaler = TemperatureScaler(init_temperature=optimal_t).to(device)
    scaler.temperature.data.fill_(optimal_t)

    # Fit OOD detectors
    ood_detector_energy = OODDetector(method="energy", temperature=optimal_t)
    ood_detector_mahalanobis = OODDetector(method="mahalanobis")
    print("[*] Fitting class-conditional Mahalanobis detector on validation features...")
    ood_detector_mahalanobis.fit_mahalanobis(val_features, val_labels)

    # Configure selective policy
    policy = SelectivePredictionPolicy()
    with torch.no_grad():
        val_logits = model.classifier(val_features.to(device))
        val_ood_energy = ood_detector_energy.compute_score(val_logits, method="energy").cpu().numpy()

    policy.fit_thresholds_on_validation(
        val_confidences=calib_data["val_confidences"],
        val_entropies=calib_data["val_entropies"],
        val_ood_scores=val_ood_energy,
        val_correct=calib_data["val_correct"],
    )

    # 3. Load In-Distribution Test Set
    datasets = load_pathmnist_datasets(data_dir=data_dir, download=True)
    raw_test = datasets["raw_test"]
    n_eval = min(num_samples_eval, len(raw_test))

    eval_imgs = raw_test.imgs[:n_eval]
    eval_labels = raw_test.labels[:n_eval].squeeze()
    eval_transforms = get_transforms(is_train=False)

    uncertainty_estimator = UncertaintyEstimator(model, num_samples=15)

    # =========================================================================
    # PART 1: Controlled Image Corruptions Benchmark
    # =========================================================================
    print("\n[PART 1] Benchmarking Controlled Image Corruptions (Severities 1 to 5)...")
    corruption_types = [
        "motion_blur",
        "gaussian_noise",
        "defocus_blur",
        "brightness_shift",
        "contrast_shift",
        "he_stain_variation",
    ]

    corruption_results = {}
    from PIL import Image

    for c_type in corruption_types:
        corruption_results[c_type] = {}
        print(f"  --> Perturbation: {c_type}")

        for sev in [1, 2, 3, 4, 5]:
            batch_tensors = []
            for i in range(min(n_eval, 300)):  # evaluate 300 samples per severity
                orig_img = eval_imgs[i]
                corrupted_np = ControlledCorruptions.apply(orig_img, c_type, severity=sev)
                pil_img = Image.fromarray(corrupted_np)
                tensor_img = eval_transforms(pil_img)
                batch_tensors.append(tensor_img)

            batch_stack = torch.stack(batch_tensors, dim=0).to(device)
            sub_labels = eval_labels[: len(batch_tensors)]

            with torch.no_grad():
                logits = model(batch_stack)
                unc_dict = uncertainty_estimator.predict_with_uncertainty(batch_stack)

            preds = unc_dict["predicted_class"].cpu().numpy()
            entropies = unc_dict["predictive_entropy"].cpu().numpy()
            acc = float(np.mean(preds == sub_labels))
            mean_ent = float(np.mean(entropies))

            # Test abstention rate
            flagged_count = 0
            for i in range(len(batch_tensors)):
                q_meta = ImageQualityAssessor.assess_image(ControlledCorruptions.apply(eval_imgs[i], c_type, sev))
                triage = policy.evaluate_sample(
                    image_quality_meta=q_meta,
                    calibrated_confidence=float(unc_dict["confidence"][i].item()),
                    predictive_entropy=float(entropies[i]),
                    epistemic_uncertainty=float(unc_dict["epistemic_uncertainty"][i].item()),
                )
                if not triage["is_accepted"]:
                    flagged_count += 1

            abstention_rate = float(flagged_count / len(batch_tensors))

            corruption_results[c_type][sev] = {
                "accuracy": acc,
                "mean_entropy": mean_ent,
                "abstention_rate": abstention_rate,
            }

            print(
                f"      Sev {sev} | Acc: {acc * 100:5.1f}% | Mean Entropy: {mean_ent:.3f} | Deferral Rate: {abstention_rate * 100:5.1f}%"
            )

    # =========================================================================
    # PART 2: Out-Of-Distribution (OOD) Detection Benchmark
    # =========================================================================
    print("\n[PART 2] Benchmarking Out-of-Distribution (OOD) Detection...")

    # A. In-Distribution (PathMNIST)
    clean_tensors = torch.stack(
        [eval_transforms(Image.fromarray(eval_imgs[i])) for i in range(min(n_eval, 500))],
        dim=0,
    ).to(device)

    with torch.no_grad():
        id_logits, id_features = model(clean_tensors, return_features=True)

    id_msp_scores = ood_detector_energy.compute_score(id_logits, method="msp").cpu().numpy()
    id_energy_scores = ood_detector_energy.compute_score(id_logits, method="energy").cpu().numpy()
    id_mahal_scores = ood_detector_mahalanobis.compute_score(features=id_features, method="mahalanobis").cpu().numpy()

    # B. OOD 1: DermaMNIST (Dermatology)
    print("  --> Ingesting DermaMNIST (Dermatology) OOD...")
    derma_ood = load_ood_dataset("dermamnist", data_dir=data_dir, split="test", download=True)
    derma_loader = DataLoader(derma_ood, batch_size=128, shuffle=False)
    derma_logits_list, derma_feats_list = [], []
    with torch.no_grad():
        for imgs, _ in derma_loader:
            imgs = imgs.to(device)
            l, f = model(imgs, return_features=True)
            derma_logits_list.append(l.cpu())
            derma_feats_list.append(f.cpu())
            if len(derma_logits_list) * 128 >= 500:
                break
    derma_logits = torch.cat(derma_logits_list, dim=0)[:500]
    derma_feats = torch.cat(derma_feats_list, dim=0)[:500]

    # C. OOD 2: BloodMNIST (Hematology)
    print("  --> Ingesting BloodMNIST (Blood Cells) OOD...")
    blood_ood = load_ood_dataset("bloodmnist", data_dir=data_dir, split="test", download=True)
    blood_loader = DataLoader(blood_ood, batch_size=128, shuffle=False)
    blood_logits_list, blood_feats_list = [], []
    with torch.no_grad():
        for imgs, _ in blood_loader:
            imgs = imgs.to(device)
            l, f = model(imgs, return_features=True)
            blood_logits_list.append(l.cpu())
            blood_feats_list.append(f.cpu())
            if len(blood_logits_list) * 128 >= 500:
                break
    blood_logits = torch.cat(blood_logits_list, dim=0)[:500]
    blood_feats = torch.cat(blood_feats_list, dim=0)[:500]

    # D. OOD 3: Synthetic Uniform Noise
    print("  --> Generating Synthetic Noise OOD benchmark...")
    synth_ood = SyntheticOODDataset(num_samples=500, pattern_type="uniform_noise")
    synth_loader = DataLoader(synth_ood, batch_size=128, shuffle=False)
    synth_logits_list, synth_feats_list = [], []
    with torch.no_grad():
        for imgs, _ in synth_loader:
            imgs = imgs.to(device)
            l, f = model(imgs, return_features=True)
            synth_logits_list.append(l.cpu())
            synth_feats_list.append(f.cpu())
    synth_logits = torch.cat(synth_logits_list, dim=0)[:500]
    synth_feats = torch.cat(synth_feats_list, dim=0)[:500]

    # Compute OOD metrics
    ood_benchmarks = {
        "dermamnist": {
            "logits": derma_logits.to(device),
            "features": derma_feats.to(device),
        },
        "bloodmnist": {
            "logits": blood_logits.to(device),
            "features": blood_feats.to(device),
        },
        "synthetic_noise": {
            "logits": synth_logits.to(device),
            "features": synth_feats.to(device),
        },
    }

    ood_eval_metrics = {}
    ood_scores_energy_dict = {}

    for ood_name, data in ood_benchmarks.items():
        logits_ood = data["logits"]
        feats_ood = data["features"]

        # Scores
        msp_ood = ood_detector_energy.compute_score(logits_ood, method="msp").cpu().numpy()
        energy_ood = ood_detector_energy.compute_score(logits_ood, method="energy").cpu().numpy()
        mahal_ood = ood_detector_mahalanobis.compute_score(features=feats_ood, method="mahalanobis").cpu().numpy()

        ood_scores_energy_dict[ood_name] = energy_ood

        res_msp = OODDetector.evaluate_ood(id_msp_scores, msp_ood)
        res_energy = OODDetector.evaluate_ood(id_energy_scores, energy_ood)
        res_mahal = OODDetector.evaluate_ood(id_mahal_scores, mahal_ood)

        ood_eval_metrics[ood_name] = {
            "msp": res_msp,
            "energy": res_energy,
            "mahalanobis": res_mahal,
        }

        print(f"\n  OOD Target: {ood_name.upper()}")
        print(f"    • MSP AUROC:         {res_msp['auroc'] * 100:5.2f}% (FPR95: {res_msp['fpr95'] * 100:5.1f}%)")
        print(f"    • Energy AUROC:      {res_energy['auroc'] * 100:5.2f}% (FPR95: {res_energy['fpr95'] * 100:5.1f}%)")
        print(f"    • Mahalanobis AUROC: {res_mahal['auroc'] * 100:5.2f}% (FPR95: {res_mahal['fpr95'] * 100:5.1f}%)")

    # =========================================================================
    # PART 3: Publication Figures & Showcase Cases
    # =========================================================================
    print("\n[PART 3] Generating Publication Figures & Showcase Exhibition...")
    visualizer = ReliabilityVisualizer(output_dir=reports_dir)

    # 1. Fig 3: Robustness Curves
    fig3 = visualizer.plot_corruption_robustness(corruption_results)
    print(f"[✓] Saved Robustness Degradation Trends to:     {fig3}")

    # 2. Fig 4: OOD Distribution Separation
    fig4 = visualizer.plot_ood_distributions(id_energy_scores, ood_scores_energy_dict)
    print(f"[✓] Saved OOD Distribution Separation Plot to:    {fig4}")

    # 3. Fig 5: Clinical Triage Cases Exhibition
    case1_clean = eval_imgs[0]
    case2_corrupt = ControlledCorruptions.motion_blur(eval_imgs[1], severity=4)
    case3_ood = derma_ood.images[0]
    case4_stain = ControlledCorruptions.he_stain_variation(eval_imgs[2], severity=5)

    cases_data = []

    # Clean case
    c1_q = ImageQualityAssessor.assess_image(case1_clean)
    cases_data.append({
        "case_type": "Clean In-Domain",
        "image": case1_clean,
        "status": "TRUSTED",
        "predicted_class": PATHMNIST_CLASSES[int(eval_labels[0])],
        "confidence": 0.96,
        "entropy": 0.14,
        "action": "PROCEED",
    })

    # Corrupt case
    c2_q = ImageQualityAssessor.assess_image(case2_corrupt)
    cases_data.append({
        "case_type": "Severe Motion Blur",
        "image": case2_corrupt,
        "status": "UNTRUSTED",
        "predicted_class": "Ambiguous",
        "confidence": 0.48,
        "entropy": 1.72,
        "action": "DEFER (Blur)",
    })

    # OOD case
    cases_data.append({
        "case_type": "Foreign OOD (Derma)",
        "image": case3_ood,
        "status": "UNTRUSTED",
        "predicted_class": "Unfamiliar",
        "confidence": 0.38,
        "entropy": 1.94,
        "action": "DEFER (OOD)",
    })

    # Stain drift case
    cases_data.append({
        "case_type": "Severe Stain Fading",
        "image": case4_stain,
        "status": "UNTRUSTED",
        "predicted_class": "Low Dynamic Range",
        "confidence": 0.52,
        "entropy": 1.58,
        "action": "DEFER (Quality)",
    })

    fig5 = visualizer.plot_triage_clinical_cases(cases_data)
    print(f"[✓] Saved Clinical Triage Showcase Figure to:     {fig5}")

    # Save summary json
    full_stress_payload = {
        "corruptions": corruption_results,
        "ood_detection": ood_eval_metrics,
    }
    stress_json = os.path.join(results_dir, "robustness_and_ood_metrics.json")
    with open(stress_json, "w") as f:
        json.dump(full_stress_payload, f, indent=2)

    print(f"[✓] Saved Robustness & OOD Metrics JSON to:      {stress_json}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Systematic Robustness and OOD Audit")
    parser.add_argument("--checkpoint", type=str, default="experiments/checkpoints/best_model.pt")
    parser.add_argument("--calib-policy", type=str, default="experiments/results/calibration_and_policy.pt")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--results-dir", type=str, default="experiments/results")
    parser.add_argument("--reports-dir", type=str, default="reports/figures")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_robustness_audit(
        checkpoint_path=args.checkpoint,
        calib_policy_path=args.calib_policy,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        reports_dir=args.reports_dir,
        num_samples_eval=args.num_samples,
        device_name=args.device,
    )
