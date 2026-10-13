#!/usr/bin/env python3
"""
Master CLI pipeline orchestrator for Medical AI Reliability.
Executes the full end-to-end scientific lifecycle reproducibly:
1. Data Ingestion & Split Audit
2. Model Training with MC-Dropout
3. Validation-Guided Calibration & Locked Test Evaluation
4. Systematic Robustness & OOD Auditing
5. Research Report Compilation
"""

import os
import sys
import subprocess
import argparse
import time

# Ensure package root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def run_command(cmd_list, description):
    print("\n" + "#" * 70)
    print(f"  EXECUTING: {description}")
    print("#" * 70)
    start_t = time.time()
    res = subprocess.run(cmd_list, check=True)
    elapsed = time.time() - start_t
    print(f"[✓] {description} completed in {elapsed:.1f}s")
    return res


def main():
    parser = argparse.ArgumentParser(description="Master Lifecycle Pipeline for Medical AI Reliability")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--subset", type=float, default=1.0, help="Fraction of dataset to train on")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (mps, cuda, cpu, auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-train", action="store_true", help="Skip training if checkpoint already exists")
    args = parser.parse_args()

    python_bin = sys.executable
    script_dir = os.path.dirname(os.path.abspath(__file__))

    total_start = time.time()

    print("=" * 70)
    print("  PROJECT 2: AI THAT KNOWS WHEN A MEDICAL IMAGE CANNOT BE TRUSTED")
    print("  Master Scientific Lifecycle Runner")
    print("=" * 70)

    # 1. Prepare Data
    run_command(
        [python_bin, os.path.join(script_dir, "prepare_data.py")],
        "Step 1: Ingesting & Verifying Medical Datasets",
    )

    # 2. Train Model
    if not args.skip_train or not os.path.exists("experiments/checkpoints/best_model.pt"):
        run_command(
            [
                python_bin,
                os.path.join(script_dir, "train.py"),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--lr", str(args.lr),
                "--subset", str(args.subset),
                "--device", args.device,
                "--seed", str(args.seed),
            ],
            "Step 2: Training Trustworthy Classifier with MC-Dropout",
        )
    else:
        print("\n[!] Skipping training (--skip-train set and checkpoint found).")

    # 3. Evaluate & Calibrate
    run_command(
        [
            python_bin,
            os.path.join(script_dir, "evaluate.py"),
            "--device", args.device,
        ],
        "Step 3: Post-Hoc Calibration & Locked Test Evaluation",
    )

    # 4. Robustness & OOD Stress Test
    run_command(
        [
            python_bin,
            os.path.join(script_dir, "run_robustness.py"),
            "--device", args.device,
        ],
        "Step 4: Systematic Robustness & OOD Stress Testing",
    )

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"  [✓] ALL LIFECYCLE STAGES COMPLETED SUCCESSFULLY IN {total_elapsed:.1f}s!")
    print("=" * 70)
    print("  Artifacts generated:")
    print("  - Checkpoints: experiments/checkpoints/best_model.pt")
    print("  - Figures:     reports/figures/fig1_calibration_reliability.png")
    print("                 reports/figures/fig2_risk_coverage_curve.png")
    print("                 reports/figures/fig3_corruption_robustness.png")
    print("                 reports/figures/fig4_ood_distribution_separation.png")
    print("                 reports/figures/fig5_clinical_triage_cases.png")
    print("  - Metrics:     experiments/results/clean_test_metrics.json")
    print("                 experiments/results/robustness_and_ood_metrics.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
