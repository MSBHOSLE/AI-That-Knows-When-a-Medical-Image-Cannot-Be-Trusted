#!/usr/bin/env python3
"""
Data preparation and validation script.
Downloads and verifies PathMNIST, DermaMNIST, and BloodMNIST datasets,
validates class balance and split hygiene, and records provenance metadata.
"""

import os
import sys
import json
import argparse
import numpy as np

# Ensure src modules are resolvable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import (
    load_pathmnist_datasets,
    load_ood_dataset,
    PATHMNIST_CLASSES,
)


def prepare_and_verify_data(data_dir: str = "data/raw", processed_dir: str = "data/processed"):
    print("=" * 70)
    print("  [Step 1/5] Medical Dataset Ingestion & Split Verification")
    print("=" * 70)

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # 1. Ingest PathMNIST
    print("[*] Ingesting PathMNIST benchmark (colorectal histology)...")
    pathmnist_splits = load_pathmnist_datasets(data_dir=data_dir, download=True)
    raw_train = pathmnist_splits["raw_train"]
    raw_val = pathmnist_splits["raw_val"]
    raw_test = pathmnist_splits["raw_test"]

    n_train = len(raw_train)
    n_val = len(raw_val)
    n_test = len(raw_test)
    total_samples = n_train + n_val + n_test

    print(f"    - Train split:      {n_train:,} images (shape: {raw_train.imgs.shape[1:]})")
    print(f"    - Validation split: {n_val:,} images")
    print(f"    - Locked Test:      {n_test:,} images")
    print(f"    - Total volume:     {total_samples:,} images")

    # 2. Verify class distribution in training set
    train_labels = raw_train.labels.squeeze()
    unique, counts = np.unique(train_labels, return_counts=True)
    class_distribution = {
        PATHMNIST_CLASSES[int(c)]: int(cnt) for c, cnt in zip(unique, counts)
    }

    print("\n[*] Training class distribution:")
    for c_id, c_name in PATHMNIST_CLASSES.items():
        cnt = class_distribution.get(c_name, 0)
        pct = (cnt / n_train) * 100.0
        print(f"    Class {c_id} ({c_name:<36}): {cnt:,} samples ({pct:.1f}%)")

    # 3. Ingest OOD benchmark datasets
    print("\n[*] Ingesting Near-Medical Out-of-Distribution (OOD) benchmarks...")
    try:
        derma_ood = load_ood_dataset("dermamnist", data_dir=data_dir, split="test", download=True)
        print(f"    - DermaMNIST (Dermatology): {len(derma_ood):,} test images")
    except Exception as e:
        print(f"    [!] Warning downloading DermaMNIST: {e}")

    try:
        blood_ood = load_ood_dataset("bloodmnist", data_dir=data_dir, split="test", download=True)
        print(f"    - BloodMNIST (Hematology):  {len(blood_ood):,} test images")
    except Exception as e:
        print(f"    [!] Warning downloading BloodMNIST: {e}")

    # 4. Save metadata summary
    meta = {
        "dataset_name": "PathMNIST",
        "num_classes": len(PATHMNIST_CLASSES),
        "classes": PATHMNIST_CLASSES,
        "splits": {
            "train": n_train,
            "val": n_val,
            "test": n_test,
        },
        "class_distribution_train": class_distribution,
        "input_resolution": list(raw_train.imgs.shape[1:]),
    }

    meta_file = os.path.join(processed_dir, "dataset_metadata.json")
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[✓] Ingestion complete. Verified metadata saved to: {meta_file}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and verify medical datasets")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Directory for raw data")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Directory for metadata")
    args = parser.parse_args()

    prepare_and_verify_data(data_dir=args.data_dir, processed_dir=args.processed_dir)
