#!/usr/bin/env python3
"""
Model training script for Trustworthy Medical AI.
Trains TrustworthyMedicalCNN with MC-Dropout on PathMNIST,
monitoring validation loss/accuracy and preserving checkpoints.
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

# Ensure src modules are resolvable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import load_pathmnist_datasets, create_dataloaders
from src.models.classifier import build_model


def select_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def train_model(
    data_dir: str = "data/raw",
    checkpoint_dir: str = "experiments/checkpoints",
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    dropout_rate: float = 0.2,
    subset_fraction: float = 1.0,
    device_name: str = "auto",
    seed: int = 42,
):
    print("=" * 70)
    print("  [Step 2/5] Training Trustworthy Medical Classifier")
    print("=" * 70)

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = select_device(device_name)
    print(f"[*] Execution device: {device}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Load datasets
    print("[*] Loading PathMNIST splits...")
    datasets = load_pathmnist_datasets(data_dir=data_dir, download=True)
    train_dataset = datasets["train"]
    val_dataset = datasets["val"]

    if subset_fraction < 1.0:
        n_sub_train = int(len(train_dataset) * subset_fraction)
        n_sub_val = int(len(val_dataset) * subset_fraction)
        train_dataset = Subset(train_dataset, range(n_sub_train))
        val_dataset = Subset(val_dataset, range(n_sub_val))
        print(f"    - Using subset fraction: {subset_fraction:.2f} ({n_sub_train:,} train, {n_sub_val:,} val)")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # 2. Build model
    model = build_model(
        num_classes=9,
        in_channels=3,
        dropout_rate=dropout_rate,
        base_channels=64,
    ).to(device)

    print(f"[*] Model architecture: TrustworthyMedicalCNN (Base Channels=64, Dropout={dropout_rate})")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    - Trainable parameters: {total_params:,}")

    # 3. Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    history = []

    # 4. Training Loop
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct_train += preds.eq(labels).sum().item()
            total_train += labels.size(0)

        scheduler.step()

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100.0

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)

                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(logits, dim=-1)
                correct_val += preds.eq(labels).sum().item()
                total_val += labels.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100.0

        log_entry = {
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "val_loss": epoch_val_loss,
            "val_acc": epoch_val_acc,
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(log_entry)

        is_best = epoch_val_acc > best_val_acc
        if is_best:
            best_val_acc = epoch_val_acc
            best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": best_val_acc,
                    "val_loss": epoch_val_loss,
                    "num_classes": 9,
                    "dropout_rate": dropout_rate,
                },
                best_model_path,
            )

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:5.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:5.2f}%"
            f"{' [★ BEST]' if is_best else ''}"
        )

    elapsed = time.time() - start_time
    print(f"\n[✓] Training complete in {elapsed:.1f}s. Peak Validation Accuracy: {best_val_acc:.2f}%")

    # Save training history
    log_path = os.path.join(checkpoint_dir, "training_history.json")
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[✓] Checkpoint saved to: {os.path.join(checkpoint_dir, 'best_model.pt')}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Trustworthy Medical CNN")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--subset", type=float, default=1.0, help="Fraction of dataset to use")
    parser.add_argument("--device", type=str, default="auto", help="Device (mps, cuda, cpu, auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout_rate=args.dropout,
        subset_fraction=args.subset,
        device_name=args.device,
        seed=args.seed,
    )
