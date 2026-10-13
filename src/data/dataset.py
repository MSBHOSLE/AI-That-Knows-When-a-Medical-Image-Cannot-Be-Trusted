"""
Data loading and preprocessing module for Medical AI Reliability.
Handles PathMNIST and OOD benchmarks (DermaMNIST, BloodMNIST) with strict split hygiene.
"""

import os
from typing import Dict, Tuple, Optional, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO


# Label names for PathMNIST (9 colorectal cancer histology tissue types)
PATHMNIST_CLASSES = {
    0: "adipose",
    1: "background",
    2: "debris",
    3: "lymphocytes",
    4: "mucus",
    5: "smooth_muscle",
    6: "normal_colon_mucosa",
    7: "cancer_associated_stroma",
    8: "colorectal_adenocarcinoma_epithelium",
}

# Empirical PathMNIST normalization constants
PATHMNIST_MEAN = [0.7405, 0.5330, 0.7058]
PATHMNIST_STD = [0.1237, 0.1768, 0.1244]


class MedicalImageDataset(Dataset):
    """
    Standard PyTorch dataset wrapper around NumPy arrays of medical images.
    Ensures consistent tensor shapes (C, H, W) and normalized float ranges.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            images: np.ndarray of shape (N, H, W, C) or (N, H, W), uint8 in [0, 255].
            labels: np.ndarray of shape (N, 1) or (N,).
            transform: torchvision Compose transforms pipeline.
        """
        self.images = images
        self.labels = labels.squeeze().astype(np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.images[idx]
        label = int(self.labels[idx])

        # If grayscale (H, W), repeat to 3 channels (H, W, 3)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

        # Convert to PIL Image or directly apply transform
        from PIL import Image
        pil_img = Image.fromarray(img)

        if self.transform is not None:
            tensor_img = self.transform(pil_img)
        else:
            tensor_img = transforms.functional.to_tensor(pil_img)

        return tensor_img, label


def get_transforms(
    is_train: bool = True,
    mean: Tuple[float, float, float] = tuple(PATHMNIST_MEAN),
    std: Tuple[float, float, float] = tuple(PATHMNIST_STD),
) -> transforms.Compose:
    """
    Build transforms pipeline for training or evaluation.
    Controlled augmentations are strictly applied ONLY during training.
    """
    if is_train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def load_pathmnist_datasets(
    data_dir: str = "data/raw",
    download: bool = True,
) -> Dict[str, Dataset]:
    """
    Load train, validation, and locked test datasets for PathMNIST.

    Returns:
        Dictionary with keys 'train', 'val', 'test'.
    """
    os.makedirs(data_dir, exist_ok=True)
    DataClass = getattr(medmnist, "PathMNIST")

    train_data = DataClass(split="train", root=data_dir, download=download)
    val_data = DataClass(split="val", root=data_dir, download=download)
    test_data = DataClass(split="test", root=data_dir, download=download)

    train_dataset = MedicalImageDataset(
        train_data.imgs, train_data.labels, transform=get_transforms(is_train=True)
    )
    val_dataset = MedicalImageDataset(
        val_data.imgs, val_data.labels, transform=get_transforms(is_train=False)
    )
    test_dataset = MedicalImageDataset(
        test_data.imgs, test_data.labels, transform=get_transforms(is_train=False)
    )

    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
        "raw_train": train_data,
        "raw_val": val_data,
        "raw_test": test_data,
    }


def load_ood_dataset(
    dataset_name: str = "dermamnist",
    data_dir: str = "data/raw",
    split: str = "test",
    download: bool = True,
) -> Dataset:
    """
    Load an Out-Of-Distribution (OOD) benchmark dataset from MedMNIST.
    Common choices: 'dermamnist' (skin lesions), 'bloodmnist' (blood cells).
    """
    os.makedirs(data_dir, exist_ok=True)
    dataset_name_clean = dataset_name.lower()
    
    # Capitalize for medmnist class lookup, e.g. DermaMNIST, BloodMNIST
    class_map = {
        "dermamnist": "DermaMNIST",
        "bloodmnist": "BloodMNIST",
        "tissuemnist": "TissueMNIST",
        "organamnist": "OrganAMNIST",
    }
    class_name = class_map.get(dataset_name_clean, dataset_name.capitalize())
    DataClass = getattr(medmnist, class_name)

    ood_data = DataClass(split=split, root=data_dir, download=download)
    ood_dataset = MedicalImageDataset(
        ood_data.imgs, ood_data.labels, transform=get_transforms(is_train=False)
    )
    return ood_dataset


class SyntheticOODDataset(Dataset):
    """
    Procedurally generated synthetic OOD dataset.
    Generates controlled uniform noise, Gaussian white noise, or geometric patterns.
    Guaranteed zero external download requirements.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        image_size: int = 28,
        in_channels: int = 3,
        pattern_type: str = "uniform_noise",
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.in_channels = in_channels
        self.pattern_type = pattern_type
        rng = np.random.default_rng(seed)

        if pattern_type == "uniform_noise":
            self.images = (rng.uniform(0, 255, (num_samples, image_size, image_size, in_channels))).astype(np.uint8)
        elif pattern_type == "gaussian_noise":
            noise = rng.normal(128, 50, (num_samples, image_size, image_size, in_channels))
            self.images = np.clip(noise, 0, 255).astype(np.uint8)
        elif pattern_type == "high_frequency_checkerboard":
            checker = np.indices((image_size, image_size)).sum(axis=0) % 2 * 255
            checker = np.stack([checker] * in_channels, axis=-1).astype(np.uint8)
            self.images = np.stack([checker] * num_samples, axis=0)
        else:
            raise ValueError(f"Unknown pattern_type: {pattern_type}")

        self.labels = np.full(num_samples, -1, dtype=np.int64)  # -1 indicates OOD
        self.transform = get_transforms(is_train=False)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        from PIL import Image
        pil_img = Image.fromarray(self.images[idx])
        return self.transform(pil_img), int(self.labels[idx])


def create_dataloaders(
    datasets: Dict[str, Dataset],
    batch_size: int = 128,
    num_workers: int = 2,
) -> Dict[str, DataLoader]:
    """
    Construct DataLoaders for train, validation, and test datasets.
    """
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }
    return loaders
