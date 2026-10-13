"""
Image corruptions and pre-inference Image Quality Assessment (IQA) module.
Implements controlled clinical perturbations (motion blur, noise, stain drift, illumination)
and deterministic quality screening metrics.
"""

from typing import Dict, Tuple, Any
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image, ImageEnhance, ImageFilter
import torch


class ControlledCorruptions:
    """
    Applies controlled image corruptions across severity levels 1 to 5.
    Operates on uint8 NumPy arrays (H, W, C) in [0, 255] or PIL Images.
    """

    @staticmethod
    def gaussian_noise(img: np.ndarray, severity: int = 1) -> np.ndarray:
        """
        Add zero-mean Gaussian detector/sensor noise.
        Severity controls noise variance sigma.
        """
        sigmas = [0.08, 0.16, 0.28, 0.42, 0.60]
        sigma = sigmas[severity - 1] * 255.0
        rng = np.random.default_rng(seed=42 + severity)
        noise = rng.normal(0, sigma, img.shape)
        corrupted = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return corrupted

    @staticmethod
    def motion_blur(img: np.ndarray, severity: int = 1) -> np.ndarray:
        """
        Simulate scanner stage vibration or patient motion blur along an angle.
        """
        kernel_sizes = [3, 5, 7, 9, 11]
        ksize = kernel_sizes[severity - 1]
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        kernel[int((ksize - 1) / 2), :] = np.ones(ksize)
        kernel = kernel / ksize

        # Convolve per channel
        corrupted = np.zeros_like(img)
        for c in range(img.shape[2]):
            corrupted[:, :, c] = ndimage.convolve(img[:, :, c].astype(np.float32), kernel, mode="nearest")
        return np.clip(corrupted, 0, 255).astype(np.uint8)

    @staticmethod
    def defocus_blur(img: np.ndarray, severity: int = 1) -> np.ndarray:
        """
        Simulate optical focal plane drift (out-of-focus microscope objective).
        """
        sigmas = [0.8, 1.4, 2.0, 2.8, 3.8]
        sigma = sigmas[severity - 1]
        corrupted = np.zeros_like(img)
        for c in range(img.shape[2]):
            corrupted[:, :, c] = ndimage.gaussian_filter(img[:, :, c].astype(np.float32), sigma=sigma)
        return np.clip(corrupted, 0, 255).astype(np.uint8)

    @staticmethod
    def brightness_shift(img: np.ndarray, severity: int = 1) -> np.ndarray:
        """
        Simulate illumination lamp attenuation or over-exposure.
        """
        factors = [0.75, 0.60, 0.45, 0.30, 0.15]
        factor = factors[severity - 1]
        pil_img = Image.fromarray(img)
        enhancer = ImageEnhance.Brightness(pil_img)
        corrupted_pil = enhancer.enhance(factor)
        return np.array(corrupted_pil, dtype=np.uint8)

    @staticmethod
    def contrast_shift(img: np.ndarray, severity: int = 1) -> np.ndarray:
        """
        Simulate low dynamic range or degraded condenser aperture contrast.
        """
        factors = [0.75, 0.55, 0.40, 0.25, 0.10]
        factor = factors[severity - 1]
        pil_img = Image.fromarray(img)
        enhancer = ImageEnhance.Contrast(pil_img)
        corrupted_pil = enhancer.enhance(factor)
        return np.array(corrupted_pil, dtype=np.uint8)

    @staticmethod
    def he_stain_variation(img: np.ndarray, severity: int = 1) -> np.ndarray:
        """
        Histology-specific domain corruption:
        Simulate Hematoxylin and Eosin (H&E) chemical stain fading and reagent batch shift.
        Attenuates Hematoxylin blue/purple channels or introduces eosinophilia.
        """
        stain_factors = [
            (0.95, 0.90, 0.80),
            (0.90, 0.80, 0.65),
            (0.85, 0.70, 0.50),
            (0.75, 0.60, 0.35),
            (0.65, 0.50, 0.20),
        ]
        r_f, g_f, b_f = stain_factors[severity - 1]
        img_float = img.astype(np.float32)
        img_float[:, :, 0] *= r_f
        img_float[:, :, 1] *= g_f
        img_float[:, :, 2] *= b_f
        return np.clip(img_float, 0, 255).astype(np.uint8)

    @classmethod
    def apply(cls, img: np.ndarray, corruption_type: str, severity: int = 1) -> np.ndarray:
        """
        Dispatcher method to apply a named corruption at a specified severity.
        """
        if not hasattr(cls, corruption_type):
            raise ValueError(f"Unknown corruption type: {corruption_type}")
        method = getattr(cls, corruption_type)
        return method(img, severity=severity)


class ImageQualityAssessor:
    """
    Computes deterministic spatial quality metrics on raw input images
    to detect severe physical degradation prior to neural inference.
    """

    @staticmethod
    def compute_blur_score(img: np.ndarray) -> float:
        """
        Compute sharpness via the variance of the Laplacian filter.
        Higher indicates sharper image; very low (< 50) indicates severe blur.
        """
        if img.ndim == 3:
            # Convert RGB to grayscale
            gray = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
        else:
            gray = img.astype(np.float32)

        laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        lap_img = ndimage.convolve(gray, laplacian_kernel, mode="reflect")
        return float(np.var(lap_img))

    @staticmethod
    def compute_snr_estimate(img: np.ndarray) -> float:
        """
        Estimate approximate Signal-to-Noise Ratio (in dB) using local standard deviation.
        """
        gray = (
            0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
            if img.ndim == 3
            else img.astype(np.float32)
        )
        signal = np.mean(gray)
        noise = np.std(gray) + 1e-6
        snr = 20.0 * np.log10(max(signal / noise, 1e-3))
        return float(snr)

    @staticmethod
    def compute_clipping_ratio(img: np.ndarray) -> float:
        """
        Compute fraction of pixels saturated at extreme intensities (0 or 255).
        """
        extreme_pixels = np.sum((img <= 2) | (img >= 253))
        total_pixels = img.size
        return float(extreme_pixels / total_pixels)

    @classmethod
    def assess_image(cls, img: np.ndarray) -> Dict[str, Any]:
        """
        Run full suite of image quality checks.
        """
        blur_score = cls.compute_blur_score(img)
        snr_db = cls.compute_snr_estimate(img)
        clipping = cls.compute_clipping_ratio(img)
        mean_intensity = float(np.mean(img))

        # Quality flag rules
        is_acceptable = True
        warnings = []

        if blur_score < 40.0:
            is_acceptable = False
            warnings.append(f"Severe blur detected (Laplacian Var = {blur_score:.1f} < 40.0)")

        if clipping > 0.45:
            is_acceptable = False
            warnings.append(f"Excessive clipping/saturation ({clipping * 100:.1f}% extreme pixels)")

        if mean_intensity < 20.0 or mean_intensity > 235.0:
            is_acceptable = False
            warnings.append(f"Abnormal illumination (Mean intensity = {mean_intensity:.1f})")

        return {
            "acceptable": is_acceptable,
            "blur_score": blur_score,
            "snr_db": snr_db,
            "clipping_ratio": clipping,
            "mean_intensity": mean_intensity,
            "warnings": warnings,
        }
