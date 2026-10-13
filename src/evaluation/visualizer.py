"""
Visualization module for publication-quality figures and clinical triage inspections.
Generates:
1. Reliability diagrams (before and after temperature scaling)
2. Risk-coverage curves for selective prediction
3. Corruption robustness curves across 5 severity steps
4. In-distribution vs Out-of-Distribution separation plots
5. Error-uncertainty alignment & silent failure distributions
6. Multi-panel clinical triage case exhibition
"""

import os
from typing import Dict, List, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class ReliabilityVisualizer:
    """
    Renders research figures formatted with clean typography and palettes.
    """

    def __init__(self, output_dir: str = "reports/figures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Apply standard clean styling
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
        plt.rcParams["axes.edgecolor"] = "#cccccc"
        plt.rcParams["axes.linewidth"] = 0.8

    def plot_calibration_comparison(
        self,
        uncal_stats: Dict[str, Any],
        cal_stats: Dict[str, Any],
        save_name: str = "fig1_calibration_reliability.png",
    ) -> str:
        """
        Render 2-panel side-by-side reliability diagram (Uncalibrated vs Calibrated).
        """
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)

        for ax, stats, title, color in zip(
            axes,
            [uncal_stats, cal_stats],
            [
                f"Uncalibrated (ECE = {uncal_stats['ece'] * 100:.2f}%)",
                f"Temperature Scaled (ECE = {cal_stats['ece'] * 100:.2f}%)",
            ],
            ["#d9534f", "#2b7bb9"],
        ):
            # Perfect calibration line
            ax.plot([0, 1], [0, 1], "--", color="#666666", label="Perfect Calibration")

            confs = stats["bin_confidences"]
            accs = stats["bin_accuracies"]
            counts = stats["bin_counts"]

            # Filter non-empty bins
            valid = [i for i, c in enumerate(counts) if c > 0]
            if len(valid) > 0:
                ax.plot(
                    [confs[i] for i in valid],
                    [accs[i] for i in valid],
                    marker="o",
                    linewidth=2.0,
                    color=color,
                    label="Model Confidence",
                )
                ax.bar(
                    [confs[i] for i in valid],
                    [accs[i] for i in valid],
                    width=0.04,
                    alpha=0.25,
                    color=color,
                    edgecolor=color,
                )

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Predicted Confidence", fontsize=11, fontweight="medium")
            ax.set_ylabel("Empirical Accuracy", fontsize=11, fontweight="medium")
            ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
            ax.legend(frameon=True, loc="upper left")

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_risk_coverage(
        self,
        risk_cov_data: Dict[str, Any],
        save_name: str = "fig2_risk_coverage_curve.png",
    ) -> str:
        """
        Render selective prediction Risk vs Coverage curve.
        """
        covs = np.array(risk_cov_data["coverages"])
        risks = np.array(risk_cov_data["selective_risks"])
        aurc = risk_cov_data.get("aurc", 0.0)
        clean_acc = risk_cov_data.get("clean_accuracy", 0.0)

        fig, ax = plt.subplots(figsize=(6.8, 5.0), dpi=300)

        # Baseline error line at 100% coverage
        baseline_error = 1.0 - clean_acc
        ax.axhline(
            baseline_error,
            linestyle="--",
            color="#888888",
            label=f"Full Automation Error ({baseline_error * 100:.1f}%)",
        )

        # Selective risk curve
        ax.plot(
            covs * 100.0,
            risks * 100.0,
            linewidth=2.5,
            color="#1b7837",
            label=f"Selective Predictor (AURC = {aurc * 100:.2f})",
        )
        ax.fill_between(covs * 100.0, 0, risks * 100.0, color="#1b7837", alpha=0.15)

        # Safe clinical zone annotation
        ax.axhline(5.0, linestyle=":", color="#d95f02", label="Target Clinical Risk (5%)")

        ax.set_xlim(0, 100)
        ax.set_ylim(0, max(float(baseline_error * 120), 15.0))
        ax.set_xlabel("Coverage (% Cases Handled by AI)", fontsize=11, fontweight="medium")
        ax.set_ylabel("Selective Risk (% Error on Accepted Cases)", fontsize=11, fontweight="medium")
        ax.set_title("Selective Prediction: Clinical Risk vs Coverage Tradeoff", fontsize=12, fontweight="bold", pad=12)
        ax.legend(frameon=True, loc="upper left")

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_corruption_robustness(
        self,
        corruption_results: Dict[str, Dict[int, Dict[str, float]]],
        save_name: str = "fig3_corruption_robustness.png",
    ) -> str:
        """
        Render 2-panel plot: Accuracy degradation and Uncertainty rise across severity levels.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.0), dpi=300)

        palette = sns.color_palette("tab10", n_colors=len(corruption_results))

        for (corr_name, severities), color in zip(corruption_results.items(), palette):
            sevs = sorted(severities.keys())
            accs = [severities[s]["accuracy"] * 100.0 for s in sevs]
            entropies = [severities[s]["mean_entropy"] for s in sevs]

            clean_label = corr_name.replace("_", " ").title()
            ax1.plot(sevs, accs, marker="o", linewidth=2.0, color=color, label=clean_label)
            ax2.plot(sevs, entropies, marker="s", linewidth=2.0, color=color, label=clean_label)

        ax1.set_xlabel("Corruption Severity Level", fontsize=11, fontweight="medium")
        ax1.set_ylabel("Diagnostic Accuracy (%)", fontsize=11, fontweight="medium")
        ax1.set_title("Performance Degradation under Stress", fontsize=12, fontweight="bold")
        ax1.set_xticks([1, 2, 3, 4, 5])
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2.set_xlabel("Corruption Severity Level", fontsize=11, fontweight="medium")
        ax2.set_ylabel("Predictive Entropy H(p)", fontsize=11, fontweight="medium")
        ax2.set_title("Uncertainty Growth with Degradation", fontsize=12, fontweight="bold")
        ax2.set_xticks([1, 2, 3, 4, 5])
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(frameon=True, loc="upper left", bbox_to_anchor=(1.02, 1.0))

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_ood_distributions(
        self,
        id_scores: np.ndarray,
        ood_scores_dict: Dict[str, np.ndarray],
        metric_name: str = "Energy OOD Score",
        save_name: str = "fig4_ood_distribution_separation.png",
    ) -> str:
        """
        Render score distributions for In-Distribution vs Out-of-Distribution sets.
        """
        fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=300)

        # Clean ID histogram
        sns.kdeplot(
            id_scores,
            ax=ax,
            fill=True,
            color="#2ca02c",
            label="In-Distribution (PathMNIST)",
            linewidth=2.0,
            alpha=0.35,
        )

        ood_colors = ["#d62728", "#9467bd", "#ff7f0e"]
        for (ood_name, scores), color in zip(ood_scores_dict.items(), ood_colors):
            clean_name = ood_name.replace("_", " ").title()
            sns.kdeplot(
                scores,
                ax=ax,
                fill=True,
                color=color,
                label=f"OOD ({clean_name})",
                linewidth=2.0,
                alpha=0.35,
            )

        ax.set_xlabel(metric_name, fontsize=11, fontweight="medium")
        ax.set_ylabel("Density", fontsize=11, fontweight="medium")
        ax.set_title("Domain Separation: In-Distribution vs Foreign Modalities", fontsize=12, fontweight="bold", pad=12)
        ax.legend(frameon=True, loc="upper right")

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_triage_clinical_cases(
        self,
        cases_data: List[Dict[str, Any]],
        save_name: str = "fig5_clinical_triage_cases.png",
    ) -> str:
        """
        Render multi-panel clinical triage showcase exhibiting:
        - Panel 1: Trusted In-Distribution prediction
        - Panel 2: Untrusted Corrupted input (flagged for review)
        - Panel 3: Untrusted OOD input (deferred)
        - Panel 4: Untrusted Ambiguous boundary sample (high epistemic uncertainty)
        """
        num_cases = len(cases_data)
        fig, axes = plt.subplots(1, num_cases, figsize=(3.5 * num_cases, 4.8), dpi=300)

        if num_cases == 1:
            axes = [axes]

        for ax, case in zip(axes, cases_data):
            img = case["image"]
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                # If PyTorch format (C, H, W), transpose to (H, W, C)
                img = np.transpose(img, (1, 2, 0))

            # Un-normalize if float in [-3, 3]
            if img.dtype != np.uint8:
                img_disp = (img - img.min()) / (img.max() - img.min() + 1e-8)
            else:
                img_disp = img

            ax.imshow(img_disp)
            ax.axis("off")

            status = case.get("status", "TRUSTED")
            status_color = "#1b7837" if status == "TRUSTED" else "#b2182b"

            title = f"{case.get('case_type', 'Sample')}\n[{status}]"
            ax.set_title(title, fontsize=11, fontweight="bold", color=status_color, pad=8)

            # Metadata caption underneath
            text_str = (
                f"Pred: {case.get('predicted_class', 'N/A')}\n"
                f"Conf: {case.get('confidence', 0.0) * 100:.1f}%\n"
                f"Entropy: {case.get('entropy', 0.0):.2f}\n"
                f"Action: {case.get('action', 'N/A')}"
            )
            ax.text(
                0.5,
                -0.08,
                text_str,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#cccccc"),
            )

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path
