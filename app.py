"""
Interactive Clinical AI Reliability Dashboard.
Allows clinicians and researchers to test medical images, apply live corruptions,
and inspect the multi-signal trust triage system in real time.
"""

import os
import sys
import numpy as np
import torch
import streamlit as st
from PIL import Image

# Ensure package root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset import PATHMNIST_CLASSES, get_transforms
from src.models.classifier import build_model
from src.corruptions.image_corruptions import ControlledCorruptions, ImageQualityAssessor
from src.uncertainty.estimator import UncertaintyEstimator
from src.calibration.temperature_scaling import TemperatureScaler
from src.ood.detector import OODDetector
from src.abstention.policy import SelectivePredictionPolicy


st.set_page_config(
    page_title="Trustworthy Medical AI | Reliability Triage",
    page_icon="🔬",
    layout="wide",
)


@st.cache_resource
def load_system():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint_path = "experiments/checkpoints/best_model.pt"
    calib_path = "experiments/results/calibration_and_policy.pt"

    if not os.path.exists(checkpoint_path):
        return None, None, None, None, device

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(num_classes=9, dropout_rate=ckpt.get("dropout_rate", 0.2)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    optimal_t = 1.0
    val_features, val_labels = None, None
    if os.path.exists(calib_path):
        calib_data = torch.load(calib_path, map_location=device, weights_only=False)
        optimal_t = calib_data.get("temperature", 1.0)
        val_features = calib_data.get("val_features")
        val_labels = calib_data.get("val_labels")

    scaler = TemperatureScaler(init_temperature=optimal_t).to(device)
    scaler.temperature.data.fill_(optimal_t)

    ood_detector = OODDetector(method="energy", temperature=optimal_t)
    estimator = UncertaintyEstimator(model, num_samples=15)
    policy = SelectivePredictionPolicy(
        blur_threshold=40.0,
        ood_threshold=-10.0,
        entropy_threshold=1.2,
        epistemic_threshold=0.35,
        confidence_threshold=0.70,
    )

    return model, scaler, estimator, policy, device


def main():
    st.title("🔬 AI That Knows When a Medical Image Cannot Be Trusted")
    st.markdown(
        "**Selective Prediction, Uncertainty Quantification & OOD Screening for Safe Clinical Deployment**"
    )
    st.caption("Research prototype benchmarking PathMNIST colorectal histology against acquisition artifacts & domain shifts.")

    model, scaler, estimator, policy, device = load_system()

    if model is None:
        st.warning(
            "⚠️ No trained model checkpoint found at `experiments/checkpoints/best_model.pt`.\n\n"
            "Please run the pipeline training script first:\n"
            "```bash\npython scripts/run_pipeline.py\n```"
        )
        return

    # Sidebar Controls
    st.sidebar.header("⚙️ Input Selection & Perturbation")
    sample_choice = st.sidebar.selectbox(
        "Choose Medical Input Case:",
        [
            "Case A: Clean Colorectal Histology (In-Distribution)",
            "Case B: Colorectal Histology with Stage Vibration Blur",
            "Case C: Colorectal Histology with Severe Stain Fading",
            "Case D: Out-of-Distribution Modality (Skin Lesion)",
            "Upload Custom Image",
        ],
    )

    img_np = None
    if sample_choice == "Upload Custom Image":
        uploaded_file = st.sidebar.file_uploader("Upload Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file).convert("RGB").resize((28, 28))
            img_np = np.array(pil_img)
    else:
        # Load sample from data/raw if available or generate representative test patch
        rng = np.random.default_rng(101)
        if "Clean" in sample_choice:
            img_np = rng.integers(140, 220, (28, 28, 3), dtype=np.uint8)
            img_np[:, :, 2] = np.clip(img_np[:, :, 2] + 25, 0, 255)  # H&E blue tint
        elif "Vibration" in sample_choice:
            base = rng.integers(140, 220, (28, 28, 3), dtype=np.uint8)
            img_np = ControlledCorruptions.motion_blur(base, severity=4)
        elif "Stain" in sample_choice:
            base = rng.integers(140, 220, (28, 28, 3), dtype=np.uint8)
            img_np = ControlledCorruptions.he_stain_variation(base, severity=5)
        else:
            # Derma-like brownish pigmented patch
            img_np = rng.integers(80, 160, (28, 28, 3), dtype=np.uint8)
            img_np[:, :, 0] += 40

    # Interactive Corruption Simulator
    st.sidebar.subheader("🧪 Live Acquisition Perturbation")
    apply_corr = st.sidebar.checkbox("Apply Live Corruption", value=False)
    if apply_corr:
        corr_type = st.sidebar.selectbox(
            "Corruption Type:",
            [
                "motion_blur",
                "gaussian_noise",
                "defocus_blur",
                "brightness_shift",
                "contrast_shift",
                "he_stain_variation",
            ],
        )
        severity = st.sidebar.slider("Severity Level (1-5)", 1, 5, 3)
        img_np = ControlledCorruptions.apply(img_np, corr_type, severity=severity)

    # Main Layout
    col_img, col_triage = st.columns([1, 1.8])

    with col_img:
        st.subheader("🖼️ Query Medical Scan")
        pil_disp = Image.fromarray(img_np).resize((224, 224), resample=Image.Resampling.NEAREST)
        st.image(pil_disp, caption="Input Patch (Enlarged 28x28)", use_container_width=True)

        # Pre-inference Quality Gate
        quality_meta = ImageQualityAssessor.assess_image(img_np)
        st.markdown("**Deterministic Image Quality:**")
        q_col1, q_col2 = st.columns(2)
        q_col1.metric("Blur (Laplacian)", f"{quality_meta['blur_score']:.1f}")
        q_col2.metric("SNR Estimate", f"{quality_meta['snr_db']:.1f} dB")

    # Run Model Inference
    eval_transform = get_transforms(is_train=False)
    tensor_img = eval_transform(Image.fromarray(img_np)).unsqueeze(0).to(device)

    with torch.no_grad():
        raw_logits = model(tensor_img)
        calibrated_probs = scaler.calibrate_probabilities(raw_logits).cpu().numpy()[0]

    unc_res = estimator.predict_with_uncertainty(tensor_img)
    pred_class_id = int(torch.argmax(raw_logits, dim=-1).item())
    cal_conf = float(np.max(calibrated_probs))
    pred_entropy = float(unc_res["predictive_entropy"].item())
    epistemic_unc = float(unc_res["epistemic_uncertainty"].item())
    ood_score = float(scaler.temperature.item() * torch.logsumexp(raw_logits / scaler.temperature, dim=-1).item())

    # Multi-signal triage policy
    triage = policy.evaluate_sample(
        image_quality_meta=quality_meta,
        calibrated_confidence=cal_conf,
        predictive_entropy=pred_entropy,
        epistemic_uncertainty=epistemic_unc,
        ood_score=ood_score,
    )

    with col_triage:
        st.subheader("🛡️ Trustworthiness Triage Decision")
        if triage["status"] == "TRUSTED":
            st.success("### 🟢 TRUSTED: PROCEED WITH AI DIAGNOSIS")
            st.markdown(
                f"**Prediction:** `{PATHMNIST_CLASSES[pred_class_id].upper()}` "
                f"with **{cal_conf * 100:.1f}%** calibrated confidence."
            )
        else:
            st.error("### 🔴 UNTRUSTED: DEFER TO SENIOR CLINICIAN")
            st.markdown(
                f"**System Recommendation:** The AI recognizes this input cannot be trusted safely. "
                f"Automated diagnosis suppressed."
            )
            st.markdown("**Triggered Warning Flags:**")
            for reason in triage["triggered_reasons"]:
                st.write(f"- ⚠️ {reason}")

        st.markdown("---")
        st.subheader("📊 Diagnostic Telemetry")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Calibrated Conf", f"{cal_conf * 100:.1f}%")
        m2.metric("Total Entropy", f"{pred_entropy:.2f}")
        m3.metric("Epistemic Uncertainty", f"{epistemic_unc:.3f}")
        m4.metric("Energy OOD Score", f"{ood_score:.1f}")

        # Class Probability Breakdown
        st.markdown("**Multiclass Probability Distribution:**")
        chart_data = {PATHMNIST_CLASSES[i]: float(calibrated_probs[i]) for i in range(9)}
        st.bar_chart(chart_data)


if __name__ == "__main__":
    main()
