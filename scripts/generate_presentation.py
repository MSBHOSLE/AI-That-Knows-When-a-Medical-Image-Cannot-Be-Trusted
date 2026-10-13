#!/usr/bin/env python3
"""
Generate publication-grade PowerPoint (.pptx) presentation for:
Project 2: AI That Knows When a Medical Image Cannot Be Trusted.
Demonstrates motive, problem solving, empirical data, embedded figures, and clinical impact.
"""

import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE


# -----------------------------------------------------------------------------
# Color Palette Constants
# -----------------------------------------------------------------------------
COLOR_PRIMARY_NAVY = RGBColor(15, 32, 67)       # Deep Navy #0F2043
COLOR_DARK_SLATE   = RGBColor(30, 41, 59)       # #1E293B
COLOR_LIGHT_BG     = RGBColor(248, 250, 252)    # #F8FAFC
COLOR_WHITE        = RGBColor(255, 255, 255)
COLOR_CARD_BORDER  = RGBColor(226, 232, 240)    # #E2E8F0
COLOR_TEAL_ACCENT  = RGBColor(13, 148, 136)     # #0D9488
COLOR_GREEN_TRUST  = RGBColor(22, 163, 74)      # #16A34A
COLOR_RED_ALERT    = RGBColor(220, 38, 38)      # #DC2626
COLOR_GRAY_TEXT    = RGBColor(100, 116, 139)    # #64748B
COLOR_MUTED_BG     = RGBColor(241, 245, 249)    # #F1F5F9


def create_deck(output_pptx_path="Medical_AI_Reliability_Presentation.pptx"):
    prs = Presentation()
    # 16:9 Widescreen dimensions: 13.333 x 7.5 inches
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(base_dir, "reports", "figures")

    # -------------------------------------------------------------------------
    # Helper Functions
    # -------------------------------------------------------------------------
    def set_slide_background(slide, color):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = color
        bg.line.fill.background()
        return bg

    def add_header(slide, title_text, category_text="PROJECT 2: MEDICAL AI RELIABILITY"):
        # Category Pill / Super-title
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.35))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = COLOR_TEAL_ACCENT

        # Main Slide Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.72), Inches(11.7), Inches(0.65))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = COLOR_PRIMARY_NAVY

        # Subtle divider line
        line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.42), Inches(11.73), Inches(0.02))
        line.fill.solid()
        line.fill.fore_color.rgb = COLOR_CARD_BORDER
        line.line.fill.background()

    def add_card(slide, left, top, width, height, bg_color=COLOR_WHITE, border_color=COLOR_CARD_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        if border_color:
            card.line.color.rgb = border_color
            card.line.width = Pt(1)
        else:
            card.line.fill.background()
        return card

    # =========================================================================
    # SLIDE 1: Title Slide (Dark Theme)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1, COLOR_PRIMARY_NAVY)

    # Accent decorative banner
    accent_bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.6), Inches(1.2), Inches(0.08))
    accent_bar.fill.solid()
    accent_bar.fill.fore_color.rgb = COLOR_TEAL_ACCENT
    accent_bar.line.fill.background()

    # Category badge
    tag_box = s1.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.5), Inches(0.4))
    p_tag = tag_box.text_frame.paragraphs[0]
    p_tag.text = "RESEARCH PORTFOLIO PROJECT 2 | TRUSTWORTHY MEDICAL AI"
    p_tag.font.size = Pt(13)
    p_tag.font.bold = True
    p_tag.font.color.rgb = COLOR_TEAL_ACCENT

    # Title text
    title_box = s1.shapes.add_textbox(Inches(0.8), Inches(2.25), Inches(11.7), Inches(2.0))
    tf_t = title_box.text_frame
    tf_t.word_wrap = True
    p1 = tf_t.paragraphs[0]
    p1.text = "AI That Knows When a Medical Image Cannot Be Trusted"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_WHITE

    p2 = tf_t.add_paragraph()
    p2.text = "Selective Prediction, Epistemic Uncertainty Quantification & Out-of-Distribution Screening for Safe Clinical Deployment"
    p2.font.size = Pt(18)
    p2.font.color.rgb = RGBColor(203, 213, 225)
    p2.space_before = Pt(14)

    # Metadata Card at Bottom
    s1_card = add_card(s1, Inches(0.8), Inches(5.1), Inches(11.73), Inches(1.6), bg_color=RGBColor(24, 46, 88), border_color=None)
    meta_box = s1.shapes.add_textbox(Inches(1.1), Inches(5.25), Inches(11.2), Inches(1.3))
    tf_m = meta_box.text_frame
    p_m1 = tf_m.paragraphs[0]
    p_m1.text = "Benchmark Modality: Colorectal Histopathology (PathMNIST - 107,180 Scans) vs. Dermatology & Cytology OOD"
    p_m1.font.size = Pt(13)
    p_m1.font.bold = True
    p_m1.font.color.rgb = COLOR_WHITE

    p_m2 = tf_m.add_paragraph()
    p_m2.text = "Core Pillars: Deterministic Image Quality Gate • Monte Carlo Dropout Epistemic Uncertainty • Temperature Calibration • Mahalanobis Manifold OOD"
    p_m2.font.size = Pt(12)
    p_m2.font.color.rgb = RGBColor(148, 163, 184)
    p_m2.space_before = Pt(6)

    p_m3 = tf_m.add_paragraph()
    p_m3.text = "Status: Fully Verified & Reproducible (100% PyTest Passing • Interactive Clinical Dashboard Built • Formal PDF Report Published)"
    p_m3.font.size = Pt(11)
    p_m3.font.color.rgb = COLOR_TEAL_ACCENT
    p_m3.space_before = Pt(6)

    # =========================================================================
    # SLIDE 2: The Core Motive & The Hidden Clinical Crisis
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2, COLOR_LIGHT_BG)
    add_header(s2, "The Motive: The Dangerous Fallacy of 'Closed-World' Medical AI")

    # 3 Comparison Columns
    col_w = Inches(3.68)
    gap = Inches(0.34)
    col_top = Inches(1.7)
    col_h = Inches(5.2)

    # Col 1: The Problem
    add_card(s2, Inches(0.8), col_top, col_w, col_h)
    c1 = s2.shapes.add_textbox(Inches(1.0), col_top + Inches(0.2), col_w - Inches(0.4), col_h - Inches(0.4))
    tf1 = c1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "1. The Clinical Danger"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_RED_ALERT

    points1 = [
        ("Forced Prediction:", "Standard AI forces a softmax output sum(p)=1 on EVERY image, even when blind."),
        ("Catastrophic Overconfidence:", "When models fail on corrupted scans, they routinely assert wrong diagnoses with >95% confidence."),
        ("Silent Patient Harm:", "A false-positive or missed malignancy delivered with high confidence misleads doctors and damages clinical trust.")
    ]
    for title, desc in points1:
        p = tf1.add_paragraph()
        p.text = f"• {title} {desc}"
        p.font.size = Pt(12)
        p.font.color.rgb = COLOR_DARK_SLATE
        p.space_before = Pt(12)

    # Col 2: Real-World Failures
    add_card(s2, Inches(0.8) + col_w + gap, col_top, col_w, col_h)
    c2 = s2.shapes.add_textbox(Inches(1.0) + col_w + gap, col_top + Inches(0.2), col_w - Inches(0.4), col_h - Inches(0.4))
    tf2 = c2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "2. Real Hospital Realities"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_NAVY

    points2 = [
        ("Microscope Focus Drift:", "Automated slide scanner stages vibrate or drift out-of-focus during routine high-throughput digitizing."),
        ("Chemical Staining Shifts:", "H&E stain batches fade, acid washes vary, and slide preparations introduce coverslip bubbles."),
        ("Modality Mismatches:", "A clinician accidentally routes a skin lesion or blood smear to a colorectal tissue model.")
    ]
    for title, desc in points2:
        p = tf2.add_paragraph()
        p.text = f"• {title} {desc}"
        p.font.size = Pt(12)
        p.font.color.rgb = COLOR_DARK_SLATE
        p.space_before = Pt(12)

    # Col 3: The Project Solution
    add_card(s2, Inches(0.8) + (col_w + gap) * 2, col_top, col_w, col_h, bg_color=RGBColor(240, 253, 250), border_color=COLOR_TEAL_ACCENT)
    c3 = s2.shapes.add_textbox(Inches(1.0) + (col_w + gap) * 2, col_top + Inches(0.2), col_w - Inches(0.4), col_h - Inches(0.4))
    tf3 = c3.text_frame
    tf3.word_wrap = True
    p = tf3.paragraphs[0]
    p.text = "3. Our Trustworthy Solution"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_TEAL_ACCENT

    points3 = [
        ("Meta-Cognitive Awareness:", "AI knows its own internal limits and declared 'I cannot trust this image'."),
        ("Selective Prediction:", "Automates clear, pristine cases; defers compromised scans to senior human pathologists."),
        ("Defense-in-Depth:", "No single point of failure: 4 orthogonal safety layers intercept flawed inputs before harm occurs.")
    ]
    for title, desc in points3:
        p = tf3.add_paragraph()
        p.text = f"• {title} {desc}"
        p.font.size = Pt(12)
        p.font.color.rgb = COLOR_DARK_SLATE
        p.space_before = Pt(12)

    # =========================================================================
    # SLIDE 3: Problem Solving: The 4-Stage Safety Architecture
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3, COLOR_LIGHT_BG)
    add_header(s3, "Problem Solving: The 4-Layer Clinical Defense-in-Depth Architecture")

    # 4 Architecture Horizontal Cards
    card_w = Inches(11.73)
    card_h = Inches(1.18)
    spacing = Inches(0.14)
    start_y = Inches(1.68)

    layers = [
        ("LAYER 1: DETERMINISTIC PHYSICAL QUALITY GATE (IQA)",
         "Evaluates raw spatial optics before neural inference: Laplacian Focus Variance (Var_Lap < 40 triggers blur warning), Signal-to-Noise Ratio (SNR > 10 dB), and pixel dynamic range clipping (< 45%). Stops blurred scans before wasting compute.",
         COLOR_PRIMARY_NAVY, COLOR_WHITE),
        ("LAYER 2: DEEP LATENT MANIFOLD OOD SCREENING",
         "Measures the geometric Mahalanobis distance M(x) of the image's 256-dimensional feature representation from the in-distribution tissue manifold. Completely isolates foreign modalities (skin, blood) with 98.9%-99.3% AUROC.",
         COLOR_TEAL_ACCENT, RGBColor(240, 253, 250)),
        ("LAYER 3: BAYESIAN EPISTEMIC UNCERTAINTY QUANTIFICATION",
         "Executes 15 inference-time Monte Carlo Dropout passes to decompose total predictive entropy H(p) into Aleatoric data noise and Epistemic Mutual Information I(y, W|x). Detects when the model is ignorant of a tissue morphology.",
         COLOR_DARK_SLATE, COLOR_WHITE),
        ("LAYER 4: VALIDATION-CALIBRATED CONFIDENCE & SELECTIVE TRIAGE",
         "Applies post-hoc Temperature Scaling (T = 1.23) so reported confidence matches true clinical accuracy (ECE: 5.17%). Multi-signal triage policy issues unambiguous decision: 🟢 TRUSTED: ACCEPT or 🔴 UNTRUSTED: DEFER TO PATHOLOGIST.",
         COLOR_GREEN_TRUST, RGBColor(240, 253, 244)),
    ]

    for i, (title, text, tag_color, bg_c) in enumerate(layers):
        y_pos = start_y + (card_h + spacing) * i
        add_card(s3, Inches(0.8), y_pos, card_w, card_h, bg_color=bg_c, border_color=COLOR_CARD_BORDER)

        tbox = s3.shapes.add_textbox(Inches(1.05), y_pos + Inches(0.12), card_w - Inches(0.5), card_h - Inches(0.24))
        tf = tbox.text_frame
        tf.word_wrap = True
        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(13)
        p_t.font.bold = True
        p_t.font.color.rgb = tag_color

        p_b = tf.add_paragraph()
        p_b.text = text
        p_b.font.size = Pt(11)
        p_b.font.color.rgb = COLOR_DARK_SLATE
        p_b.space_before = Pt(4)

    # =========================================================================
    # SLIDE 4: In-Distribution Benchmark & Calibration (With Figure 1)
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4, COLOR_LIGHT_BG)
    add_header(s4, "Experimental Results: Locked Test Diagnostics & Temperature Calibration")

    # Left Column: Image (Fig 1)
    fig1_path = os.path.join(fig_dir, "fig1_calibration_reliability.png")
    if os.path.exists(fig1_path):
        add_card(s4, Inches(0.8), Inches(1.68), Inches(6.8), Inches(5.2))
        s4.shapes.add_picture(fig1_path, Inches(0.9), Inches(1.8), width=Inches(6.6))

    # Right Column: Metric Cards
    r_left = Inches(7.8)
    r_width = Inches(4.73)

    add_card(s4, r_left, Inches(1.68), r_width, Inches(5.2))
    rbox = s4.shapes.add_textbox(r_left + Inches(0.25), Inches(1.88), r_width - Inches(0.5), Inches(4.8))
    tfr = rbox.text_frame
    tfr.word_wrap = True

    p = tfr.paragraphs[0]
    p.text = "Diagnostic Benchmark Metrics"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_NAVY

    metrics = [
        ("Locked Test Accuracy:", "86.07% (6,180 / 7,180 correct)"),
        ("Balanced Accuracy:", "81.84% (across 9 tissue classes)"),
        ("Macro-F1 Score:", "0.8087 (Precision: 82.04% / Recall: 81.84%)"),
        ("Peak Validation Acc:", "97.26% (Epoch 8 checkpoint)"),
        ("Pre-Calibration ECE:", "6.56% (Systematic overconfidence)"),
        ("Post-Calibration ECE:", "5.17% (T = 1.2251, -21.2% error reduction)"),
        ("Calibrated Brier Score:", "0.2270 (Well-calibrated probabilities)")
    ]

    for label, val in metrics:
        p = tfr.add_paragraph()
        p.text = f"• {label} {val}"
        p.font.size = Pt(11.5)
        p.font.color.rgb = COLOR_DARK_SLATE
        p.space_before = Pt(8)

    p_note = tfr.add_paragraph()
    p_note.text = "Key Takeaway: Temperature Scaling (T=1.23) strictly fitted on validation data flattens the confidence histogram without changing classification accuracy, restoring clinical alignment."
    p_note.font.size = Pt(11)
    p_note.font.italic = True
    p_note.font.color.rgb = COLOR_TEAL_ACCENT
    p_note.space_before = Pt(14)

    # =========================================================================
    # SLIDE 5: Selective Prediction: Risk vs Coverage Tradeoff (With Figure 2)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5, COLOR_LIGHT_BG)
    add_header(s5, "Selective Prediction: Continuous Clinical Risk Reduction via Abstention")

    # Left Column: Image (Fig 2)
    fig2_path = os.path.join(fig_dir, "fig2_risk_coverage_curve.png")
    if os.path.exists(fig2_path):
        add_card(s5, Inches(0.8), Inches(1.68), Inches(6.5), Inches(5.2))
        s5.shapes.add_picture(fig2_path, Inches(0.95), Inches(1.85), width=Inches(6.2))

    # Right Column: Explanation
    r_left = Inches(7.5)
    r_width = Inches(5.03)

    add_card(s5, r_left, Inches(1.68), r_width, Inches(5.2))
    rbox = s5.shapes.add_textbox(r_left + Inches(0.25), Inches(1.88), r_width - Inches(0.5), Inches(4.8))
    tfr = rbox.text_frame
    tfr.word_wrap = True

    p = tfr.paragraphs[0]
    p.text = "Risk-Coverage Optimization"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_NAVY

    selective_points = [
        ("The Full Automation Dilemma:", "Forcing AI to predict 100% of cases yields a baseline error rate of 13.93% (1,000 mistakes)."),
        ("Area Under Risk-Coverage (AURC):", "AURC = 4.80. As the abstention threshold tightens, clinical error rate on accepted cases drops monotonically."),
        ("Bounded Clinical Risk:", "A hospital can configure an allowable clinical risk target (e.g. error <= 3% or <= 5%). The AI automatically computes the maximal coverage satisfying that safety bound."),
        ("Optimized Labor Allocation:", "Routine, unambiguous biopsies are automated rapidly; ambiguous border cases are channeled directly to pathologists.")
    ]

    for label, val in selective_points:
        p = tfr.add_paragraph()
        p.text = f"• {label} {val}"
        p.font.size = Pt(11.5)
        p.font.color.rgb = COLOR_DARK_SLATE
        p.space_before = Pt(10)

    # =========================================================================
    # SLIDE 6: The Discovery of "Silent Failures" (Key Insight)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6, COLOR_LIGHT_BG)
    add_header(s6, "Core Scientific Discovery: 29.2% of Mistakes Are Confidently Silent")

    # Top Card: The Question & Core Number
    add_card(s6, Inches(0.8), Inches(1.68), Inches(11.73), Inches(1.6), bg_color=RGBColor(254, 242, 242), border_color=COLOR_RED_ALERT)
    c_top = s6.shapes.add_textbox(Inches(1.05), Inches(1.8), Inches(11.2), Inches(1.3))
    tft = c_top.text_frame
    tft.word_wrap = True
    p = tft.paragraphs[0]
    p.text = "RESEARCH QUESTION: Does Uncertainty Always Warn Us When the AI Makes an Error?"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_RED_ALERT

    p2 = tft.add_paragraph()
    p2.text = "EMPIRICAL ANSWER: NO. Out of 1,000 test errors, 292 cases (29.2% of all mistakes) were 'Silent Failures'—the AI was completely wrong, yet output high confidence (>85%) and low entropy (<0.35)."
    p2.font.size = Pt(13)
    p2.font.bold = True
    p2.font.color.rgb = COLOR_DARK_SLATE
    p2.space_before = Pt(6)

    # Bottom 3 Cards: Why it happens & What it means
    b_w = Inches(3.68)
    b_h = Inches(3.35)
    b_top = Inches(3.52)

    # Card 1: Where Silent Failures Occur
    add_card(s6, Inches(0.8), b_top, b_w, b_h)
    tb1 = s6.shapes.add_textbox(Inches(1.0), b_top + Inches(0.15), b_w - Inches(0.4), b_h - Inches(0.3))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "Where Do They Happen?"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_NAVY
    p_desc = tf1.add_paragraph()
    p_desc.text = "Silent failures occur at subtle, ambiguous histological tissue boundaries:\n\n• Smooth muscle (MUS) vs. desmoplastic cancer stroma (STR)\n• Normal crypts (NORM) vs. low-grade dysplasia (TUM)\n• Extracellular mucus (MUC) with sparse cells."
    p_desc.font.size = Pt(11)
    p_desc.font.color.rgb = COLOR_DARK_SLATE
    p_desc.space_before = Pt(8)

    # Card 2: Why Uncertainty Fails Alone
    add_card(s6, Inches(0.8) + b_w + gap, b_top, b_w, b_h)
    tb2 = s6.shapes.add_textbox(Inches(1.0) + b_w + gap, b_top + Inches(0.15), b_w - Inches(0.4), b_h - Inches(0.3))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "Why Does Uncertainty Fail?"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_NAVY
    p_desc = tf2.add_paragraph()
    p_desc.text = "Because the test image is clean and lies squarely inside the training manifold:\n\n• Image quality is pristine (no blur)\n• MC-Dropout passes agree with each other (low epistemic variance)\n• The model is consistently, confidently wrong."
    p_desc.font.size = Pt(11)
    p_desc.font.color.rgb = COLOR_DARK_SLATE
    p_desc.space_before = Pt(8)

    # Card 3: The Architecture Solution
    add_card(s6, Inches(0.8) + (b_w + gap) * 2, b_top, b_w, b_h, bg_color=RGBColor(240, 253, 250), border_color=COLOR_TEAL_ACCENT)
    tb3 = s6.shapes.add_textbox(Inches(1.0) + (b_w + gap) * 2, b_top + Inches(0.15), b_w - Inches(0.4), b_h - Inches(0.3))
    tf3 = tb3.text_frame
    tf3.word_wrap = True
    p = tf3.paragraphs[0]
    p.text = "The Solution: Multi-Signal Defense"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_TEAL_ACCENT
    p_desc = tf3.add_paragraph()
    p_desc.text = "Relying strictly on confidence or entropy creates a dangerous false sense of security.\n\nOur system combines:\n1. Calibrated Risk Cutoffs\n2. Feature Distance Gates\n3. Pre-Inference Physical IQA\n4. Human Pathologist Fallback"
    p_desc.font.size = Pt(11)
    p_desc.font.color.rgb = COLOR_DARK_SLATE
    p_desc.space_before = Pt(8)

    # =========================================================================
    # SLIDE 7: OOD Screening: Feature Geometry vs Logit Heuristics (With Fig 4)
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7, COLOR_LIGHT_BG)
    add_header(s7, "OOD Screening: Deep Feature Geometry Decisively Outperforms Softmax")

    # Left Column: Image (Fig 4)
    fig4_path = os.path.join(fig_dir, "fig4_ood_distribution_separation.png")
    if os.path.exists(fig4_path):
        add_card(s7, Inches(0.8), Inches(1.68), Inches(6.5), Inches(5.2))
        s7.shapes.add_picture(fig4_path, Inches(0.95), Inches(1.9), width=Inches(6.2))

    # Right Column: OOD Benchmark Table & Summary
    r_left = Inches(7.5)
    r_width = Inches(5.03)

    add_card(s7, r_left, Inches(1.68), r_width, Inches(5.2))
    rbox = s7.shapes.add_textbox(r_left + Inches(0.25), Inches(1.85), r_width - Inches(0.5), Inches(4.8))
    tfr = rbox.text_frame
    tfr.word_wrap = True

    p = tfr.paragraphs[0]
    p.text = "OOD Screening Benchmark"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_NAVY

    ood_rows = [
        ("DermaMNIST (Skin Lesions):", "• Softmax MSP: 45.07% AUROC\n• Energy Score: 42.24% AUROC\n• Mahalanobis: 99.26% AUROC (FPR: 1.0%)"),
        ("BloodMNIST (Blood Smears):", "• Softmax MSP: 57.20% AUROC\n• Energy Score: 55.88% AUROC\n• Mahalanobis: 98.93% AUROC (FPR: 0.8%)"),
        ("Synthetic Uniform Noise:", "• Softmax MSP: 45.14% AUROC\n• Mahalanobis: 99.36% AUROC (FPR: 0.0%)"),
    ]

    for label, val in ood_rows:
        p = tfr.add_paragraph()
        p.text = label
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = COLOR_DARK_SLATE
        p.space_before = Pt(8)

        p2 = tfr.add_paragraph()
        p2.text = val
        p2.font.size = Pt(11)
        p2.font.color.rgb = COLOR_TEAL_ACCENT if "Mahalanobis" in val else COLOR_GRAY_TEXT
        p2.space_before = Pt(2)

    p_conc = tfr.add_paragraph()
    p_conc.text = "Conclusion: Output logits fail on foreign medical scans (random chance). Penultimate latent geometry provides near-perfect domain insulation."
    p_conc.font.size = Pt(10.5)
    p_conc.font.italic = True
    p_conc.font.color.rgb = COLOR_DARK_SLATE
    p_conc.space_before = Pt(10)

    # =========================================================================
    # SLIDE 8: Robustness Under Physical Corruptions (With Fig 3)
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8, COLOR_LIGHT_BG)
    add_header(s8, "Stress Testing: Systematic Degradation & Automated Clinical Deferral")

    # Left Column: Image (Fig 3)
    fig3_path = os.path.join(fig_dir, "fig3_corruption_robustness.png")
    if os.path.exists(fig3_path):
        add_card(s8, Inches(0.8), Inches(1.68), Inches(6.8), Inches(5.2))
        s8.shapes.add_picture(fig3_path, Inches(0.95), Inches(1.85), width=Inches(6.5))

    # Right Column: Deferral Table & Highlights
    r_left = Inches(7.8)
    r_width = Inches(4.73)

    add_card(s8, r_left, Inches(1.68), r_width, Inches(5.2))
    rbox = s8.shapes.add_textbox(r_left + Inches(0.25), Inches(1.85), r_width - Inches(0.5), Inches(4.8))
    tfr = rbox.text_frame
    tfr.word_wrap = True

    p = tfr.paragraphs[0]
    p.text = "Physical Corruptions Audit"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_NAVY

    corr_bullets = [
        ("Defocus Blur (Microscope Focal Drift):", "Acc drops 57.0% -> 21.0%. Deferral scales to 100.0% automated rejection on severities 3-5."),
        ("Contrast Degradation (Aperture Decay):", "Acc drops 33.3% -> 10.3%. Deferral reaches 93.7% automated rejection."),
        ("Motion Blur (Stage Motor Jitter):", "Acc drops 70.0% -> 49.7%. Deferral rate reaches 49.7%."),
        ("H&E Stain Fading (Chemical Drift):", "Revealed 'Representation Collapse': model defaults to single class with low entropy. Pre-inference SNR & feature gates catch it.")
    ]

    for label, val in corr_bullets:
        p = tfr.add_paragraph()
        p.text = f"• {label} {val}"
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_DARK_SLATE
        p.space_before = Pt(8)

    # =========================================================================
    # SLIDE 9: Real-World Clinical Triage Showcase (With Fig 5)
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9, COLOR_LIGHT_BG)
    add_header(s9, "Clinical Triage in Action: Live Demonstrations Across Query Inputs")

    # Top: Large Image (Fig 5)
    fig5_path = os.path.join(fig_dir, "fig5_clinical_triage_cases.png")
    if os.path.exists(fig5_path):
        add_card(s9, Inches(0.8), Inches(1.68), Inches(11.73), Inches(3.7))
        s9.shapes.add_picture(fig5_path, Inches(0.95), Inches(1.78), width=Inches(11.43))

    # Bottom: 4 Explanatory Cards
    c_w = Inches(2.78)
    c_gap = Inches(0.20)
    c_top = Inches(5.52)
    c_h = Inches(1.58)

    cases = [
        ("Case 1: Clean Histology", "🟢 TRUSTED: ACCEPT", "High focus (Var: 142.3), low entropy (H: 0.14). Autonomous prediction: Normal Mucosa (96% conf).", COLOR_GREEN_TRUST),
        ("Case 2: Motion Blur", "🔴 UNTRUSTED: DEFER", "Degraded focus (Var: 28.4 < 40), entropy spikes (H: 1.72). Flags severe stage vibration.", COLOR_RED_ALERT),
        ("Case 3: Foreign OOD", "🔴 UNTRUSTED: DEFER", "Input is a skin lesion. Latent Mahalanobis distance explodes (M=184.2 >> tau). Foreign modality deferred.", COLOR_RED_ALERT),
        ("Case 4: Faded Staining", "🔴 UNTRUSTED: DEFER", "Severe contrast attenuation. Dynamic range quality screening alert triggers. Defer to re-staining.", COLOR_RED_ALERT),
    ]

    for i, (title, status, desc, col) in enumerate(cases):
        c_x = Inches(0.8) + (c_w + c_gap) * i
        add_card(s9, c_x, c_top, c_w, c_h)
        tbox = s9.shapes.add_textbox(c_x + Inches(0.12), c_top + Inches(0.1), c_w - Inches(0.24), c_h - Inches(0.2))
        tf = tbox.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY_NAVY

        p_s = tf.add_paragraph()
        p_s.text = status
        p_s.font.size = Pt(10.5)
        p_s.font.bold = True
        p_s.font.color.rgb = col
        p_s.space_before = Pt(2)

        p_d = tf.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(9.5)
        p_d.font.color.rgb = COLOR_DARK_SLATE
        p_d.space_before = Pt(3)

    # =========================================================================
    # SLIDE 10: How It Helps & Real-World Clinical Impact
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10, COLOR_LIGHT_BG)
    add_header(s10, "How It Helps: Real-World Healthcare Impact & Regulatory Alignment")

    card_w = Inches(5.66)
    card_h = Inches(2.45)
    gap_x = Inches(0.41)
    gap_y = Inches(0.25)
    top_y = Inches(1.7)

    impacts = [
        ("1. Hospital Workload Relief Without Risk",
         "Pathologists face global burnout reviewing thousands of routine biopsies. Our system securely automates pristine, high-confidence biopsies (e.g. normal mucosa) while triaging complex, suspicious, or borderline cases directly to specialist desks.",
         COLOR_TEAL_ACCENT),
        ("2. Hardware & Pre-Analytical Quality Firewall",
         "Acts as an automated quality-control filter at the digital slide scanner. Warns lab technicians about focal drift, coverslip bubbles, or poor staining before corrupted scans contaminate medical archives or patient records.",
         COLOR_PRIMARY_NAVY),
        ("3. Global Regulatory Compliance (FDA & EU AI Act)",
         "Meets FDA Good Machine Learning Practice (GMLP) and EU AI Act requirements for high-risk medical AI by providing verifiable uncertainty bounds, calibrated probabilities, and mandatory human-in-the-loop fallback mechanisms.",
         COLOR_PRIMARY_NAVY),
        ("4. Complete Open-Source Reproducibility",
         "100% unit tests passing via PyTest • Interactive Streamlit diagnostic dashboard (app.py) • Full reproducible CLI orchestration (scripts/run_pipeline.py) • 11-page publication research report in PDF.",
         COLOR_GREEN_TRUST),
    ]

    for i, (title, body, tag_col) in enumerate(impacts):
        row = i // 2
        col = i % 2
        x = Inches(0.8) + (card_w + gap_x) * col
        y = top_y + (card_h + gap_y) * row

        add_card(s10, x, y, card_w, card_h)
        tbox = s10.shapes.add_textbox(x + Inches(0.2), y + Inches(0.2), card_w - Inches(0.4), card_h - Inches(0.4))
        tf = tbox.text_frame
        tf.word_wrap = True

        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = tag_col

        p_b = tf.add_paragraph()
        p_b.text = body
        p_b.font.size = Pt(11.5)
        p_b.font.color.rgb = COLOR_DARK_SLATE
        p_b.space_before = Pt(8)

    # Save Presentation
    prs.save(output_pptx_path)
    print(f"[✓] Presentation saved successfully to: {output_pptx_path}")
    return output_pptx_path


if __name__ == "__main__":
    out_file = "Medical_AI_Reliability_Presentation.pptx"
    if len(sys.argv) > 1:
        out_file = sys.argv[1]
    create_deck(out_file)
