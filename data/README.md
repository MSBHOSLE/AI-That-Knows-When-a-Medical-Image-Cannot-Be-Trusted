# Dataset Provenance, Licence & Specifications

## Primary In-Distribution (ID) Dataset: PathMNIST (MedMNIST v2)

- **Source:** MedMNIST v2 Benchmark (Yang et al., Nature Scientific Data 2023) based on the colorectal histology dataset from Kather et al. (2016, 2019).
- **Official URL:** https://medmnist.com/
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0). Commercial and non-commercial reuse allowed with attribution.
- **Modality:** Digital Pathology (H&E stained colorectal adenocarcinoma tissue sections).
- **Resolution:** $28 \times 28$ pixels, 3 color channels (RGB).
- **Total Images:** 107,180 images:
  - **Train split:** 89,996 images
  - **Validation split:** 10,004 images
  - **Test split (Locked):** 7,180 images
- **Classes (9 tissue types):**
  1. `adipose` (ADI) - Adipose tissue
  2. `background` (BACK) - Empty slide / background glass
  3. `debris` (DEB) - Necrotic debris, dust, slide artifacts
  4. `lymphocytes` (LYM) - Inflammatory infiltrate (immune cells)
  5. `mucus` (MUC) - Mucinous colloid / extracellular mucus
  6. `smooth_muscle` (MUS) - Smooth muscle layer
  7. `normal_colon_mucosa` (NORM) - Normal colonic epithelium & crypts
  8. `cancer_associated_stroma` (STR) - Desmoplastic tumor stroma
  9. `colorectal_adenocarcinoma_epithelium` (TUM) - Malignant glandular epithelium

---

## Out-of-Distribution (OOD) Candidate Datasets

To test whether the system detects foreign clinical modalities and rejects them rather than outputting misleading confidence:
1. **Near-Medical OOD: DermaMNIST (MedMNIST v2)**
   - Modality: Dermatoscopic images of pigmented skin lesions (HAM10000 dataset).
   - Channels: RGB ($28 \times 28$).
   - Completely disjoint label space (melanoma, nevus, basal cell carcinoma, etc.).
2. **Near-Medical OOD: BloodMNIST (MedMNIST v2)**
   - Modality: Microscopic peripheral blood smear cells (Acevedo et al.).
   - Channels: RGB ($28 \times 28$).
   - Distinct morphology (neutrophils, basophils, eosinophils, platelets).
3. **Synthetic / Far-OOD: Corrupted / Noise / Texture Drift**
   - Uniform noise, Gaussian blur, Stain variability, and non-medical benchmarks.

---

## Data Partition Hygiene & Anti-Leakage Rules
1. **Locked Test Split:** The test partition is strictly untouched during training, hyperparameter search, temperature calibration, and threshold selection.
2. **Validation Only for Post-Processing:** Temperature scaling parameter $T$, conformal quantiles, and selective abstention thresholds $\tau$ are fitted strictly on the validation set.
3. **Data Normalization:** Normalized using ImageNet statistics or dataset empirical mean and standard deviation:
   - Mean: `[0.7405, 0.5330, 0.7058]`
   - Std: `[0.1237, 0.1768, 0.1244]`
