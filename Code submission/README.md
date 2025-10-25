# 🧬 ML4G Project 1 — Predicting Gene Expression from Epigenomic Signals

> Predict gene expression for unseen cell lines using epigenomic features.
> LightGBM + robust preprocessing, complementary features, and chromosome-aware validation.

---

## 📘 What this repo does

* Loads histone marks (H3K27ac, H3K4me3, H3K27me3, H3K36me3, H3K4me1, H3K9me3) and DNase from BED (peaks) and bigWig (continuous signal).
* Preprocesses signals with `log1p` → z-score to tame outliers and reduce batch effects.
* Builds feature sets that capture promoter vs gene-body patterns, activation vs repression balance, topology, and cross-layer interactions.
* Trains LightGBM in a two-stage setup (binary non-zero classifier + rank regressor).
* Uses Leave-One-Chromosome-Out (LOCO) + chr-based K-Fold and probability masking to generalize across genomic regions.
* Creates the submission ZIP in the required format.

---

## 📂 Data

* CAGE: `gex` targets and gene info (`gene_name, chr, gene_start, gene_end, TSS_start, TSS_end, strand`)
* ChIP-seq marks: H3K27me3, H3K4me1, H3K4me3, H3K27ac, H3K36me3, H3K9me3
* DNase-seq: chromatin accessibility
* Each assay available as BED (peaks) and bigWig (base-wise signal)

---

## 🧱 Preprocessing

Why: biological signals are heavy-tailed; different cell lines show scale shifts (batch effects).

1. `log1p` on raw tracks to compress outliers while preserving mid/low signals
2. Per-mark z-score (on top of log1p) to align scales across cell lines/assays
   I call these “log-z” features in code.

---

## 🧩 Feature Engineering (high-level)

I fuse BED and bigWig because they are complementary.

* BED (peak-centric): peak counts, coverage ratio, peak density/entropy, nearest-peak distances → structure/organization
* bigWig (signal-centric): mean/std/min/max, gradients over promoter/TSS and gene body → signal magnitude/shape

Implemented families (see `feature_engineer.py`):

* Promoter vs gene body: means/stds, deltas, sums
* Ratios: `{mark}_tss / {mark}_gene` (scale-robust enrichment)
* Activation–repression balance: `H3K27ac − H3K27me3`, `H3K4me3 − H3K9me3`, plus activation/repression indices
* Promoter entropy & variability: std across marks; Shannon entropy of normalized TSS means
* Openness vs repression indices: mean activating vs mean repressive TSS signals
* Strand flags: `strand_is_plus`, `strand_is_minus`
* Cross-mark interactions: products/ratios/diffs of TSS means/stds; promoter–gene cross terms
* TSS geometry: distances from TSS midpoint to gene boundaries (absolute and length-normalized)
* Advanced chromatin: `(TSS−gene)/gene_length`, `(TSS−gene)/(TSS+gene)`, `H3K27ac_gene × H3K4me3_tss`, `H3K27ac × DNase` synergy/ratio, bivalency (K27ac vs K27me3), promoter–gene coherence (row-wise Pearson), entropy diversity
* Cross-layer (BED × bigWig): `bw_mean/peak_density`, `bw_entropy − peak_entropy`, `peak_density × bw_mean`, bigWig promoter–gene balances

---

## 🧪 Feature Selection (cross–cell-line & robust)

After building a large feature set from BED + bigWig, I select a subset of the features that **transfers across cell lines** (so it’s more likely to work on X3). I use **cross–cell-line SHAP** to rank features by importance and stability, then optionally prune redundancy.

**How I select features**
1. **Train across cells (directional):** Run X1→X2 and X2→X1 with chromosome-aware folds.
2. **Compute SHAP per direction & task:** For both **binary** (non-zero) and **regression** (rank), compute mean |SHAP| on the validation splits.
3. **Aggregate & stabilize:** Average SHAP across folds, then merge directions to score **stability** (features that stay important in both directions).
4. **Take the union of top-N:** From binary + regression and both directions; de-duplicate while preserving order.
5. **Optional pruning:**  
   - **Sparsity filter:** drop features with very low non-zero rates across cells.  
   - **Correlation filter:** identify highly correlated pairs and keep one representative.

---

## 🧠 Model & Why LightGBM

I use **LightGBM (LGBM)** for both binary classification (is expression > 0?) and rank regression (predict normalized global rank). The reason I used this model is because it is fast, strong on tabular data, and captures non-linear interactions well. Stacking more model families can help, but in this setup LGBM alone performs strongly.

---

## 🎯 Training Strategy

### Two-stage prediction

* EDA shows many zeros (e.g., X1: 58.02% zeros; X2: 49.61% zeros)
* Stage 1: binary classifier (non-zero vs zero)
* Stage 2: rank regressor predicting normalized global rank; matches Spearman (default, average-ties) and is robust to heavy tails

### CV & Ensembling

* LOCO outer loop + chr-based K-Fold inner loop
* I train and stack five setups:

  1. X1+X2 pooled: mix X1+X2 for train/val
  2. X1-only: train/val on X1
  3. X2-only: train/val on X2
  4. X1 cross cell line validation: train on X1, val on X2 (Didn't use in stacking. See result section for detail.)
  5. X2 cross cell line validation: train on X2, val on X1 (Didn't use in stacking. See result section for detail.)

* Stacking rule:

  * final rank = mean of the 3 rank predictions
  * final mask = mean of the 3 classifier probability, thresholded at 0.40
  * final prediction = rank × mask
    (Union/interaction masks were tested; mean + 0.4 worked best)
    
| Different threshold vs performance | Different strategies of masking |
|:-------------------------:|:---------------:|
| <img src="/plot/threshold%20of%20binary.png" width="420"/> | <img src="/plot/mask%20methods%20LOCO%20k-fold.png" width="440"/> |


---

## 🧪 Validation

* Leave-Chromosome-Out (chr2–chr22): generalization to unseen regions; avoids positional leakage
* K-Fold on genes: overall robustness


---

## Results


| Mask on unseen cell line | Mask on seen cell line|
|:-------------------------:|:---------------:|
| <img src="/plot/see%20mask%20on%20unseen%20cell%20line.png" width="490"/> | <img src="/plot/see%20mask%20on%20the%20same%20cell%20line.png" width="420"/> |


The mask significantly improves predictions in most settings. One exception: when mixing X1+X2 to train models, masking can hurt. However, this does not necessarily indicate degradation. Mixing encourages the model to learn a more general decision boundary across two data domains (X1 and X2) that have different noise characteristics and distributions. Even on the same chromosome and the same gene, X1 and X2 can show distinct distributions in histone marks (see EDA.ipynb). To remain reliable under “the same gene but different distributions,” the model may trade a small amount of single-domain performance in the short term for better generalization. Based on this, I therefore also stack the mixed (X1+X2) binary and regression models with the within-cell-line models (averaging by task) so they complement each other’s weaknesses and improve cross-domain robustness.

![Performance analysis cross vs within cell line](./plot/cross%20vs%20within%20cell%20line.png)

As shown in the figure, the gap between cross– and within–cell-line validation is minimal. After log-z score normalization, the distributional difference between X1 and X2 is largely reduced for histone-mark features. Consequently, many predictive signals are cell-line agnostic or partially conserved across X1 and X2, which explains why the two validation schemes yield similar accuracy when predicting a single target cell line.

Given this, I use within–cell-line validation as the basis for model selection and stacking because it avoids domain averaging and lets the model fully exploit cell-line–specific patterns that remain after normalization. To mitigate any remaining trade-offs, I also stack the X1+X2 mixed models alongside the within–cell-line models, allowing them to complement each other’s weaknesses and preserving on-distribution calibration while adding cross-domain robustness.

### Ablations

* BED + bigWig > bigWig-only > BED-only

| Both direction TSS | One direction TSS |
|:----------:|:--------:|
| <img src="/plot/compare%20training%20on%20diff%20dataset%28Both%20side%29.png" width="95%"/> | <img src="/plot/compare%20training%20on%20diff%20dataset%28one%20side%29.png" width="95%"/> |

* TSS window: strand-aware one-sided outperforms symmetric both-sides. 100 bp is the sweet spot; performance drops beyond 100 bp (tested 50–5000 bp)
  
| Both-sided VS one-sided TSS | The size of TSS |
|:-------------------------:|:---------------:|
| <img src="/plot/tss%20windows%20chrom_kfold.png" width="420"/> | <img src="/plot/tss%20one%20side.png" width="435"/> |


---

## 🔍 Model Interpretation (SHAP on LGBM)

![SHAP analysis](/plot/SHAP%20analyze.png)

According to the top important features, we could see that the model focuses on how open the promoter is (DNase) and whether it’s more open than the gene body. If the promoter is uniformly open and more open than the body, the gene is likely on.

---

## ⚙️ Environment

* **Platform:** macOS (Apple Silicon **M1**). All experiments and scripts were developed and tested on Mac (M1).
* **Important:** The **bigWig** processing pipeline depends on `pyBigWig`. Native Windows setups are **not supported/reliable** for this step. For reproduction on Windows, use **WSL2 (Ubuntu)** or a Linux/macOS machine.
* **Python / deps:** See `Code submission/environment.yml` for exact versions and packages.

---

## 🔁 Reproducibility

* Scripts are modular and numbered; run them **in order**.
* If you are on Windows, run the feature extraction steps that use bigWig **inside WSL2/Linux** (or on macOS) to avoid `pyBigWig` issues.

---

## 📜 License

This project is licensed under **CC BY-NC 4.0** (non-commercial, attribution required).  

* **Non-commercial use only.** Any commercial use is **prohibited** without prior written permission.

* **Citation required.** If you use this code or derived results in academic work, please **cite this repository**:

  > Wang, Ding-Yang. *ML4G Project 1 – Predicting Gene Expression from Epigenomic Signals*. GitHub repository, 2025.

  Example (BibTeX):
  ```
  @misc{wang2025ml4g,
  author = {Wang, Ding-Yang},
  title  = {ML4G Project 1 – Predicting Gene Expression from Epigenomic Signals},
  year   = {2025},
  howpublished = {\url{[https://github.com/](https://github.com/)<your-repo>}},
  note = {Non-commercial use; citation required}
  }
  ```
