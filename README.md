# 🧬 ML4G Project 1 — Predicting Gene Expression from Epigenomic Signals

> Predict gene expression for **unseen cell lines** using epigenomic features.
> Fast, simple, and strong: **LightGBM** with robust preprocessing, careful features, and chromosome-aware validation.

---

## 📘 What this repo does

* Loads **histone marks** (H3K27ac, H3K4me3, H3K27me3, H3K36me3, H3K4me1, H3K9me3) and **DNase** (accessibility) from **BED** (peaks) and **bigWig** (continuous signal).
* **Preprocesses** signals with `log1p` → **z-score** to tame outliers and reduce batch effects.
* Builds **feature sets** that capture promoter vs gene-body patterns, activation vs repression balance, topology, and cross-layer interactions.
* Trains **LightGBM** models in a **two-stage** setup (binary non-zero classifier + rank regressor).
* Uses **Leave-One-Chromosome-Out (LOCO)** + **chr-based K-Fold** and **probability masking** to generalize across genomic regions.
* Creates the **submission ZIP** in the exact format required.

---

## 📂 Data

* **CAGE** (Cap Analysis of Gene Expression): target `gex` and `gex_rank`.
* **ChIP-seq** marks: H3K27me3, H3K4me1, H3K4me3, H3K27ac, H3K36me3, H3K9me3.
* **DNase-seq**: chromatin accessibility.
* Each assay available as **BED** (peaks) and **bigWig** (base-wise signal).

---

## 🧱 Preprocessing

Why: biological signals are **heavy-tailed**; different cell lines have **scale shifts** (batch effects).

1. **`log1p`** on raw tracks → compresses outliers more than mid/low signals.
2. **Per-mark z-score** (on top of log1p) → aligns scales across cell lines/assays.
   We refer to these as **“log-z”** features in code.

---

## 🧩 Feature Engineering (high-level)

We fuse **BED** and **bigWig** because they are complementary:

* **BED (peak-centric):** peak counts, coverage ratio, peak density/entropy, nearest peak distances — captures **structure/organization**.
* **bigWig (signal-centric):** mean/std/min/max, gradients over **promoter/TSS** and **gene body** — captures **signal magnitude/shape**.

**Feature families (see scripts for details):**

* **Promoter vs gene body:** means/stds, **deltas** and **sums**.
* **Ratios:** `{mark}_tss / {mark}_gene` (scale-robust enrichment).
* **Activation–repression balance:** e.g., `H3K27ac − H3K27me3`, `H3K4me3 − H3K9me3`, plus activation/repression indices.
* **Promoter entropy & variability:** std across marks; Shannon entropy of normalized TSS means.
* **Openness vs repression indices:** mean of activating vs repressive TSS signals.
* **Strand flags:** `strand_is_plus`, `strand_is_minus`.
* **Cross-mark interactions:** products/ratios/diffs of TSS means/stds; promoter–gene cross terms.
* **TSS geometry:** distances from TSS midpoint to gene boundaries (absolute & length-normalized).
* **Advanced chromatin:** `(TSS−gene)/gene_length`, `(TSS−gene)/(TSS+gene)`, `H3K27ac_gene × H3K4me3_tss`, `H3K27ac × DNase` synergy/ratio, bivalency (K27ac vs K27me3), promoter–gene **coherence** (row-wise Pearson), entropy **diversity**.
* **Cross-layer (BED × bigWig):** `bw_mean/peak_density`, `bw_entropy − peak_entropy`, `peak_density × bw_mean`, promoter–gene balances on bigWig.

**Ablations (what worked best):**

* **BED + bigWig > bigWig-only > BED-only**.
* **TSS window:** **strand-aware one-sided** outperforms symmetric both-sides. **100 bp** is the sweet spot; performance drops beyond 100 bp (tested 50–5000 bp).

---

## 🧠 Model & Why LightGBM

I use **LightGBM (LGBM)** for both **binary classification** (is expression > 0?) and **rank regression** (predict normalized global rank) because it’s **fast**, consistently strong on **tabular** data, and captures **non-linear interactions** well.
Stacking more model families can help, but in practice **LGBM alone is strong enough** here.

---

## 🎯 Training Strategy

### Two-stage prediction

* EDA shows many zeros (e.g., X1: 58.02% zeros; X2: 49.61% zeros).
* **Stage 1:** Binary classifier (non-zero vs zero).
  CV means: **AUC 0.9158**, **ACC 0.8313**, **F1 0.8386**.
* **Stage 2:** Rank regressor predicting **normalized global rank**. This aligns with **Spearman (default, average-ties)** and is robust to heavy tails.

### CV & Ensembling

* **LOCO** outer loop + **chr-based K-Fold** inner loop.
* Three setups are trained and **stacked**:

  1. **X1+X2 pooled**,
  2. **X1-only**,
  3. **X2-only** (all with LOCO + chr-KFold).
* **Stacking rule:**

  * final **rank** = mean of the 3 rank predictions;
  * final **mask** = mean of the 3 classifier probabilities **≥ 0.40**;
  * **final prediction** = rank × mask.
    Union/interaction masks were tested; **mean+0.4** threshold worked best.

---

## ✅ Results (snapshot)

* **Fusion wins:** BED+bigWig consistently beats either alone.
* **TSS matters:** strand-aware **100 bp** works best; longer windows degrade performance.
* **Two-stage + mask:** best Spearman with classifier + rank regressor + **prob≥0.4** mask.

---

## 🔍 Model Interpretation (SHAP on LGBM)

**Plain takeaway:** the model mostly looks at **how open the promoter is (DNase)** and **whether it’s more open than the gene body**. If the promoter is uniformly open and **more** open than the body, the gene is **likely on**.

---

## 📦 Submission Format

* Output columns: **(row index, no header)**, **`gene_name`**, **`gex_predicted`**.
* File name inside ZIP must be **`gex_predicted.csv`**.
* ZIP must be named **`LastName_FirstName_Project1.zip`**.

Example:
///
///,gene_name,gex_predicted
0,CAPN9,0.1
1,ILF2,3.5
...
///

This repo includes code to **assert shapes/dtypes** and to **zip** in the exact format.

---

## 🧪 Validation

* **K-Fold** on genes → overall robustness.
* **Leave-Chromosome-Out (chr2–chr22)** → generalization to unseen regions; avoids positional leakage.

---

## ⚙️ Environment

```
```Python ≥ 3.8
Libraries: pandas, numpy, lightgbm, pyBigWig, scikit-learn, scipy, tqdm
```

---

## 🚀 Quickstart

///
///# 1) Prepare data (unzip & arrange)
ML4G_Project_1_Data/...

# 2) Preprocess (log1p → zscore; build features)

python scripts/preprocess/build_features.py

# 3) Train with LOCO + chr-KFold and export predictions

python scripts/train/train_lgbm_nested.py

# 4) Stack runs and create masked final predictions

python scripts/postprocess/stack_and_mask.py

# 5) Package submission (creates LastName_FirstName_Project1.zip)

python scripts/postprocess/make_submission.py
///

---

## 🔁 Reproducibility

* Deterministic chromosome splits; configs (thresholds, params) logged.
* Scripts are modular and numbered; follow them in order.
* Need the exact env spec (`yaml`)? Ping me—happy to include it.

---

## 📜 License

Add your license of choice here.

---

## 🙌 Acknowledgements

Course materials and datasets provided via Polybox/Moodle.
