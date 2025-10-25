# 🧬 ML4G Project 1 — Predicting Gene Expression from Epigenomic Signals

> Predict gene expression for unseen cell lines using epigenomic features.
> LightGBM + robust preprocessing, complementary features, and chromosome-aware validation.

**🏅 Result:** Ranked **1st** in **ETH Zürich – Machine Learning for Genomics (263-5351-00L, HS2025)** Project 1, supervised by **Prof. Valentina Boeva** (Head TA: **Lovro Rabuzin**).

See the full project description here: [ML4G_Project_1_2025.pdf](./ML4G_Project_1_2025.pdf)

If you only need a quick start, see the **[Reproduction](#-reproduction)** section below; detailed notebooks and plots live in **[`Code submission/`](./Code%20submission/)**.


---

## 📘 What this repo does (short)

* Loads histone marks (H3K27ac, H3K4me3, H3K27me3, H3K36me3, H3K4me1, H3K9me3) + DNase (BED + bigWig).
* Preprocesses with **log1p → per-mark z-score** (“log-z”) to reduce outliers/batch effects.
* Engineers promoter↔gene-body, activation↔repression, and cross-layer (BED × bigWig) features.
* Trains **LightGBM** in two stages: binary (non-zero) → rank regression (Spearman-friendly).
* Validates with **LOCO** + chr-aware K-Fold; applies **probability masking**.
* Final ensemble/stacking + ready-to-submit outputs.

👉 Full details, ablations, and SHAP interpretation live in **`Code submission/`**.

---

## 🗂 Full pipeline (see `Code submission/`)

All numbered notebooks, plots, configs, and outputs are under **Code submission/**, e.g.:

```
Code submission/
  0. reference autosome.ipynb
  1. Global mean std.ipynb
  2. extract bed.ipynb
  2. extract bigwig.ipynb
  2. merge y.ipynb
  3. merge bed bi.ipynb
  4. features engineer.ipynb
  5. merge y with all features.ipynb
  6. features selection.ipynb
  7. train lgbm.ipynb
  EDA.ipynb
  README.md
  environment.yml
  plot/ ... (figures)
  result/ ... (per-setup outputs + final ensemble)
```

---

## 🚀 Reproduce My Results

### Option A — Docker (no local setup)

```bash
docker pull deweywang/ml4g-project1:latest

docker run --rm -it \
  -p 8888:8888 \
  -v "$PWD":/workspace \
  deweywang/ml4g-project1:latest
```

Then open **[http://localhost:8888](http://localhost:8888)** (runs without token; use on trusted networks).
Your current folder is mounted to **/workspace** in the container.

---

### Option B — Run Locally (.venv / pip)

> ✅ Everything installs into **`.venv`** and won’t touch your global Python.

```bash
make init                  # Step 1: create venv and install deps
source .venv/bin/activate  # Step 2 (macOS/Linux)
# .venv\Scripts\activate   # Step 2 (Windows, WSL2 recommended for pyBigWig)
make lab                   # Step 3: launch JupyterLab
```

Inside JupyterLab: pick **Python (.venv) ml4g_project1** as the kernel.

Extras:

```bash
make clean   # remove the venv and kernel spec
make reset   # clean everything and reinitialize from scratch
```

> Note: `pyBigWig` requires Linux/macOS toolchains. On Windows, use **WSL2 (Ubuntu)** or Docker.

---

## 📂 Data

The **ML4G_Project_1_Data** folder data is available via **Polybox**:
**Link:** [https://polybox.ethz.ch/index.php/s/XJZFdLSZNHpEDLw](https://polybox.ethz.ch/index.php/s/XJZFdLSZNHpEDLw)
**Password:** `transcription_factor_2025`

Place downloaded data under the repo root (the Docker command above mounts it to `/workspace`).

---

## ⚙️ Environment

* Dev platform: **macOS (Apple Silicon M1)**.
* `pyBigWig` requires Linux/macOS toolchains. For Windows, use **WSL2** or Docker.
* Exact versions are pinned in **`Code submission/environment.yml`** (Docker image is built with conda-forge + bioconda).

---

## 📜 License & Citation

**CC BY-NC 4.0** (non-commercial, attribution required).

Please cite:

> Wang, Ding-Yang. *ML4G Project 1 – Predicting Gene Expression from Epigenomic Signals*. GitHub repository, 2025.

```bibtex
@misc{wang2025ml4g,
  author       = {Wang, Ding-Yang},
  title        = {ML4G Project 1 – Predicting Gene Expression from Epigenomic Signals},
  year         = {2025},
  howpublished = {\url{https://github.com/<your-repo>}},
  note         = {Non-commercial use; citation required}
}
```
