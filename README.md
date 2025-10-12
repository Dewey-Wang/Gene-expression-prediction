
# 🧬 ML4G Project 1 – Predicting Gene Expression from Epigenomic Signals

## 📘 Overview
This project aims to **predict gene expression levels** for different cell lines using epigenomic features derived from histone modification and chromatin accessibility datasets.  
The datasets include:

- **CAGE (Cap Analysis of Gene Expression)** — Target gene expression values  
- **ChIP-seq marks:**  
  - H3K27me3  
  - H3K4me1  
  - H3K4me3  
  - H3K27ac  
  - H3K36me3  
  - H3K9me3  
- **DNase-seq:** Chromatin accessibility signals  

Each dataset is provided in both **BED** and **BigWig** formats.

---

## 🧱 Data Preprocessing Workflow

### 1️⃣ Unzipping & Organization
All raw `.zip` archives were extracted and organized under:
```
ML4G_Project_1_Data/
```

### 2️⃣ CAGE Merging (Train)
For both training cell lines **X1** and **X2**, the train and validation files were merged. Also, the gene expression was merged into info file:

```
X1_train_info.tsv + X1_val_info.tsv + X1_train_y.tsv + X1_val_y.tsv → X1_merged.tsv
X2_train_info.tsv + X2_val_info.tsv + X2_train_y.tsv + X2_val_y.tsv → X2_merged.tsv
```

Merged outputs are stored under:

```
preprocessed_data/CAGE-merged/
```


### 3️⃣ Chromosome-wise Splitting
- **X1** and **X2** (training): only chromosomes **chr2–chr22** retained  
- **X3** (test): only **chr1**
Each subfolder contains:
```
{cell}genes{chr}.tsv
{mark}{cell}{chr}.bed
```

Final structure:

```
preprocessed_data/
└── chromosomes/
        ├───test
        │   └───chr1
        │       └───X3
        ├───train
        │   └───chr1
        │       └───X3
        └───train
            ├── chr2/
            │ ├── X1/
            │ └── X2/
            ├── chr3/
            │ ├── X1/
            │ └── X2/
            ├── ...
```


---

# Planned Next Step
BigWig signals will be parsed using `pyBigWig` to extract average or maximum signal values around each gene’s TSS (±2 kb) as tabular numerical features.


## 🧩 Feature Representation
Each gene will be represented as a **tabular feature vector** combining:
- Mean or max values of each ChIP-seq mark  
- Chromatin accessibility (DNase-seq signal)  
- Optional genomic metadata (strand, GC content, etc.)

---

## 🧠 Model Design
Currently focusing on **tabular regression models**, suitable for structured genomic features.

Candidate models:
- **LightGBM**
- **XGBoost**  

---

## 🧪 Validation Strategy

Two complementary validation methods are planned:

### 🔹 Normal K-Fold Cross-Validation
- Randomly split genes into *k* folds  
- Each fold serves as validation once  
- Evaluates general model robustness and stability  

### 🔹 Leave-Chromosome-Out K-Fold
- Each chromosome (chr2–chr22) is left out in turn  
- Tests the ability to generalize to **unseen genomic regions**  
- Prevents position-based information leakage  

---

## ⚙️ Environment
```
Python 3.8.20
```

