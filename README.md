
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
