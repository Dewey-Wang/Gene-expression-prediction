import os
import random
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# FUNCTION: Cross-chromosome training pipeline
# ============================================================
def run_cross_chr_training(
    train_path,
    val_path,
    meta_cols,
    features_path=None,
    target_rank="gex_rank",
    target_binary="gex_binary",
    seed=42,
    n_folds=5,
    thresholds=np.arange(0.1, 1.0, 0.1),
    params_bin=None,
    params_reg=None,
):
    """
    🧬 Cross-chromosome training pipeline for LightGBM binary + regression models.

    Parameters
    ----------
    train_path : str
        Path to training data (with Y)
    val_path : str
        Path to validation data (with Y)
    meta_cols : list
        Metadata columns (excluded from features)
    features_path : str or None
        Optional TSV file containing preselected features (column 'feature')
    target_rank : str
        Regression target column name (default: 'gex_rank')
    target_binary : str
        Binary classification target column name (default: 'gex_binary')
    seed : int
        Random seed
    n_folds : int
        Number of chromosome folds (default: 5)
    thresholds : np.ndarray
        Thresholds to sweep (default: np.arange(0.1, 1.0, 0.1))
    params_bin : dict or None
        Custom LightGBM parameters for binary classifier
    params_reg : dict or None
        Custom LightGBM parameters for regression model

    Returns
    -------
    summary : pd.DataFrame
        Fold-wise model metrics (AUC, ACC, F1, rho_reg, rho_weighted)
    chr_mean : pd.DataFrame
        Mean per-chromosome Spearman correlation vs threshold
    """

    # ============================================================
    # SEED SETUP
    # ============================================================
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"🔒 Global seed set to {seed}")

    # ============================================================
    # LOAD DATA
    # ============================================================
    df_train_full = pd.read_csv(train_path, sep="\t")
    df_val_full = pd.read_csv(val_path, sep="\t")

    df_train_full[target_binary] = (df_train_full["gex"] > 0.0).astype(int)
    df_val_full[target_binary] = (df_val_full["gex"] > 0.0).astype(int)

    # === Load features ===
    if features_path is not None and os.path.exists(features_path):
        feature_cols = pd.read_csv(features_path, sep="\t")["feature"].tolist()
        print(f"✅ Loaded {len(feature_cols)} selected features from {features_path}")
    else:
        feature_cols = [c for c in df_train_full.columns if c not in meta_cols + [target_binary]]
        print(f"⚙️ Using all {len(feature_cols)} features (no FEATURES_PATH provided)")

    # ============================================================
    # CHROMOSOME SPLITS
    # ============================================================
    chromosomes = [f"chr{i}" for i in range(2, 23)]
    folds = [chromosomes[i::n_folds] for i in range(n_folds)]
    print("🧩 Chromosome folds:")
    for i, fset in enumerate(folds):
        print(f"  Fold {i+1}: {fset}")

    # ============================================================
    # MODEL PARAMS (with default)
    # ============================================================
    default_params_bin = {
        "objective": "binary",
        "metric": ["auc"],
        "learning_rate": 0.016676974956976915,
        "num_leaves": 48,
        "max_depth": 8,
        "feature_fraction": 0.64561553423692,
        "bagging_fraction": 0.8113835038425429,
        "bagging_freq": 6,
        "lambda_l1": 0.3316673054635859,
        "lambda_l2": 0.8969317795206216,
        "min_gain_to_split": 0.04923442843722911,
        "min_data_in_leaf": 38,
        "verbose": -1,
        "seed": seed,
    }

    default_params_reg = {
        **default_params_bin,
        "objective": "regression",
        "metric": "rmse",
    }

    # === allow override ===
    params_bin = params_bin or default_params_bin
    params_reg = params_reg or default_params_reg

    print(f"🧩 Binary params: {len(params_bin)} keys")
    print(f"🧩 Regression params: {len(params_reg)} keys")

    # ============================================================
    # RESULTS STORAGE
    # ============================================================
    results = []
    threshold_records = []
    chr_rho_records = []

    # ============================================================
    # CROSS-CHROMOSOME TRAINING LOOP
    # ============================================================
    for fold_idx, val_chrs in enumerate(folds):
        print(f"\n🚀 Fold {fold_idx+1} | Validation chromosomes: {val_chrs}")

        train_chrs = [c for c in chromosomes if c not in val_chrs]
        df_train = df_train_full[df_train_full["chr"].isin(train_chrs)].copy()
        df_val = df_val_full[df_val_full["chr"].isin(val_chrs)].copy()

        X_train, y_train_bin, y_train_reg = (
            df_train[feature_cols],
            df_train[target_binary],
            df_train[target_rank],
        )
        X_val, y_val_bin, y_val_reg = (
            df_val[feature_cols],
            df_val[target_binary],
            df_val[target_rank],
        )

        # === Binary Classifier ===
        print("🧠 Training Binary Classifier...")
        dtrain_bin = lgb.Dataset(X_train, label=y_train_bin)
        dval_bin = lgb.Dataset(X_val, label=y_val_bin, reference=dtrain_bin)

        model_bin = lgb.train(
            params_bin,
            dtrain_bin,
            valid_sets=[dtrain_bin, dval_bin],
            valid_names=["train", "val"],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(stopping_rounds=100)],
        )

        df_val["pred_prob"] = model_bin.predict(X_val, num_iteration=model_bin.best_iteration)
        auc = roc_auc_score(y_val_bin, df_val["pred_prob"])
        acc = accuracy_score(y_val_bin, (df_val["pred_prob"] >= 0.5))
        f1 = f1_score(y_val_bin, (df_val["pred_prob"] >= 0.5))
        print(f"📈 Binary Classifier: AUC={auc:.4f}, ACC={acc:.4f}, F1={f1:.4f}")

        # === Regression Model ===
        print("🧩 Training Regression Model...")
        dtrain_reg = lgb.Dataset(X_train, label=y_train_reg)
        dval_reg = lgb.Dataset(X_val, label=y_val_reg, reference=dtrain_reg)

        model_reg = lgb.train(
            params_reg,
            dtrain_reg,
            valid_sets=[dtrain_reg, dval_reg],
            valid_names=["train", "val"],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(stopping_rounds=100)],
        )

        df_val["pred_reg"] = model_reg.predict(X_val, num_iteration=model_reg.best_iteration)

        # === Threshold Sweep ===
        for thr in thresholds:
            df_val["pred_masked_thr"] = df_val["pred_reg"] * (df_val["pred_prob"] >= thr).astype(int)
            rho_thr = spearmanr(df_val[target_rank], df_val["pred_masked_thr"])[0]
            threshold_records.append({"fold": fold_idx + 1, "threshold": thr, "rho": rho_thr})

            for chrom, subdf in df_val.groupby("chr"):
                if len(subdf) < 2:
                    continue
                rho_chr = spearmanr(subdf[target_rank], subdf["pred_masked_thr"])[0]
                chr_rho_records.append({
                    "fold": fold_idx + 1,
                    "chr": chrom,
                    "threshold": thr,
                    "rho_chr": rho_chr
                })

        # === Baseline Comparisons ===
        rho_reg = spearmanr(df_val[target_rank], df_val["pred_reg"])[0]
        rho_weighted = spearmanr(df_val[target_rank], df_val["pred_reg"] * df_val["pred_prob"])[0]

        results.append({
            "fold": fold_idx + 1,
            "auc": auc,
            "acc": acc,
            "f1": f1,
            "rho_reg": rho_reg,
            "rho_weighted": rho_weighted,
        })

    # ============================================================
    # SUMMARY
    # ============================================================
    summary = pd.DataFrame(results)
    thr_df = pd.DataFrame(threshold_records)
    chr_df = pd.DataFrame(chr_rho_records)
    chr_mean = chr_df.groupby("threshold", as_index=False)["rho_chr"].mean()

    print("\n===== Cross-Chromosome Summary =====")
    print(summary[["fold", "auc", "acc", "f1", "rho_reg", "rho_weighted"]])

    print(f"\nMean AUC = {summary['auc'].mean():.4f}")
    print(f"Mean ACC = {summary['acc'].mean():.4f}")
    print(f"Mean F1  = {summary['f1'].mean():.4f}")
    print(f"Mean ρ (reg)      = {summary['rho_reg'].mean():.4f}")
    print(f"Mean ρ (weighted) = {summary['rho_weighted'].mean():.4f}")

    # === Visualization ===
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=chr_mean, x="threshold", y="rho_chr", marker="o", color="black", linewidth=2)
    plt.title("Mean Per-Chromosome Spearman vs Threshold", fontsize=14)
    plt.xlabel("Threshold")
    plt.ylabel("Mean Spearman (ρ)")
    plt.tight_layout()
    plt.show()

    best_thr = chr_mean.loc[chr_mean["rho_chr"].idxmax()]
    print(f"\n🌟 Best threshold = {best_thr['threshold']:.2f}, mean per-chr ρ = {best_thr['rho_chr']:.4f}")

    return summary, chr_mean



# ============================================================
# Example Call
# ============================================================
if __name__ == "__main__":
    META_COLS = ["gene_name", "chr", "gene_start", "gene_end", "TSS_start", "TSS_end", "strand", "gex", "gex_rank"]

    run_cross_chr_training(
        train_path="../preprocessed_data/reference/1. merged data/with_y_100_one_side/X1_all_rank_features_with_y.tsv",
        val_path="../preprocessed_data/reference/1. merged data/with_y_100_one_side/X2_all_rank_features_with_y.tsv",
        meta_cols=META_COLS,
        features_path=None,  # ← 自動使用所有非 meta 欄位
    )
