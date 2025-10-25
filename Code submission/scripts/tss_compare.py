import os
import random
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

def run_cv(train_name, val_name, data_dir, META_COLS, TARGET_COL, chromosomes, params,
           mode="loco", n_splits=5):
    """
    Perform CV (Leave-One-Chromosome-Out or K-Fold) between train_name → val_name.

    mode:
        - "loco": Leave-One-Chromosome-Out (default)
        - "chrom_kfold": Standard random K-Fold on training set

    Returns:
        mean_cv (float): average Spearman correlation across folds
        rho_full (float): Spearman correlation of full model baseline
    """
    print(f"\n==============================")
    print(f" 🔁 CV ({mode.upper()}) | {train_name} → {val_name}")
    print(f"==============================")

    # === Load datasets ===
    df_train_full = pd.read_csv(f"{data_dir}/{train_name}_all_logzscore_logzscore_with_y.tsv", sep="\t")
    df_val_full   = pd.read_csv(f"{data_dir}/{val_name}_all_logzscore_logzscore_with_y.tsv", sep="\t")
    feature_cols = [c for c in df_train_full.columns if c not in META_COLS]

    results = []

    # ============================================================
    # 🧬 MODE 1: LOCO (Leave-One-Chromosome-Out)
    # ============================================================
    if mode == "loco":
        for chrom_val in chromosomes:
            df_train = df_train_full[df_train_full["chr"] != chrom_val].copy()
            df_val   = df_val_full[df_val_full["chr"] == chrom_val].copy()

            if df_val.empty:
                print(f"⚠️ Skip {chrom_val} — no validation samples")
                continue

            X_train, y_train = df_train[feature_cols], df_train[TARGET_COL]
            X_val, y_val     = df_val[feature_cols], df_val[TARGET_COL]

            dtrain = lgb.Dataset(X_train, label=y_train)
            dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

            model = lgb.train(
                params,
                dtrain,
                valid_sets=[dtrain, dval],
                valid_names=["train", "val"],
                num_boost_round=2000,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100),
                    lgb.log_evaluation(period=0)
                ],
            )

            df_val["pred"] = model.predict(X_val, num_iteration=model.best_iteration)
            rho = spearmanr(df_val[TARGET_COL], df_val["pred"])[0]
            results.append({"fold": chrom_val, "spearman": rho})

    # ============================================================
    # 🧪 MODE 2: K-FOLD (Standard random splits on training set)
    # ============================================================
    elif mode == "chrom_kfold":
        chrom_folds = [chromosomes[i::n_splits] for i in range(n_splits)]
        for i, chrom_group in enumerate(chrom_folds):
            print(f"\n🚀 Chromosome-KFold {i+1}/{n_splits}: Valid={chrom_group}")
            df_train = df_train_full[~df_train_full["chr"].isin(chrom_group)].copy()
            df_val   = df_val_full[df_val_full["chr"].isin(chrom_group)].copy()
            if df_val.empty:
                print(f"⚠️ Skip fold {i+1} — no val chromosomes")
                continue

            X_train, y_train = df_train[feature_cols], df_train[TARGET_COL]
            X_val, y_val     = df_val[feature_cols], df_val[TARGET_COL]
            
            dtrain = lgb.Dataset(X_train, label=y_train)
            dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

            model = lgb.train(
                params,
                dtrain,
                valid_sets=[dtrain, dval],
                valid_names=["train", "val"],
                num_boost_round=2000,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100),
                    lgb.log_evaluation(period=0)
                ],
            )

            preds = model.predict(X_val, num_iteration=model.best_iteration)
            rho = spearmanr(y_val, preds)[0]
            results.append({"fold": f"Fold{i+1}", "spearman": rho})

    else:
        raise ValueError(f"❌ Unknown mode: {mode}. Use 'loco' or 'kfold'.")

    # ============================================================
    # 📈 Summary + Full Model Baseline
    # ============================================================
    summary_df = pd.DataFrame(results)
    mean_cv = summary_df["spearman"].mean()

    print(f"\n📊 {mode.upper()} mean Spearman = {mean_cv:.4f}")
    return mean_cv


def run_all(data_dir, META_COLS, TARGET_COL, chromosomes, params , mode="loco", n_splits=5):
    """
    Perform CV (LOCO / Chromosome-KFold / KFold)
    on merged X1 + X2 dataset, while tracking each subset’s performance.

    Returns:
        X1_mean, X2_mean, X1_rho, X2_rho
    """
    print(f"\n==============================")
    print(f" 🔁 {mode.upper()} CV | Merged X1 + X2 | {data_dir}")
    print(f"==============================")

    # === Load datasets ===
    df_X1 = pd.read_csv(f"{data_dir}/X1_all_logzscore_logzscore_with_y.tsv", sep="\t")
    df_X2 = pd.read_csv(f"{data_dir}/X2_all_logzscore_logzscore_with_y.tsv", sep="\t")
    df_X1["cell"] = "X1"
    df_X2["cell"] = "X2"

    # --- Merge ---
    df_all = pd.concat([df_X1, df_X2], axis=0, ignore_index=True)
    feature_cols = [c for c in df_all.columns if c not in META_COLS + ["cell"]]
    results = []

    # ============================================================
    # 🧬 LOCO
    # ============================================================
    if mode == "loco":
        for chrom_val in chromosomes:
            df_train = df_all[df_all["chr"] != chrom_val].copy()
            df_val   = df_all[df_all["chr"] == chrom_val].copy()

            if df_val.empty:
                print(f"⚠️ Skip {chrom_val} — no validation samples")
                continue

            X_train, y_train = df_train[feature_cols], df_train[TARGET_COL]
            X_val, y_val     = df_val[feature_cols], df_val[TARGET_COL]

            if len(X_train) == 0 or len(X_val) == 0:
                print(f"⚠️ Skip {chrom_val} — missing training or validation data")
                continue

            dtrain = lgb.Dataset(X_train, label=y_train)
            dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

            model = lgb.train(
                params,
                dtrain,
                valid_sets=[dtrain, dval],
                valid_names=["train", "val"],
                num_boost_round=2000,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100),
                    lgb.log_evaluation(period=0)
                ],
            )

            preds = model.predict(X_val, num_iteration=model.best_iteration)
            df_val["pred"] = preds
            results.append(df_val[["gene_name", "chr", "cell", TARGET_COL, "pred"]])

    elif mode == "chrom_kfold":
        chrom_folds = [chromosomes[i::n_splits] for i in range(n_splits)]
        for i, chrom_group in enumerate(chrom_folds):
            print(f"\n🚀 Chromosome-KFold {i+1}/{n_splits}: Valid={chrom_group}")
            df_train = df_all[~df_all["chr"].isin(chrom_group)].copy()
            df_val   = df_all[df_all["chr"].isin(chrom_group)].copy()
            if df_val.empty:
                print(f"⚠️ Skip fold {i+1} — no val chromosomes")
                continue

            X_train, y_train = df_train[feature_cols], df_train[TARGET_COL]
            X_val, y_val     = df_val[feature_cols], df_val[TARGET_COL]

            if len(X_train) == 0 or len(X_val) == 0:
                print(f"⚠️ Skip fold {i+1} — missing data")
                continue

            dtrain = lgb.Dataset(X_train, label=y_train)
            dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

            model = lgb.train(
                params,
                dtrain,
                valid_sets=[dtrain, dval],
                valid_names=["train", "val"],
                num_boost_round=2000,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100),
                    lgb.log_evaluation(period=0)
                ],
            )

            df_val["pred"] = model.predict(X_val, num_iteration=model.best_iteration)
            results.append(df_val[["gene_name", "chr", "cell", TARGET_COL, "pred"]])
        print(results)
    else:
        raise ValueError(f"❌ Unknown mode: {mode}")

        # ============================================================
    # 📊 Combine & Evaluate
    # ============================================================
    if not results:
        raise ValueError("❌ No valid folds were processed — check your chromosomes or data!")

    df_pred = pd.concat(results, axis=0, ignore_index=True)
    print(f"\n📘 Combined CV predictions: {df_pred.shape}")

    # ============================================================
    # 📈 Spearman evaluation
    # ============================================================
    def _compute_spearman(subset, label):
        if subset.empty:
            print(f"⚠️ {label} subset is empty")
            return np.nan
        rho = spearmanr(subset[TARGET_COL], subset["pred"])[0]
        print(f"📈 {label} Spearman ρ = {rho:.4f}")
        return rho

    # -- Mean Spearman by cell
    X1_mean = _compute_spearman(df_pred[df_pred["cell"] == "X1"], "X1 Mean (CV)")
    X2_mean = _compute_spearman(df_pred[df_pred["cell"] == "X2"], "X2 Mean (CV)")

    # -- Global (all merged) mean
    global_mean = _compute_spearman(df_pred, "Global Mean (CV)")

    # ============================================================
    # ✅ Summary
    # ============================================================
    print("\n==============================")
    print("📊 Final Summary Across Subsets")
    print("==============================")
    print(f"X1 : Mean(CV)={X1_mean:.4f}")
    print(f"X2 : Mean(CV)={X2_mean:.4f}")
    print(f"Global Mean(CV)={global_mean:.4f}")
    print(f"Δ_mean = {X1_mean - X2_mean:.4f}")
    print("==============================")

    return X1_mean, X2_mean, global_mean
