import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, f1_score
from typing import List
from pathlib import Path

# ============================================================
# CORE FUNCTION
# ============================================================
def run_cross_direction_shap(
    train_path, val_path, out_dir, tag,
    TARGET_BINARY, TARGET_RANK, META_COLS, chromosomes, folds,
    params_bin, params_reg, feature_cols=None,
    MASK_THRESHOLD=0.4
):
    """Run X(train)->Y(val) direction SHAP analysis (with masked evaluation)."""
    os.makedirs(f"{out_dir}/{tag}/binary", exist_ok=True)
    os.makedirs(f"{out_dir}/{tag}/regression", exist_ok=True)

    # === Load Data ===
    df_train = pd.read_csv(train_path, sep="\t")
    df_val   = pd.read_csv(val_path, sep="\t")

    df_train[TARGET_BINARY] = (df_train["gex"] > 0).astype(int)
    df_val[TARGET_BINARY]   = (df_val["gex"] > 0).astype(int)

    if feature_cols is None:
        feature_cols = [c for c in df_train.columns if c not in META_COLS + [TARGET_BINARY]]
    results = []
    shap_binary_folds, shap_reg_folds = [], []

    print(f"\n🚀 Running SHAP cross-direction: {tag}")
    print(f"📊 Using {len(feature_cols)} features")

    for fold_idx, val_chrs in enumerate(folds):
        print(f"\n🧩 Fold {fold_idx+1} | Validation chromosomes: {val_chrs}")

        train_chrs = [c for c in chromosomes if c not in val_chrs]
        df_train_fold = df_train[df_train["chr"].isin(train_chrs)]
        df_val_fold   = df_val[df_val["chr"].isin(val_chrs)]

        X_train, y_bin_train, y_reg_train = (
            df_train_fold[feature_cols],
            df_train_fold[TARGET_BINARY],
            df_train_fold[TARGET_RANK],
        )
        X_val, y_bin_val, y_reg_val = (
            df_val_fold[feature_cols],
            df_val_fold[TARGET_BINARY],
            df_val_fold[TARGET_RANK],
        )

        # ============================================================
        # 🧠 Binary Classifier
        # ============================================================
        dtrain_bin = lgb.Dataset(X_train, label=y_bin_train)
        dval_bin   = lgb.Dataset(X_val, label=y_bin_val)
        model_bin = lgb.train(
            params_bin, dtrain_bin, valid_sets=[dtrain_bin, dval_bin],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )

        pred_prob = model_bin.predict(X_val, num_iteration=model_bin.best_iteration)
        auc = roc_auc_score(y_bin_val, pred_prob)
        f1  = f1_score(y_bin_val, (pred_prob >= 0.5))
        print(f"📈 Binary Classifier: AUC={auc:.4f}, F1={f1:.4f}")

        # SHAP for binary
        shap_bin = np.abs(shap.TreeExplainer(model_bin).shap_values(X_val)).mean(axis=0)
        shap_binary_folds.append(pd.DataFrame({"feature": feature_cols, "mean_abs_shap": shap_bin}))

        # ============================================================
        # 📈 Regression Model
        # ============================================================
        dtrain_reg = lgb.Dataset(X_train, label=y_reg_train)
        dval_reg   = lgb.Dataset(X_val, label=y_reg_val)
        model_reg = lgb.train(
            params_reg, dtrain_reg, valid_sets=[dtrain_reg, dval_reg],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )

        pred_reg = model_reg.predict(X_val, num_iteration=model_reg.best_iteration)
        rho_raw = spearmanr(y_reg_val, pred_reg)[0]

        # ============================================================
        # 🎯 Apply binary mask (> threshold)
        # ============================================================
        mask = (pred_prob >= MASK_THRESHOLD).astype(float)
        pred_reg_masked = pred_reg * mask
        rho_masked = spearmanr(y_reg_val, pred_reg_masked)[0]

        print(f"📊 Spearman ρ (raw) = {rho_raw:.4f} | (masked @>{MASK_THRESHOLD}) = {rho_masked:.4f}")

        # Save results
        results.append({
            "fold": fold_idx + 1,
            "auc": auc,
            "f1": f1,
            "rho_reg": rho_raw,
            "rho_masked": rho_masked
        })

        # === SHAP for binary ===
        shap_bin = np.abs(shap.TreeExplainer(model_bin).shap_values(X_val)).mean(axis=0)
        shap_df_bin = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": shap_bin})
        shap_binary_folds.append(shap_df_bin)

        # === SHAP for regression ===
        shap_reg = np.abs(shap.TreeExplainer(model_reg).shap_values(X_val)).mean(axis=0)
        shap_df_reg = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": shap_reg})
        shap_reg_folds.append(shap_df_reg)

        # === Create fold directories ===
        fold_dir_bin = f"{out_dir}/{tag}/binary/fold_{fold_idx+1}"
        fold_dir_reg = f"{out_dir}/{tag}/regression/fold_{fold_idx+1}"
        os.makedirs(fold_dir_bin, exist_ok=True)
        os.makedirs(fold_dir_reg, exist_ok=True)

        # === Save per-fold SHAP results ===
        shap_df_bin.to_csv(f"{fold_dir_bin}/binary_shap_fold.tsv", sep="\t", index=False)
        shap_df_reg.to_csv(f"{fold_dir_reg}/regression_shap_fold.tsv", sep="\t", index=False)


    # ============================================================
    # 🔹 Summarize SHAP
    # ============================================================
    def summarize(shap_list):
        df_all = pd.concat(shap_list)
        return (
            df_all.groupby("feature", as_index=False)
            .agg(mean_shap=("mean_abs_shap", "mean"))
            .sort_values("mean_shap", ascending=False)
        )

    summary_bin = summarize(shap_binary_folds)
    summary_reg = summarize(shap_reg_folds)

    summary_bin.to_csv(f"{out_dir}/{tag}/binary_shap_summary.tsv", sep="\t", index=False)
    summary_reg.to_csv(f"{out_dir}/{tag}/regression_shap_summary.tsv", sep="\t", index=False)

    # ============================================================
    # 📊 Final summary printout
    # ============================================================
    results_df = pd.DataFrame(results)
    print("\n===== 📊 OVERALL PERFORMANCE =====")
    print(results_df)
    print(f"\nMean Spearman ρ (raw):     {results_df['rho_reg'].mean():.4f}")
    print(f"Mean Spearman ρ (masked):  {results_df['rho_masked'].mean():.4f}")
    print(f"Mean AUC (binary):         {results_df['auc'].mean():.4f}")
    print(f"Mean F1 (binary):          {results_df['f1'].mean():.4f}")

    return summary_bin, summary_reg, results_df

# === merge SHAP importance in both directions ===
def merge_shap_stability(df1, df2, model_type, OUTPUT_DIR):
    merged = df1.merge(df2, on="feature", suffixes=("_fwd", "_rev"))
    merged["mean_rank"] = merged[["mean_shap_fwd", "mean_shap_rev"]].mean(axis=1)
    merged["stability"] = 1 - np.abs(merged["mean_shap_fwd"] - merged["mean_shap_rev"]) / merged["mean_rank"]
    merged = merged.sort_values("mean_rank", ascending=False)
    merged.to_csv(f"{OUTPUT_DIR}/{model_type}_shap_stability.tsv", sep="\t", index=False)
    return merged


def load_top_shap_features(
    binary_path: str,
    regression_path: str,
    top_n: int = 1000
) -> List[str]:
    """(Existing function; use as-is)"""
    df_bin = pd.read_csv(binary_path, sep="\t")
    df_reg = pd.read_csv(regression_path, sep="\t")

    df_bin_sorted = df_bin.sort_values("mean_shap", ascending=False)
    df_reg_sorted = df_reg.sort_values("mean_shap", ascending=False)

    top_bin = df_bin_sorted.head(top_n)["feature"].tolist()
    top_reg = df_reg_sorted.head(top_n)["feature"].tolist()

    selected_features = list(dict.fromkeys(top_bin + top_reg))
    print(
        f"🧩 Extracted {len(selected_features)} unique features "
        f"from top-{top_n} of binary & regression SHAP summaries."
    )
    return selected_features


def select_bidirectional_top_features(
    base_dir: str,
    directions: List[str] = ["X1_to_X2", "X2_to_X1"],
    top_n: int = 1000,
    output_path: str | None = None,
) -> List[str]:
    """
    Combine top-N SHAP features from both directions (binary + regression).

    Parameters
    ----------
    base_dir : str
        Directory containing subfolders like X1_to_X2/, X2_to_X1/.
    directions : list[str], optional
        Direction tags to include (default: ["X1_to_X2", "X2_to_X1"]).
    top_n : int, optional
        Number of top features to select from each SHAP summary (default: 1000).
    output_path : str, optional
        If provided, saves the union feature list as a TSV to this path.

    Returns
    -------
    list[str]
        Union of top-N features across directions and model types.
    """
    all_features: list[str] = []

    for tag in directions:
        bin_path = os.path.join(base_dir, tag, "binary_shap_summary.tsv")
        reg_path = os.path.join(base_dir, tag, "regression_shap_summary.tsv")

        if not os.path.exists(bin_path) or not os.path.exists(reg_path):
            print(f"⚠️ Missing SHAP files for direction: {tag}")
            continue

        print(f"\n🔄 Processing direction: {tag}")
        feats = load_top_shap_features(bin_path, reg_path, top_n=top_n)
        all_features.extend(feats)

    # --- De-duplicate while preserving order ---
    unique_features = list(dict.fromkeys(all_features))
    print(
        f"\n✅ Total {len(unique_features)} unique features selected "
        f"from {len(directions)} directions (top {top_n} each)."
    )

    # --- Save to disk (optional) ---
    if output_path:
        Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
        pd.Series(unique_features, name="feature").to_csv(
            output_path, index=False, sep="\t"
        )
        print(f"💾 Saved combined bidirectional feature list to: {output_path}")

    return unique_features

def get_high_corr_pairs(df, feature_cols, threshold=0.99, method="pearson"):
    X = df[feature_cols]
    corr = X.corr(method=method)
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "feature_1", "level_1": "feature_2", 0: "correlation"})
    )
    pairs = pairs[pairs["correlation"].abs() > threshold]
    return pairs


def remove_sparse_features(dfs, feature_cols, threshold=0.01):
    """
    移除非零比例過低（太稀疏）的特徵。
    threshold=0.01 表示若 <1% 非零值就刪除。
    """
    print(f"\n🧹 Removing sparse features (non-zero ratio < {threshold:.2%}) ...")
    zero_stats = []

    for f in feature_cols:
        # 計算在所有 cell 合併後的非零比例
        non_zero_ratio = np.mean([
            np.count_nonzero(dfs[cell][f].values) / len(dfs[cell])
            for cell in dfs
            if f in dfs[cell].columns
        ])
        zero_stats.append((f, non_zero_ratio))

    zero_df = pd.DataFrame(zero_stats, columns=["feature", "nonzero_ratio"])
    sparse_feats = zero_df.loc[zero_df["nonzero_ratio"] < threshold, "feature"].tolist()

    print(f"🚫 Found {len(sparse_feats)} sparse features to remove (avg non-zero ratio < {threshold})")

    # 回傳保留後的 feature 列表
    kept_feats = [f for f in feature_cols if f not in sparse_feats]
    return kept_feats, zero_df
