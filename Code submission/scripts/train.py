import os
import random
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr
from datetime import datetime
import json


# ============================================================
# FUNCTION: Leave-one-chr + Inner KFold + Unlabeled Test
# ============================================================
def run_leaveonechr_with_innerkfold(
    train_path,
    test_path,
    unlabeled_test_path=None,   # 🆕 無標籤 test
    meta_cols=None,
    features_path=None,
    output_dir="./results/",
    target_rank="gex_rank",
    target_binary="gex_binary",
    seed=42,
    n_inner_folds=5,
    THRESHOLD=0.4,
    params_bin=None,
    params_reg=None,
):
    """
    🧬 Nested CV:
      - Outer: leave-one-chromosome
      - Inner: chr-KFold (on remaining chromosomes)
      - test_path: labeled test (有 Y)
      - unlabeled_test_path: 真正預測用的 test（無 label）
      - output_dir: logs + configs + results
    """

    # ============================================================
    # INIT
    # ============================================================
    os.makedirs(output_dir, exist_ok=True)
    LOG_PATH = os.path.join(output_dir, "log.txt")
    CONFIG_PATH = os.path.join(output_dir, "config.json")

    def log(msg):
        """同時列印與寫入 log 檔"""
        print(msg)
        with open(LOG_PATH, "a") as f:
            f.write(f"{msg}\n")

    # 清空舊 log
    with open(LOG_PATH, "w") as f:
        f.write(f"==== New Experiment {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ====\n")

    # ============================================================
    # SEED SETUP
    # ============================================================
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    log(f"🔒 Global seed set to {seed}")

    # ============================================================
    # LOAD DATA
    # ============================================================
    df_train_full = pd.read_csv(train_path, sep="\t")
    df_test_full = pd.read_csv(test_path, sep="\t")
    df_unlabeled_test = None

    if unlabeled_test_path and os.path.exists(unlabeled_test_path):
        df_unlabeled_test = pd.read_csv(unlabeled_test_path, sep="\t")
        log(f"🧪 Loaded unlabeled test set: {unlabeled_test_path} (shape={df_unlabeled_test.shape})")

    # Add binary labels
    df_train_full[target_binary] = (df_train_full["gex"] > 0).astype(int)
    df_test_full[target_binary] = (df_test_full["gex"] > 0).astype(int)

    # === Load features ===
    if features_path and os.path.exists(features_path):
        feature_cols = pd.read_csv(features_path, sep="\t")["feature"].tolist()
        log(f"✅ Loaded {len(feature_cols)} selected features from {features_path}")
    else:
        feature_cols = [c for c in df_train_full.columns if c not in meta_cols + [target_binary]]
        log(f"⚙️ Using all {len(feature_cols)} features")

    # ============================================================
    # MODEL PARAMS
    # ============================================================
    default_params_bin = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.0167,
        "num_leaves": 48,
        "max_depth": 8,
        "feature_fraction": 0.65,
        "bagging_fraction": 0.81,
        "bagging_freq": 6,
        "lambda_l1": 0.33,
        "lambda_l2": 0.9,
        "verbose": -1,
        "seed": seed,
    }
    default_params_reg = {**default_params_bin, "objective": "regression", "metric": "rmse"}
    params_bin = params_bin or default_params_bin
    params_reg = params_reg or default_params_reg

    # ============================================================
    # SAVE CONFIG
    # ============================================================
    config = {
        "train_path": train_path,
        "test_path": test_path,
        "unlabeled_test_path": unlabeled_test_path,
        "features_path": features_path,
        "target_col": target_rank,
        "seed": seed,
        "n_inner_folds": n_inner_folds,
        "threshold": THRESHOLD,
        "params_bin": params_bin,
        "params_reg": params_reg,
        "features": feature_cols,
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)
    log(f"💾 Config saved → {CONFIG_PATH}")

    # ============================================================
    # STORAGE
    # ============================================================
    outer_results = []
    inner_results = []
    test_pred_prob_chr = []
    test_pred_reg_chr = []
    unlabeled_pred_prob_chr = []  # 🆕
    unlabeled_pred_reg_chr = []   # 🆕

    chromosomes = [f"chr{i}" for i in range(2, 23)]
    log(f"Chromosomes: {chromosomes}")

    # ============================================================
    # OUTER LOOP (Leave-one-chromosome)
    # ============================================================
    for outer_chr in chromosomes:
        log(f"\n🚀 Outer Fold = Leave out {outer_chr}")

        df_outer_val = df_train_full[df_train_full["chr"] == outer_chr].copy()
        df_outer_train = df_train_full[df_train_full["chr"] != outer_chr].copy()

        inner_chrs = [c for c in chromosomes if c != outer_chr]
        inner_folds = [inner_chrs[i::n_inner_folds] for i in range(n_inner_folds)]

        preds_prob_outer_folds, preds_reg_outer_folds = [], []
        preds_prob_test_folds, preds_reg_test_folds = [], []
        preds_prob_unlabeled_folds, preds_reg_unlabeled_folds = [], []

        # ========================================================
        # INNER LOOP
        # ========================================================
        for inner_idx, val_chrs in enumerate(inner_folds, start=1):
            log(f"  ⚙️ Inner Fold {inner_idx} | val_chrs={val_chrs}")
            df_inner_train = df_outer_train[~df_outer_train["chr"].isin(val_chrs)].copy()
            df_inner_val = df_outer_train[df_outer_train["chr"].isin(val_chrs)].copy()

            X_train, y_bin_train, y_reg_train = (
                df_inner_train[feature_cols],
                df_inner_train[target_binary],
                df_inner_train[target_rank],
            )
            X_val, y_bin_val, y_reg_val = (
                df_inner_val[feature_cols],
                df_inner_val[target_binary],
                df_inner_val[target_rank],
            )

            # === Binary classifier ===
            dtrain_bin = lgb.Dataset(X_train, label=y_bin_train)
            dval_bin = lgb.Dataset(X_val, label=y_bin_val)
            model_bin = lgb.train(
                params_bin,
                dtrain_bin,
                valid_sets=[dval_bin],
                num_boost_round=2000,
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )

            # === Regression model ===
            dtrain_reg = lgb.Dataset(X_train, label=y_reg_train)
            dval_reg = lgb.Dataset(X_val, label=y_reg_val)
            model_reg = lgb.train(
                params_reg,
                dtrain_reg,
                valid_sets=[dval_reg],
                num_boost_round=2000,
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )

            # === Validation ===
            prob_val = model_bin.predict(X_val)
            reg_val = model_reg.predict(X_val)
            rho_nomask = spearmanr(y_reg_val, reg_val)[0]
            rho_masked = spearmanr(y_reg_val, reg_val * (prob_val >= THRESHOLD).astype(int))[0]
            inner_results.append({
                "outer_chr": outer_chr,
                "inner_fold": inner_idx,
                "rho_nomask": rho_nomask,
                "rho_masked": rho_masked,
            })
            log(f"     chr-kfold Val:{val_chrs} ρ_nomask={rho_nomask:.4f}, ρ_masked={rho_masked:.4f}")

            # === Predict outer val ===
            X_outer = df_outer_val[feature_cols]
            prob_outer = model_bin.predict(X_outer)
            reg_outer = model_reg.predict(X_outer)
            preds_prob_outer_folds.append(prob_outer)
            preds_reg_outer_folds.append(reg_outer)

            rho_outer_nomask = spearmanr(df_outer_val[target_rank], reg_outer)[0]
            rho_outer_masked = spearmanr(df_outer_val[target_rank], reg_outer * (prob_outer >= THRESHOLD).astype(int))[0]
            log(f"     chr-kfold Loco {outer_chr}: ρ_nomask={rho_outer_nomask:.4f}, ρ_masked={rho_outer_masked:.4f}")

            # === Predict test ===
            X_test = df_test_full[feature_cols]
            prob_test = model_bin.predict(X_test)
            reg_test = model_reg.predict(X_test)
            preds_prob_test_folds.append(prob_test)
            preds_reg_test_folds.append(reg_test)
            rho_test_nomask = spearmanr(df_test_full[target_rank], reg_test)[0]
            rho_test_masked = spearmanr(df_test_full[target_rank], reg_test * (prob_test >= THRESHOLD).astype(int))[0]
            log(f"     chr-kfold Test: ρ_nomask={rho_test_nomask:.4f}, ρ_masked={rho_test_masked:.4f}")

            # === Predict unlabeled ===
            if df_unlabeled_test is not None:
                X_unlabeled = df_unlabeled_test[feature_cols]
                prob_unlabeled = model_bin.predict(X_unlabeled)
                reg_unlabeled = model_reg.predict(X_unlabeled)
                preds_prob_unlabeled_folds.append(prob_unlabeled)
                preds_reg_unlabeled_folds.append(reg_unlabeled)

        # ========================================================
        # Aggregate Outer Validation
        # ========================================================
        mean_prob_outer = np.mean(np.vstack(preds_prob_outer_folds), axis=0)
        mean_reg_outer = np.mean(np.vstack(preds_reg_outer_folds), axis=0)
        rho_outer_nomask = spearmanr(df_outer_val[target_rank], mean_reg_outer)[0]
        rho_outer_masked = spearmanr(
            df_outer_val[target_rank],
            mean_reg_outer * (mean_prob_outer >= THRESHOLD).astype(int)
        )[0]
        log(f"📊 Validation (all chr-kfold) {outer_chr}: ρ_nomask={rho_outer_nomask:.4f}, ρ_masked={rho_outer_masked:.4f}")

        # ========================================================
        # Aggregate Test
        # ========================================================
        mean_prob_test = np.mean(np.vstack(preds_prob_test_folds), axis=0)
        mean_reg_test = np.mean(np.vstack(preds_reg_test_folds), axis=0)
        rho_test_nomask = spearmanr(df_test_full[target_rank], mean_reg_test)[0]
        rho_test_masked = spearmanr(
            df_test_full[target_rank],
            mean_reg_test * (mean_prob_test >= THRESHOLD).astype(int)
        )[0]
        test_pred_reg_chr.append(mean_reg_test)
        test_pred_prob_chr.append(mean_prob_test)
        outer_results.append({
            "outer_chr": outer_chr,
            "rho_outer_nomask": rho_outer_nomask,
            "rho_outer_masked": rho_outer_masked,
            "rho_all_test_nomask": rho_test_nomask,
            "rho_all_test_masked": rho_test_masked
        })
        log(f"🧪 Test (all chr-kfold): ρ_nomask={rho_test_nomask:.4f}, ρ_masked={rho_test_masked:.4f}")

        # ========================================================
        # Aggregate Unlabeled
        # ========================================================
        if df_unlabeled_test is not None:
            mean_prob_unlabeled = np.mean(np.vstack(preds_prob_unlabeled_folds), axis=0)
            mean_reg_unlabeled = np.mean(np.vstack(preds_reg_unlabeled_folds), axis=0)
            unlabeled_pred_prob_chr.append(mean_prob_unlabeled)
            unlabeled_pred_reg_chr.append(mean_reg_unlabeled)

    # ============================================================
    # Final Aggregation
    # ============================================================
    df_outer = pd.DataFrame(outer_results)
    df_inner = pd.DataFrame(inner_results)

    mean_test_reg = np.mean(np.vstack(test_pred_reg_chr), axis=0)
    mean_test_prob = np.mean(np.vstack(test_pred_prob_chr), axis=0)
    labeled_test_predictions_df = pd.DataFrame({
            "gene_name": df_test_full["gene_name"],
            "mean_bn_prob": mean_test_prob,
            "gex_predicted": mean_test_reg
        })
    
    rho_test_nomask_final = spearmanr(df_test_full[target_rank], mean_test_reg)[0]
    rho_test_masked_final = spearmanr(
        df_test_full[target_rank],
        mean_test_reg * (mean_test_prob >= THRESHOLD).astype(int)
    )[0]
    log(f"\n🧪 Final Test (all chr): ρ_nomask={rho_test_nomask_final:.4f}, ρ_masked={rho_test_masked_final:.4f}")

    unlabeled_pred_reg_final, unlabeled_pred_prob_final = None, None
    if df_unlabeled_test is not None:
        unlabeled_pred_reg_final = np.mean(np.vstack(unlabeled_pred_reg_chr), axis=0)
        unlabeled_pred_prob_final = np.mean(np.vstack(unlabeled_pred_prob_chr), axis=0)
        unlabeled_pred_prob_final_df = pd.DataFrame({
            "gene_name": df_unlabeled_test["gene_name"],
            "mean_bn_prob": unlabeled_pred_prob_final,
            "gex_predicted": unlabeled_pred_reg_final
        })
        log(f"✅ Generated predictions for unlabeled test ({len(unlabeled_pred_reg_final)} samples)")

    # ============================================================
    # SUMMARY
    # ============================================================
    log("\n===== Outer Fold and Test Summary =====")
    log(df_outer.to_string(index=False))
    log(f"\nMean outer ρ_nomask={df_outer['rho_outer_nomask'].mean():.4f}")
    log(f"Mean outer ρ_masked={df_outer['rho_outer_masked'].mean():.4f}")
    log(f"Mean test ρ_nomask={df_outer['rho_all_test_nomask'].mean():.4f}")
    log(f"Mean test ρ_masked={df_outer['rho_all_test_masked'].mean():.4f}")


    return df_outer, df_inner,labeled_test_predictions_df,unlabeled_pred_prob_final_df
    


# file: train_lgbm_nested.py
import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from datetime import datetime

# ---- utilities ----

def run_lgbm_nested_training(
    train_1_path,
    train_2_path,
    test_path,
    features_path,
    output_dir,
    meta_cols,
    target_rank="gex_rank",
    target_binary="gex_binary",
    seed=42,
    n_inner_folds=5,
    mask_threshold=0.4,
):
    """
    Nested Leave-One-Chromosome (outer) + Chromosome-KFold (inner) LightGBM.
    Returns:
      results_summary_df, results_inner_df, test_predictions_df, train_test_predictions_df
    """
    # ---- setup/logging ----
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "log.txt")
    with open(log_path, "w") as f:
        f.write(f"==== New Experiment {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ====\n")

    def log(msg: str):
        print(msg)
        with open(log_path, "a") as f:
            f.write(f"{msg}\n")

    log("🚀 Starting nested LGBM training pipeline")

    # ---- load data ----
    df_train_1_full = pd.read_csv(train_1_path, sep="\t")
    df_train_2_full = pd.read_csv(train_2_path, sep="\t")
    df_test = pd.read_csv(test_path, sep="\t")

    # feature list
    if features_path and os.path.exists(features_path):
        selected_features = pd.read_csv(features_path, sep="\t")["feature"].tolist()
        feature_cols = [c for c in selected_features if c not in meta_cols]
        log(f"🔑 Loaded {len(feature_cols)} selected features from {features_path}")
    else:
        feature_cols = [c for c in df_train_1_full.columns if c not in meta_cols]
        log(f"⚙️ Using all {len(feature_cols)} features")

    # labels
    df_train_1_full[target_binary] = (df_train_1_full["gex"] > 0.0).astype(int)
    df_train_2_full[target_binary] = (df_train_2_full["gex"] > 0.0).astype(int)

    # merged training data
    keep_cols = feature_cols + [target_rank, target_binary, "chr", "gene_name"]
    df_full = (
        pd.concat([df_train_1_full, df_train_2_full], ignore_index=True)[keep_cols]
        .reset_index(drop=True)
    )
    log("🔀 Merged TRAIN 1 and TRAIN 2 for training.")

    # chromosomes for outer CV
    chromosomes = [f"chr{i}" for i in range(2, 23)]  # matches your code

    log(f"Chromosomes: {chromosomes}")

    # ---- model params ----
    params_bin = {
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
    params_reg = {**params_bin, "objective": "regression", "metric": "rmse"}

    # save config
    config = {
        "train_1_path": train_1_path,
        "train_2_path": train_2_path,
        "test_path": test_path,
        "seed": seed,
        "params_reg": params_reg,
        "folds": n_inner_folds,
        "target_col": target_rank,
        "feature_count": len(feature_cols),
        "params_bin": params_bin,
        "mask_threshold": mask_threshold,
        "outer_chromosomes": chromosomes,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    log(f"✅ Config saved to {output_dir}/config.json")

    # ---- accumulators ----
    results_summary = []           # list[dict]  ← FIX
    results_inner = []             # list[dict]
    test_rg_result_chr = []        # list[np.ndarray]
    test_bn_result_chr = []        # list[np.ndarray]
    train_test_predictions_all = []  # list[pd.DataFrame]

    # ---- outer loop ----
    for _, val_chr in enumerate(chromosomes, start=1):
        
        log(f"\n🚀 Leave-one-chromosome: {val_chr}")

        df_train = df_full[df_full["chr"] != val_chr].copy()
        df_train_test = df_full[df_full["chr"] == val_chr].copy()

        inner_chrs = [c for c in chromosomes if c != val_chr]
        # split inner_chrs into n_inner_folds buckets
        folds = [inner_chrs[i::n_inner_folds] for i in range(n_inner_folds)]
        log("🧩 Chromosome folds:")

        pred_train_test_reg_folds, pred_train_test_prob_folds = [], []
        pred_test_reg_folds, pred_test_prob_folds = [], []

        for fold_idx, fset in enumerate(folds):
            log(f"Fold {fold_idx+1}: {fset}")

            df_inner_train = df_train[~df_train["chr"].isin(fset)].copy()
            df_inner_val = df_train[df_train["chr"].isin(fset)].copy()

            X_train = df_inner_train[feature_cols]
            y_train_reg = df_inner_train[target_rank]
            y_train_bin = df_inner_train[target_binary]

            X_val = df_inner_val[feature_cols]
            y_val_reg = df_inner_val[target_rank]
            y_val_bin = df_inner_val[target_binary]

            # binary model
            dtrain_bin = lgb.Dataset(X_train, label=y_train_bin)
            dval_bin = lgb.Dataset(X_val, label=y_val_bin, reference=dtrain_bin)
            model_bin = lgb.train(
                params_bin,
                dtrain_bin,
                valid_sets=[dtrain_bin, dval_bin],
                num_boost_round=2000,
                callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
            )

            # regression model
            dtrain_reg = lgb.Dataset(X_train, label=y_train_reg)
            dval_reg = lgb.Dataset(X_val, label=y_val_reg, reference=dtrain_reg)
            model_reg = lgb.train(
                params_reg,
                dtrain_reg,
                valid_sets=[dtrain_reg, dval_reg],
                num_boost_round=2000,
                callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
            )

            # inner validation eval
            pred_val_reg = model_reg.predict(X_val)
            pred_val_bin = model_bin.predict(X_val)
            rho_fold = spearmanr(y_val_reg.values, pred_val_reg)[0]
            rho_fold_masked = spearmanr(
                y_val_reg.values,
                pred_val_reg * (pred_val_bin >= mask_threshold).astype(int),
            )[0]
            log(f" → Validation fold {fold_idx+1} ({','.join(fset)}): ρ_reg={rho_fold:.4f}, ρ_reg_masked={rho_fold_masked:.4f}")

            # predict outer validation chr
            X_train_test = df_train_test[feature_cols]
            pred_train_test_reg = model_reg.predict(X_train_test)
            pred_train_test_prob = model_bin.predict(X_train_test)
            pred_train_test_reg_folds.append(pred_train_test_reg)
            pred_train_test_prob_folds.append(pred_train_test_prob)

            rho_outer = spearmanr(df_train_test[target_rank].values, pred_train_test_reg)[0]
            rho_outer_masked = spearmanr(
                df_train_test[target_rank].values,
                pred_train_test_reg * (pred_train_test_prob >= mask_threshold).astype(int),
            )[0]
            results_inner.append({
                "Leave one chr": val_chr,
                "inner_fold": fold_idx + 1,
                "Validation on chr": ",".join(fset),
                "rho_reg": rho_outer,
                "rho_reg_masked": rho_outer_masked,
            })
            log(f" → Predict chr-kfold {val_chr} in {fold_idx+1}: ρ_reg={rho_outer:.4f}, ρ_reg_masked={rho_outer_masked:.4f}")

            # predict test
            X_test = df_test[feature_cols]
            pred_test_reg = model_reg.predict(X_test)
            pred_test_prob = model_bin.predict(X_test)
            pred_test_reg_folds.append(pred_test_reg)
            pred_test_prob_folds.append(pred_test_prob)

        # ---- aggregate inner predictions for this outer chr ----
        mean_train_test_reg = np.mean(np.vstack(pred_train_test_reg_folds), axis=0)
        mean_train_test_prob = np.mean(np.vstack(pred_train_test_prob_folds), axis=0)

        mean_test_reg = np.mean(np.vstack(pred_test_reg_folds), axis=0)
        mean_test_prob = np.mean(np.vstack(pred_test_prob_folds), axis=0)

        test_rg_result_chr.append(mean_test_reg)
        test_bn_result_chr.append(mean_test_prob)

        # masked vs unmasked on the held-out chromosome
        train_test_mask = (mean_train_test_prob >= mask_threshold).astype(int)
        rho_nomask = spearmanr(df_train_test[target_rank].values, mean_train_test_reg)[0]
        rho_masked = spearmanr(df_train_test[target_rank].values, mean_train_test_reg * train_test_mask)[0]

        # ---- FIX: append dict, not DataFrame of scalars ----
        results_summary.append({
            "chr": val_chr,
            "rho_masked": rho_masked,
            "rho_nomask": rho_nomask,
        })
        log(f"📊 Average k-fold Predict on {val_chr} → No-mask ρ={rho_nomask:.4f}, Masked ρ={rho_masked:.4f}")

        # store per-gene preds on the held-out chromosome
        train_test_predictions_all.append(pd.DataFrame({
            "chr": val_chr,
            "gene_name": df_train_test["gene_name"].values,
            "mean_bn_prob": mean_train_test_prob,
            "gex_predicted": mean_train_test_reg,
        }))

    # ---- aggregate across outer chromosomes ----
    # FIX: concat, not DataFrame(list)
    train_test_predictions_df = (
        pd.concat(train_test_predictions_all, ignore_index=True) if train_test_predictions_all else pd.DataFrame()
    )
    results_summary_df = pd.DataFrame(results_summary)
    results_inner_df = pd.DataFrame(results_inner)

    # average test predictions across outer folds
    if len(test_rg_result_chr) > 0:
        mean_test_reg_all = np.mean(np.vstack(test_rg_result_chr), axis=0)
        mean_test_prob_all = np.mean(np.vstack(test_bn_result_chr), axis=0)
    else:
        mean_test_reg_all = np.zeros(len(df_test))
        mean_test_prob_all = np.zeros(len(df_test))

    test_predictions_df = pd.DataFrame({
        "gene_name": df_test["gene_name"].values,
        "mean_bn_prob": mean_test_prob_all,
        "gex_predicted": mean_test_reg_all,
    })

    # ---- logging summary ----
    if not results_summary_df.empty:
        log("\n===== Per-Chromosome Summary =====")
        log(results_summary_df.to_string(index=False))
        log(f"\nAverage ρ (No-mask): {results_summary_df['rho_nomask'].mean():.4f}")
        log(f"Average ρ (Masked):  {results_summary_df['rho_masked'].mean():.4f}")
    else:
        log("\n⚠️ No outer folds were run.")

    if not results_inner_df.empty:
        log("\n===== Inner Fold Detail (first few rows) =====")
        log(results_inner_df.head().to_string(index=False))

    return results_summary_df, results_inner_df, test_predictions_df, train_test_predictions_df


# file: utils/aggregate_gex_predicted.py
from __future__ import annotations

import os
import pandas as pd
import numpy as np
from typing import List, Optional

def aggregate_gex_predicted(
    dirs: List[str],
    threshold: float = 0.4,
    output_dir: Optional[str] = None,
    zip_name: str = "Wang_Ding_Yang_Project1.zip",
    tsv_name: str = "gex_predicted.tsv",
) -> pd.DataFrame:
    """
    Read `gex_predicted.tsv` from multiple directories, average `mean_bn_prob`
    and `gex_predicted` per gene, then mask the averaged `gex_predicted` by
    `(mean_bn_prob >= threshold)`. Return a DataFrame with columns:
    `gene_name`, `gex_predicted`.

    If `output_dir` is provided, it is created if missing and the function writes:
      1) `<output_dir>/<tsv_name>`  (TSV with columns: gene_name, gex_predicted)
      2) `<output_dir>/<zip_name>`  (ZIP containing CSV named `gex_predicted.csv`)
    """
    if not dirs:
        raise ValueError("`dirs` must not be empty.")

    required_cols = {"gene_name", "mean_bn_prob", "gex_predicted"}
    frames: list[pd.DataFrame] = []

    # Read per-directory TSV
    for d in dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Directory does not exist: {d}")

        path = os.path.join(d, "gex_predicted.tsv")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing file: {path}")

        print(f"[INFO] Reading: {path}")
        df = pd.read_csv(path, sep="\t")
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"File missing required columns {missing}: {path}")

        frames.append(df[["gene_name", "mean_bn_prob", "gex_predicted"]])

    if not frames:
        raise FileNotFoundError("No valid `gex_predicted.tsv` files were found.")

    # Aggregate
    all_df = pd.concat(frames, ignore_index=True)
    agg = (
        all_df.groupby("gene_name", as_index=False)[["mean_bn_prob", "gex_predicted"]]
        .mean()
        .rename(columns={
            "mean_bn_prob": "mean_bn_prob_mean",
            "gex_predicted": "gex_predicted_mean",
        })
    )

    # Mask by consensus probability (why: stabilize against single-file outliers)
    mask = (agg["mean_bn_prob_mean"] >= threshold).astype(int)
    agg["gex_predicted"] = agg["gex_predicted_mean"] * mask

    out = agg[["gene_name", "gex_predicted"]].sort_values("gene_name", kind="stable")

    # Optional saves (TSV + ZIP) in the same directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)  # auto-create if not exist

        # TSV save
        tsv_path = os.path.join(output_dir, tsv_name)
        out.to_csv(tsv_path, sep="\t", index=False)
        print(f"[INFO] Saved TSV to: {tsv_path}")

        # ZIP submission save with asserts
        pred = out["gex_predicted"].to_numpy()
        test_genes = out[["gene_name"]].copy()

        # Required asserts for downstream compatibility
        assert isinstance(pred, np.ndarray), 'Prediction array must be a numpy array'
        assert np.issubdtype(pred.dtype, np.number), 'Prediction array must be numeric'
        assert pred.shape[0] == len(test_genes), 'Each gene should have a unique predicted expression'

        file_name = 'gex_predicted.csv'         # PLEASE DO NOT CHANGE THIS
        save_path_zip = os.path.join(output_dir, zip_name)
        compression_options = dict(method="zip", archive_name=file_name)

        test_genes['gex_predicted'] = pred.tolist()
        test_genes[['gene_name', 'gex_predicted']].to_csv(
            save_path_zip, index=False, compression=compression_options
        )
        print(f"[INFO] Saved ZIP to: {save_path_zip}")

    # Final summary
    print(f"[INFO] Aggregated {len(frames)} file(s) from {len(dirs)} dir(s).")
    print(f"[INFO] Output rows (unique genes): {len(out)}")
    on_rate = float(mask.sum()) / float(len(mask)) if len(mask) else 0.0
    print(f"[INFO] Mask ON rate (@{threshold:.3f}): {on_rate:.2%}")

    return out
