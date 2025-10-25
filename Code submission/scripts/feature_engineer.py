from __future__ import annotations

import os
import numpy as np
import pandas as pd
import itertools
from scipy.stats import rankdata, pearsonr
from typing import Iterable

# ============================================================
#                FEATURE ENGINEERING HELPERS
# ============================================================



def safe_div(a, b):
    """
    Elementwise safe division with a small-denominator guard.

    Why: Avoid divide-by-zero warnings and unstable large values when |b|~0.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(b) > 1e-8, a / b, 0)


def check_new_columns(df: pd.DataFrame, prev_cols, context: str = "") -> pd.DataFrame:
    """
    Validate only the *newly added* columns in `df` vs `prev_cols`.

    - Coerce new columns to numeric (non-numeric → NaN).
    - Replace ±inf with NaN.
    - Print a brief report of columns containing NaN (and example gene names).
    - Return the DataFrame (mutated in-place for those new columns).

    Parameters
    ----------
    df : pd.DataFrame
        Current dataframe after feature creation.
    prev_cols : Iterable[str]
        Column names present *before* adding new features.
    context : str, optional
        Label added to logs to indicate which processing stage this is.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with new columns coerced to numeric and inf→NaN handled.
    """
    new_cols = [c for c in df.columns if c not in prev_cols]
    if not new_cols:
        return df  # No new columns to validate

    # Coerce all new columns to numeric where possible
    for col in new_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Only validate numeric new columns
    numeric_cols = [c for c in new_cols if np.issubdtype(df[c].dtype, np.number)]

    # Normalize inf values to NaN for uniform missingness handling
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    nan_ratio = df[numeric_cols].isna().mean()
    bad = nan_ratio[nan_ratio > 0].sort_values(ascending=False)

    if not bad.empty:
        print(f"[WARN] [{context}] {len(bad)} new features contain NaN/inf:")
        # Show a few example gene_name values for the most affected columns
        for col in bad.index[:5]:
            if "gene_name" in df.columns:
                nan_genes = df.loc[df[col].isna(), "gene_name"].head(5).tolist()
                print(f"   ↳ {col}: {df[col].isna().sum()} NaN — e.g. {nan_genes}")
            else:
                print(f"   ↳ {col}: {df[col].isna().sum()} NaN (no 'gene_name' column)")
    else:
        print(f"[OK] [{context}] All {len(numeric_cols)} new numeric features are valid.")

    return df



# ============================================================
#                 FEATURE RANK TRANSFORMATION
# ============================================================



def rank_transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create direction-aware rank-transformed features at two resolutions:
    per-chromosome and global. For activating marks, higher signal → higher rank;
    for repressive marks, higher signal → lower rank.

    Why add rank features:
      - Scale-free comparability: removes unit/scale differences across assays.
      - Outlier robustness: ranks are monotonic and less sensitive to heavy tails.
      - Prior biology: inverts repressive marks so higher repressive signal maps to lower rank.
      - Cross-chromosome stability: per-chrom ranks control chromosome-specific distributions;
        global ranks capture genome-wide ordering.

    Notes
    -----
    - NaNs are preserved as NaN in the rank outputs (why: missing signal should not imply order).
    - Ranks are normalized to [0, 1] by the number of non-NaN entries.
    - Columns considered: all numeric columns except core metadata/labels.
    """
    print("🔢 Performing rank transformation...")

    # Exclude metadata/labels from ranking (why: non-feature columns).
    exclude_cols: Iterable[str] = [
        "gene_name", "chr", "gene_start", "gene_end",
        "TSS_start", "TSS_end", "strand", "gex", "gex_rank"
    ]
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]

    # Biological direction (why: encode known activating vs repressive effects).
    activating_marks = ["DNase", "H3K27ac", "H3K4me3", "H3K36me3", "H3K4me1"]
    repressive_marks = ["H3K9me3", "H3K27me3"]

    df_ranked = df.copy()

    def rank_with_direction(vals: np.ndarray, col_name: str) -> np.ndarray:
        """Apply direction-aware ranking; keep NaNs; normalize ranks to [0,1].
        Why: robust ordering and prior-aware monotonic transformation.
        """
        out = np.full(vals.shape, np.nan, dtype=float)
        finite_mask = np.isfinite(vals)
        if not finite_mask.any():
            return out

        v = vals[finite_mask]

        # Determine direction by column name (why: mark-specific semantics).
        is_repressive = any(mark in col_name for mark in repressive_marks) or ("repress" in col_name.lower())
        v_to_rank = -v if is_repressive else v

        # Average-ties; scale by count of ranked (non-NaN) elements.
        ranks = rankdata(v_to_rank, method="average") / float(v_to_rank.size)
        out[finite_mask] = ranks
        return out

    # Chromosome-based ranking (why: control per-chrom distributional shifts).
    for chrom, subdf in df.groupby("chr", sort=False):
        idx = subdf.index
        for col in numeric_cols:
            df_ranked.loc[idx, f"{col}_chr_rank"] = rank_with_direction(subdf[col].to_numpy(), col)

    # Global ranking (why: genome-wide relative ordering).
    for col in numeric_cols:
        df_ranked[f"{col}_global_rank"] = rank_with_direction(df[col].to_numpy(), col)

    return df_ranked



# ============================================================
#                       gene_structure
# ============================================================



def add_gene_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    0️⃣ Gene structure features.

    What this adds
    --------------
    - `gene_length`: end - start
    - `tss_length`:  TSS_end - TSS_start

    Why this can help accuracy
    --------------------------
    - Expression scale proxy: Longer genes tend to accumulate different signal (e.g., coverage, elongation marks) and
      have different transcription kinetics; modeling length normalizes such scale effects.
    - Promoter complexity proxy: Wider TSS annotations can indicate multiple promoters / isoforms or annotation uncertainty,
      which correlates with regulatory complexity and observed expression variability.
    - Interaction with epigenetic features: Length/TSS span modulate how much chromatin signal (peaks, bw signal) is
      expected in a window; including these as covariates helps the model disambiguate biology from window-size artifacts.

    Returns
    -------
    pd.DataFrame
        Input `df` with two additional columns when the requisite inputs are present:
        `gene_length`, `tss_length`. Missing inputs yield NaN (kept explicit for downstream handling).
    """
    prev_cols = df.columns.copy()

    # Compute gene length when coordinates exist (why: expression- and window-scale covariate)
    if {"gene_start", "gene_end"}.issubset(df.columns):
        df["gene_length"] = df["gene_end"] - df["gene_start"]
    else:
        df["gene_length"] = np.nan

    # Compute TSS span when available (why: promoter complexity / annotation spread)
    if {"TSS_start", "TSS_end"}.issubset(df.columns):
        df["tss_length"] = df["TSS_end"] - df["TSS_start"]
    else:
        df["tss_length"] = np.nan

    # Validate only newly created columns (why: early catch of NaN/inf before modeling)
    return check_new_columns(df, prev_cols, context="gene_structure")


def add_promoter_gene_ratio(df: pd.DataFrame, marks: Iterable[str]) -> pd.DataFrame:
    """
    1️⃣ Promoter vs. gene-body ratio features.

    What this adds
    --------------
    For each `mark` in `marks`, if the required inputs exist:
      - `{mark}_ratio_mean` := {mark}_tss_signal_mean / {mark}_gene_signal_mean
      - `{mark}_ratio_std`  := {mark}_tss_signal_std  / {mark}_gene_signal_std
    Division is performed via `safe_div` to avoid divide-by-zero explosions.

    Why this can help accuracy
    --------------------------
    - Scale normalization: Ratios remove per-mark amplitude differences (lab/batch/assay scale),
      letting the model focus on relative promoter enrichment vs. gene-body spread.
    - Biological prior: Active genes often show strong promoter-centric signals
      (e.g., DNase, H3K27ac, H3K4me3) relative to gene bodies; repressed/elongating
      profiles look different. Ratios encode this shape succinctly.
    - Robust to coverage: Even when absolute coverage varies across samples,
      promoter-to-body contrast is more stable and predictive of transcriptional status.

    Returns
    -------
    pd.DataFrame
        Input `df` with zero or more new ratio columns per mark (only when inputs exist).
    """
    prev_cols = df.columns.copy()

    for mark in marks:
        gene_mean, tss_mean = f"{mark}_gene_signal_mean", f"{mark}_tss_signal_mean"
        gene_std,  tss_std  = f"{mark}_gene_signal_std",  f"{mark}_tss_signal_std"

        # Ratio of promoter mean to gene-body mean (why: promoter enrichment)
        if gene_mean in df and tss_mean in df:
            df[f"{mark}_ratio_mean"] = safe_div(df[tss_mean], df[gene_mean])

        # Ratio of promoter std to gene-body std (why: local signal concentration/dispersion)
        if gene_std in df and tss_std in df:
            df[f"{mark}_ratio_std"] = safe_div(df[tss_std], df[gene_std])

    # Validate new numeric columns for NaN/inf only among what was added
    return check_new_columns(df, prev_cols, context="promoter_gene_ratio")

def add_activation_balance(df):
    """
    2️⃣ Activation–repression balance features.

    What this adds
    --------------
    - `balance_H3K27` := H3K27ac_TSS_mean − H3K27me3_TSS_mean
    - `balance_H3K4`  := H3K4me3_TSS_mean − H3K9me3_TSS_mean

    Why this can help accuracy
    --------------------------
    - Antagonistic chromatin cues: Active promoters show high H3K27ac/H3K4me3,
      while repressed ones show high H3K27me3/H3K9me3. Their difference collapses
      opposing signals into a single, directional predictor.
    - Contrast over amplitude: Differences are less sensitive to global scaling
      (batch/coverage) than raw levels, highlighting regulatory state.
    - Nonlinearity hint: Provides a linear proxy for a latent decision boundary
      (activation minus repression) that many models can exploit directly.

    Returns
    -------
    pd.DataFrame
        Input `df` with one or two balance columns when inputs exist.
    """
    prev_cols = df.columns.copy()

    # H3K27 axis: acetylation (activation) vs trimethylation (repression)
    if all(c in df for c in ["H3K27ac_tss_signal_mean", "H3K27me3_tss_signal_mean"]):
        df["balance_H3K27"] = (
            df["H3K27ac_tss_signal_mean"] - df["H3K27me3_tss_signal_mean"]
        )

    # H3K4 axis: promoter activation (H3K4me3) vs heterochromatin (H3K9me3)
    if all(c in df for c in ["H3K4me3_tss_signal_mean", "H3K9me3_tss_signal_mean"]):
        df["balance_H3K4"] = (
            df["H3K4me3_tss_signal_mean"] - df["H3K9me3_tss_signal_mean"]
        )

    return check_new_columns(df, prev_cols, "activation_balance")

def add_promoter_entropy(df, activating_marks, repressive_marks):
    """
    3️⃣ Promoter entropy & variability features.

    What this adds
    --------------
    - `promoter_variability`: Row-wise standard deviation across available
      `{mark}_tss_signal_mean` columns (activating + repressive).
    - `promoter_entropy`: Shannon entropy over the *normalized* TSS mean
      signals across marks. Higher entropy → signals spread across marks;
      lower entropy → one/few marks dominate.

    Why this can help accuracy
    --------------------------
    - Regulatory clarity vs ambiguity:
      Active promoters often show coherent activating signatures (low entropy,
      low dispersion toward a dominant activating mark like H3K27ac/H3K4me3).
      Repressed/poised or mixed states can show diffuse patterns (higher entropy/variability).
    - Batch/scale robustness:
      Entropy uses a *distribution* over marks at the promoter, reducing sensitivity
      to absolute scale differences across assays or preprocessing.
    - Complementarity:
      Variability captures magnitude dispersion; entropy captures distributional shape.
      Together, they summarize promoter mark configuration beyond raw levels.

    Notes
    -----
    - Only uses columns present in `df` to avoid KeyErrors.
    - Adds a small epsilon (1e-8) in the log to maintain numerical stability.
    - NaNs are handled via `np.nan*` and `np.nan_to_num` to avoid propagating infinities.
    """
    prev_cols = df.columns.copy()

    # Collect available TSS mean columns across activating + repressive marks
    tss_cols = [
        f"{m}_tss_signal_mean"
        for m in (activating_marks + repressive_marks)
        if f"{m}_tss_signal_mean" in df
    ]
    if not tss_cols:
        return df  # No inputs available → nothing to compute

    # Row-wise variability (why: dispersion across marks at promoter)
    df["promoter_variability"] = df[tss_cols].std(axis=1)

    # Row-wise normalized distribution over marks (why: scale-free composition)
    # Divide each row by its row-sum; rows with sum=0 become 0 after nan_to_num.
    row_sum = df[tss_cols].sum(axis=1)
    norm_vals = df[tss_cols].div(row_sum.replace(0, np.nan), axis=0)

    # Shannon entropy (why: concentration vs diffuseness of promoter signals)
    # Use nan_to_num to treat NaNs as 0 probability; add eps to log for stability.
    eps = 1e-8
    probs = np.nan_to_num(norm_vals.to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    df["promoter_entropy"] = -np.nansum(probs * np.log(probs + eps), axis=1)

    # Validate only new columns for numeric issues
    return check_new_columns(df, prev_cols, "promoter_entropy")

def add_chromatin_indices(df, activating_marks, repressive_marks):
    """
    4️⃣ Openness & repression index features.

    What this adds
    --------------
    - `openness_index`: Row-wise mean of available activating promoter signals
      `{mark}_tss_signal_mean` where `mark` ∈ activating_marks (e.g., DNase, H3K27ac, H3K4me3, H3K36me3, H3K4me1).
    - `repression_index`: Row-wise mean of available repressive promoter signals
      `{mark}_tss_signal_mean` where `mark` ∈ repressive_marks (e.g., H3K27me3, H3K9me3).

    Why this can help accuracy
    --------------------------
    - Compact state summary: Averages across related marks to produce stable,
      low-variance proxies of promoter openness vs repression.
    - Noise reduction: Aggregation attenuates assay-specific noise and batch effects,
      improving generalization compared to using single marks alone.
    - Complementarity: Paired indices let models capture both activation and repression
      magnitudes, enabling simple interactions (e.g., differences/ratios) to reflect net state.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with two new columns when inputs exist:
        `openness_index` and `repression_index`.
    """
    prev_cols = df.columns.copy()

    # Collect available TSS mean columns for activating / repressive marks
    act_cols = [f"{m}_tss_signal_mean" for m in activating_marks if f"{m}_tss_signal_mean" in df]
    rep_cols = [f"{m}_tss_signal_mean" for m in repressive_marks if f"{m}_tss_signal_mean" in df]

    # Mean across activating marks at promoter (why: robust openness proxy)
    if act_cols:
        df["openness_index"] = df[act_cols].mean(axis=1)

    # Mean across repressive marks at promoter (why: robust repression proxy)
    if rep_cols:
        df["repression_index"] = df[rep_cols].mean(axis=1)

    # Validate only the newly added numeric columns
    return check_new_columns(df, prev_cols, "chromatin_indices")


def add_strand_features(df):
    """
    5️⃣ Strand-aware features.

    What this adds
    --------------
    - `strand_is_plus`: 1 if strand == '+', else 0
    - `strand_is_minus`: 1 if strand == '-', else 0

    Why this can help accuracy
    --------------------------
    - Directional context: Many promoter/TSS-relative windows and chromatin
      features depend on transcription direction (upstream vs downstream flips
      by strand). Encoding strand lets the model capture such asymmetries.
    - Interaction ready: Binary indicators can interact with positional or
      window-based features to form strand-corrected effects without manual flipping.
    """
    prev_cols = df.columns.copy()

    if "strand" in df.columns:
        df["strand_is_plus"] = (df["strand"] == "+").astype(int)
        df["strand_is_minus"] = (df["strand"] == "-").astype(int)

    return check_new_columns(df, prev_cols, "strand_features")



def add_cross_mark_interactions(df: pd.DataFrame, marks) -> pd.DataFrame:
    """
    6️⃣ Pairwise cross-mark interaction features (enhanced).

    What this adds
    --------------
    For each unordered pair (m1, m2):
      • Promoter mean interactions (if both `{mark}_tss_signal_mean` exist):
          - `{m1}_{m2}_mul`        := mean(m1@TSS) * mean(m2@TSS)
          - `{m1}_{m2}_ratio`      := mean(m1@TSS) / mean(m2@TSS)
          - `{m2}_{m1}_ratio`      := mean(m2@TSS) / mean(m1@TSS)   (bidirectional)
          - `{m1}_{m2}_diff`       := mean(m1@TSS) - mean(m2@TSS)
          - `{m1}_{m2}_absdiff`    := |mean(m1@TSS) - mean(m2@TSS)|
      • Promoter std interactions (if both `{mark}_tss_signal_std` exist):
          - Same set as above with a `_std` suffix, e.g. `{m1}_{m2}_mul_std`
      • Promoter–Gene cross interactions (if inputs exist):
          - `{m1}_tss_{m2}_gene_cross`        := mean(m1@TSS) * mean(m2@gene)
          - `{m2}_tss_{m1}_gene_cross`
          - `{m1}_tss_{m2}_gene_std_cross`    := std(m1@TSS) * std(m2@gene)
          - `{m2}_tss_{m1}_gene_std_cross`

    Why this can help accuracy
    --------------------------
    - Combinatorial regulation: Gene expression emerges from *joint* chromatin cues.
      Pairwise products/ratios/differences capture synergy, antagonism, and balance
      between marks (e.g., H3K27ac × DNase for active promoters, H3K27me3 vs H3K4me3
      contrast for bivalency).
    - Nonlinearity with simple models: Multiplicative/ratio features approximate
      interactions that tree/LGBM models learn, but making them explicit can improve
      sample efficiency and stability.
    - Promoter–gene coupling: Cross terms between promoter and gene-body signals
      summarize initiation ↔ elongation dynamics that often correlate with expression.

    Notes
    -----
    - Uses safe division to avoid divide-by-zero explosions.
    - Std-based interactions are suffixed with `_std` to avoid name collisions with mean-based ones.
    """
    prev_cols = df.columns.copy()

    def _safe_div(a: pd.Series, b: pd.Series) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            a = a.to_numpy(dtype=float)
            b = b.to_numpy(dtype=float)
            return np.where(np.abs(b) > 1e-8, a / b, 0.0)

    # --- TSS mean interactions ---
    for m1, m2 in itertools.combinations(marks, 2):
        c1, c2 = f"{m1}_tss_signal_mean", f"{m2}_tss_signal_mean"
        if c1 in df and c2 in df:
            a, b = df[c1].astype(float), df[c2].astype(float)
            df[f"{m1}_{m2}_mul"]     = a * b
            df[f"{m1}_{m2}_ratio"]   = _safe_div(a, b)
            df[f"{m2}_{m1}_ratio"]   = _safe_div(b, a)  # bidirectional ratios
            df[f"{m1}_{m2}_diff"]    = a - b
            df[f"{m1}_{m2}_absdiff"] = np.abs(a - b)

    # --- Promoter–Gene mean cross interactions ---
    for m1, m2 in itertools.combinations(marks, 2):
        tss_col_1, gene_col_2 = f"{m1}_tss_signal_mean", f"{m2}_gene_signal_mean"
        tss_col_2, gene_col_1 = f"{m2}_tss_signal_mean", f"{m1}_gene_signal_mean"
        if tss_col_1 in df and gene_col_2 in df:
            df[f"{m1}_tss_{m2}_gene_cross"] = df[tss_col_1].astype(float) * df[gene_col_2].astype(float)
        if tss_col_2 in df and gene_col_1 in df:
            df[f"{m2}_tss_{m1}_gene_cross"] = df[tss_col_2].astype(float) * df[gene_col_1].astype(float)

    # --- TSS std interactions (suffix with `_std` to avoid overwriting mean interactions) ---
    for m1, m2 in itertools.combinations(marks, 2):
        s1, s2 = f"{m1}_tss_signal_std", f"{m2}_tss_signal_std"
        if s1 in df and s2 in df:
            a, b = df[s1].astype(float), df[s2].astype(float)
            df[f"{m1}_{m2}_mul_std"]     = a * b
            df[f"{m1}_{m2}_ratio_std"]   = _safe_div(a, b)
            df[f"{m2}_{m1}_ratio_std"]   = _safe_div(b, a)
            df[f"{m1}_{m2}_diff_std"]    = a - b
            df[f"{m1}_{m2}_absdiff_std"] = np.abs(a - b)

    # --- Promoter–Gene std cross interactions ---
    for m1, m2 in itertools.combinations(marks, 2):
        tss_std_1, gene_std_2 = f"{m1}_tss_signal_std", f"{m2}_gene_signal_std"
        tss_std_2, gene_std_1 = f"{m2}_tss_signal_std", f"{m1}_gene_signal_std"
        if tss_std_1 in df and gene_std_2 in df:
            df[f"{m1}_tss_{m2}_gene_std_cross"] = df[tss_std_1].astype(float) * df[gene_std_2].astype(float)
        if tss_std_2 in df and gene_std_1 in df:
            df[f"{m2}_tss_{m1}_gene_std_cross"] = df[tss_std_2].astype(float) * df[gene_std_1].astype(float)

    return check_new_columns(df, prev_cols, "cross_mark_interactions_v2")


def add_activation_repression_indices(df):
    """
    7️⃣ Summarized activation/repression indices.

    What this adds
    --------------
    - `activation_balance` := H3K27ac@TSS − H3K27me3@TSS  (acetylation vs Polycomb repression)
    - `promoter_activity`  := H3K4me3@TSS − H3K9me3@TSS   (promoter activation vs heterochromatin)
    - `repression_index`   := mean(H3K9me3@TSS, H3K27me3@TSS)
    - `activation_index`   := mean(H3K27ac@TSS, H3K4me3@TSS, DNase@TSS)

    Why this can help accuracy
    --------------------------
    - Net-state signal: Differences/means collapse multiple marks into a single,
      low-noise estimate of promoter openness vs repression—closer to the latent
      decision boundary for expression.
    - Batch/scale robustness: Aggregation (mean/diff) reduces sensitivity to per-assay
      amplitude/batch effects compared to raw levels.
    - Biological prior: Encodes known antagonistic pairs (K27ac↔K27me3, K4me3↔K9me3)
      and synergy among activating marks (K27ac/K4me3/DNase).

    Returns
    -------
    pd.DataFrame
        Input `df` augmented with zero or more index columns, depending on availability.
    """
    prev_cols = df.columns.copy()

    # Activation minus repression along the H3K27 axis
    if all(c in df for c in ["H3K27ac_tss_signal_mean", "H3K27me3_tss_signal_mean"]):
        df["activation_balance"] = (
            df["H3K27ac_tss_signal_mean"] - df["H3K27me3_tss_signal_mean"]
        )

    # Promoter activation minus heterochromatin along the H3K4 axis
    if all(c in df for c in ["H3K4me3_tss_signal_mean", "H3K9me3_tss_signal_mean"]):
        df["promoter_activity"] = (
            df["H3K4me3_tss_signal_mean"] - df["H3K9me3_tss_signal_mean"]
        )

    # Aggregate repression strength
    if all(c in df for c in ["H3K9me3_tss_signal_mean", "H3K27me3_tss_signal_mean"]):
        df["repression_index"] = (
            df["H3K9me3_tss_signal_mean"] + df["H3K27me3_tss_signal_mean"]
        ) / 2.0

    # Aggregate activation strength
    if all(c in df for c in ["H3K27ac_tss_signal_mean", "H3K4me3_tss_signal_mean", "DNase_tss_signal_mean"]):
        df["activation_index"] = (
            df["H3K27ac_tss_signal_mean"]
            + df["H3K4me3_tss_signal_mean"]
            + df["DNase_tss_signal_mean"]
        ) / 3.0

    return check_new_columns(df, prev_cols, "activation_repression_indices")


def add_axis_and_delta(df, marks):
    """
    8️⃣ Axis sum & promoter–body delta features.

    What this adds
    --------------
    For each mark in `marks`, when both inputs exist:
      - `{mark}_axis_sum`              := {mark}_tss_signal_mean + {mark}_gene_signal_mean
      - `{mark}_promoter_body_delta`   := {mark}_tss_signal_mean - {mark}_gene_signal_mean

    Why this can help accuracy
    --------------------------
    - Total signal proxy (axis_sum): Approximates overall chromatin load for a mark
      around a gene (promoter + body). Helpful when absolute abundance correlates
      with expression (e.g., broad H3K36me3 on active genes).
    - Promoter enrichment (promoter_body_delta): Positive values indicate promoter-centric
      activation relative to the gene body (common for DNase/H3K27ac/H3K4me3), while
      negative values indicate body-biased signal (elongation/heterochromatin contexts).
      This contrast is often more predictive than either region alone.
    - Batch robustness: Sum and difference reduce some per-assay scale noise compared
      to relying solely on raw region means.

    Returns
    -------
    pd.DataFrame
        Input `df` augmented with zero or more `{mark}_axis_sum` and
        `{mark}_promoter_body_delta` columns.
    """
    prev_cols = df.columns.copy()

    for mark in marks:
        g_mean, t_mean = f"{mark}_gene_signal_mean", f"{mark}_tss_signal_mean"
        if g_mean in df and t_mean in df:
            # Overall abundance of the mark near the gene (promoter + body)
            df[f"{mark}_axis_sum"] = df[t_mean] + df[g_mean]
            # Promoter enrichment relative to gene body
            df[f"{mark}_promoter_body_delta"] = df[t_mean] - df[g_mean]

    return check_new_columns(df, prev_cols, "axis_and_delta")

def add_tss_distance_feature(df):
    """
    9️⃣ TSS–gene boundary distance features.

    What this adds
    --------------
    - `tss_to_gene_boundary_min` : min distance from TSS midpoint to {gene_start, gene_end}
    - `tss_to_gene_boundary_max` : max distance from TSS midpoint to {gene_start, gene_end}
    - `tss_to_gene_start_dist`   : |TSS_mid - gene_start|
    - `tss_to_gene_end_dist`     : |TSS_mid - gene_end|
    - Normalized (by gene length) ratios for all the above:
        * `_gene_region_ratio` variants

    Why this can help accuracy
    --------------------------
    - Promoter placement: TSS closer to a boundary can imply compact promoters,
      alternative TSS usage, or annotation uncertainty—affecting observed marks.
    - Scale normalization: Length-normalized distances control for gene size,
      making features comparable across genes and reducing spurious correlations.
    - Interaction potential: Combines well with strand-aware and promoter/body
      chromatin features to capture architecture-dependent effects.

    Notes
    -----
    - Uses midpoint of `[TSS_start, TSS_end]` to reduce annotation noise.
    - Ratios use safe division to avoid divide-by-zero for zero-length genes.
    """
    prev_cols = df.columns.copy()
    required_cols = ["gene_start", "gene_end", "TSS_end", "TSS_start"]
    if not all(c in df.columns for c in required_cols):
        print("[WARN] Missing required columns for TSS distance computation.")
        df["tss_to_gene_boundary_min"] = np.nan
        return df

    # --- Helper: safe division to avoid divide-by-zero explosions ---
    def _safe_div(a, b, eps=1e-8):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(b) > eps, a / b, 0.0)

    # --- Compute TSS midpoint (why: stabilize against boundary noise) ---
    Mid_tss = (df["TSS_start"].astype(float) + df["TSS_end"].astype(float)) / 2.0

    # --- Distances to each gene boundary ---
    dist_to_start = np.abs(Mid_tss - df["gene_start"].astype(float))
    dist_to_end   = np.abs(Mid_tss - df["gene_end"].astype(float))
    gene_len      = np.abs(df["gene_end"].astype(float) - df["gene_start"].astype(float))

    # --- Aggregate distances ---
    df["tss_to_gene_boundary_min"] = np.minimum(dist_to_start, dist_to_end)
    df["tss_to_gene_boundary_max"] = np.maximum(dist_to_start, dist_to_end)
    df["tss_to_gene_start_dist"]   = dist_to_start
    df["tss_to_gene_end_dist"]     = dist_to_end

    # --- Length-normalized ratios (why: cross-gene comparability) ---
    df["tss_to_gene_boundary_min_gene_region_ratio"] = _safe_div(
        df["tss_to_gene_boundary_min"], gene_len
    )
    df["tss_to_gene_boundary_max_gene_region_ratio"] = _safe_div(
        df["tss_to_gene_boundary_max"], gene_len
    )
    df["tss_to_gene_start_dist_gene_region_ratio"] = _safe_div(
        df["tss_to_gene_start_dist"], gene_len
    )
    df["tss_to_gene_end_dist_gene_region_ratio"] = _safe_div(
        df["tss_to_gene_end_dist"], gene_len
    )

    return check_new_columns(df, prev_cols, "tss_to_gene_boundary_min")

def add_advanced_chromatin_features(df, marks):
    """
    🧬 Advanced chromatin-level feature engineering
    ---------------------------------------------------
    Adds biologically motivated summary features that capture spatial contrasts,
    synergy/antagonism between marks, and cross-region coherence:

      1) Accessibility gradient (promoter−gene body, length-normalized)
      2) Promoter asymmetry (directional normalized contrast)
      3) Enhancer–Promoter coupling (H3K27ac@body × H3K4me3@TSS)
      4) Co-accessibility / synergy (H3K27ac × DNase) and ratio
      5) Bivalent index / balance (activation vs repression at TSS)
      6) Promoter–Gene body coherence (per-gene corr across marks)
      7) Chromatin entropy diversity (mean/std across per-mark entropies)

    Why these help accuracy
    -----------------------
    - Contrast features (Δ, normalized Δ) emphasize promoter-centric activity vs body signal.
    - Synergy/ratio encode nonlinear interactions (activation with accessibility).
    - Bivalency captures antagonistic chromatin (poised states).
    - Coherence measures whether marks move together across promoter/body for a gene.
    - Entropy aggregates distributional complexity across marks, reducing noise and
      aligning with regulatory state.

    Requirements
    ------------
    - Expects log/standardized inputs like `{mark}_tss_logz_mean` / `{mark}_gene_logz_mean`
      for some parts, and corresponding `_entropy` columns if present.
    - Relies on `safe_div` to avoid divide-by-zero and on `check_new_columns` for QA.
    """
    prev_cols = df.columns.copy()

    # ------------------------------------------------------------
    # 1) Accessibility gradient per mark: normalize by gene length
    #    Rationale: stronger TSS vs body contrast per kb ≈ promoter sharpness.
    # ------------------------------------------------------------
    if "gene_length" in df:
        for mark in marks:
            tss_col = f"{mark}_tss_logz_mean"
            gene_col = f"{mark}_gene_logz_mean"
            if tss_col in df and gene_col in df:
                df[f"{mark}_accessibility_gradient"] = safe_div(
                    (df[tss_col] - df[gene_col]), df["gene_length"] + 1e-8
                )

    # ------------------------------------------------------------
    # 2) Promoter asymmetry (example with DNase)
    #    Rationale: normalized contrast reduces batch scale; captures promoter sharpness.
    # ------------------------------------------------------------
    if all(c in df.columns for c in ["DNase_tss_logz_mean", "DNase_gene_logz_mean"]):
        df["DNase_promoter_asymmetry"] = safe_div(
            df["DNase_tss_logz_mean"] - df["DNase_gene_logz_mean"],
            df["DNase_tss_logz_mean"] + df["DNase_gene_logz_mean"] + 1e-8,
        )

    # ------------------------------------------------------------
    # 3) Enhancer–Promoter coupling (H3K27ac body × H3K4me3 TSS)
    #    Rationale: initiation (H3K4me3@TSS) with enhancer/elongation readout (K27ac@gene).
    # ------------------------------------------------------------
    if all(c in df.columns for c in ["H3K27ac_gene_logz_mean", "H3K4me3_tss_logz_mean"]):
        df["enhancer_promoter_synergy"] = (
            df["H3K27ac_gene_logz_mean"] * df["H3K4me3_tss_logz_mean"]
        )

    # ------------------------------------------------------------
    # 4) H3K27ac × DNase synergy and ratio at TSS
    #    Rationale: accessibility with activation mark jointly signal active promoters.
    # ------------------------------------------------------------
    if all(c in df.columns for c in ["H3K27ac_tss_logz_mean", "DNase_tss_logz_mean"]):
        df["H3K27ac_DNase_synergy"] = (
            df["H3K27ac_tss_logz_mean"] * df["DNase_tss_logz_mean"]
        )
        df["H3K27ac_DNase_ratio"] = safe_div(
            df["H3K27ac_tss_logz_mean"], df["DNase_tss_logz_mean"]
        )

    # ------------------------------------------------------------
    # 5) Bivalency at TSS: product and balance (K27ac vs K27me3)
    #    Rationale: antagonism/poised state signature improves classification around TSS.
    # ------------------------------------------------------------
    if all(c in df.columns for c in ["H3K27ac_tss_logz_mean", "H3K27me3_tss_logz_mean"]):
        df["bivalent_index"] = (
            df["H3K27ac_tss_logz_mean"] * df["H3K27me3_tss_logz_mean"]
        )
        df["bivalent_balance"] = (
            df["H3K27ac_tss_logz_mean"] - df["H3K27me3_tss_logz_mean"]
        )

    # ------------------------------------------------------------
    # 6) Promoter–Gene coherence across marks
    #    Rationale: genes with coordinated promoter/body signals across marks tend to be active.
    # ------------------------------------------------------------

    # Cache presence once to avoid repeated column checks
    tss_cols = [f"{m}_tss_logz_mean" for m in marks if f"{m}_tss_logz_mean" in df.columns]
    gene_cols = [f"{m}_gene_logz_mean" for m in marks if f"{m}_gene_logz_mean" in df.columns]
    usable_marks = [m for m in marks if f"{m}_tss_logz_mean" in df.columns and f"{m}_gene_logz_mean" in df.columns]

    def promoter_gene_coherence(row):
        vals_tss = [row[f"{m}_tss_logz_mean"] for m in usable_marks]
        vals_gene = [row[f"{m}_gene_logz_mean"] for m in usable_marks]
        t = np.array(vals_tss, dtype=float)
        g = np.array(vals_gene, dtype=float)
        # Need at least 2 paired marks and no NaNs for Pearson r
        if t.size < 2 or np.isnan(t).any() or np.isnan(g).any():
            return 0.0
        try:
            r = pearsonr(t, g)[0]
            return 0.0 if np.isnan(r) else float(r)
        except Exception:
            return 0.0

    if usable_marks:
        df["promoter_gene_coherence"] = df.apply(promoter_gene_coherence, axis=1)

    # ------------------------------------------------------------
    # 7) Chromatin entropy diversity: mean/std over *_entropy features
    #    Rationale: summarizes distributional complexity across marks.
    # ------------------------------------------------------------
    entropy_cols = [c for c in df.columns if c.endswith("_entropy")]
    if entropy_cols:
        df["chromatin_entropy_mean"] = df[entropy_cols].mean(axis=1)
        df["chromatin_entropy_std"] = df[entropy_cols].std(axis=1)

    return check_new_columns(df, prev_cols, "advanced_chromatin_features")


def add_advanced_chromatin_features(df, marks):
    """
    🧬 Advanced chromatin-level feature engineering

    Adds biologically motivated summary features capturing spatial contrasts,
    synergy/antagonism between marks, and cross-region coherence:

      1) Accessibility gradient: (promoter − gene-body), length-normalized.
         Why: sharper promoter-vs-body contrast per kb ≈ active initiation.
      2) Promoter asymmetry: normalized (TSS − gene)/(TSS + gene).
         Why: scale-free measure of promoter-centric accessibility.
      3) Enhancer–Promoter coupling: H3K27ac@gene × H3K4me3@TSS.
         Why: links enhancer/elongation readout with promoter initiation.
      4) Co-accessibility / synergy (H3K27ac × DNase) + ratio.
         Why: activation with open chromatin jointly marks active promoters.
      5) Bivalent index / balance (H3K27ac vs H3K27me3 at TSS).
         Why: antagonism/poised states predictive of transcriptional status.
      6) Promoter–Gene coherence: per-gene Pearson correlation (TSS vs gene across marks).
         Why: coordinated promoter/body signals tend to track expression.
      7) Chromatin entropy diversity: mean/std of *_entropy features.
         Why: summarizes distributional complexity while reducing noise.
    """
    prev_cols = df.columns.copy()

    # ------------------------------------------------------------
    # 1) Accessibility gradient per mark (length-normalized contrast)
    # ------------------------------------------------------------
    if "gene_length" in df:
        for mark in marks:
            tss_col = f"{mark}_tss_logz_mean"
            gene_col = f"{mark}_gene_logz_mean"
            if tss_col in df and gene_col in df:
                df[f"{mark}_accessibility_gradient"] = safe_div(
                    (df[tss_col] - df[gene_col]), df["gene_length"] + 1e-8
                )

    # ------------------------------------------------------------
    # 2) Promoter asymmetry (example with DNase)
    # ------------------------------------------------------------
    if all(c in df.columns for c in ["DNase_tss_logz_mean", "DNase_gene_logz_mean"]):
        df["DNase_promoter_asymmetry"] = safe_div(
            df["DNase_tss_logz_mean"] - df["DNase_gene_logz_mean"],
            df["DNase_tss_logz_mean"] + df["DNase_gene_logz_mean"] + 1e-8
        )

    # ------------------------------------------------------------
    # 3) Enhancer–Promoter coupling (H3K27ac × H3K4me3)
    # ------------------------------------------------------------
    if all(c in df.columns for c in ["H3K27ac_gene_logz_mean", "H3K4me3_tss_logz_mean"]):
        df["enhancer_promoter_synergy"] = (
            df["H3K27ac_gene_logz_mean"] * df["H3K4me3_tss_logz_mean"]
        )

    # ------------------------------------------------------------
    # 4) H3K27ac × DNase synergy / ratio at TSS
    # ------------------------------------------------------------
    if all(c in df.columns for c in ["H3K27ac_tss_logz_mean", "DNase_tss_logz_mean"]):
        df["H3K27ac_DNase_synergy"] = (
            df["H3K27ac_tss_logz_mean"] * df["DNase_tss_logz_mean"]
        )
        df["H3K27ac_DNase_ratio"] = safe_div(
            df["H3K27ac_tss_logz_mean"], df["DNase_tss_logz_mean"]
        )

    # ------------------------------------------------------------
    # 5) Bivalency: product and balance at TSS (K27ac vs K27me3)
    # ------------------------------------------------------------
    if all(c in df.columns for c in ["H3K27ac_tss_logz_mean", "H3K27me3_tss_logz_mean"]):
        df["bivalent_index"] = (
            df["H3K27ac_tss_logz_mean"] * df["H3K27me3_tss_logz_mean"]
        )
        df["bivalent_balance"] = (
            df["H3K27ac_tss_logz_mean"] - df["H3K27me3_tss_logz_mean"]
        )

    # ------------------------------------------------------------
    # 6) Promoter–Gene coherence: per-row Pearson r across marks
    # ------------------------------------------------------------

    def promoter_gene_coherence(row):
        tss_vals, gene_vals = [], []
        for m in marks:
            tss_col, gene_col = f"{m}_tss_logz_mean", f"{m}_gene_logz_mean"
            if tss_col in df.columns and gene_col in df.columns:
                tss_vals.append(row.get(tss_col, np.nan))
                gene_vals.append(row.get(gene_col, np.nan))
        tss_vals, gene_vals = np.array(tss_vals), np.array(gene_vals)
        if np.isnan(tss_vals).any() or np.isnan(gene_vals).any() or len(tss_vals) < 2:
            return 0.0  # conservative default when insufficient data
        try:
            return pearsonr(tss_vals, gene_vals)[0]
        except Exception:
            return 0.0

    df["promoter_gene_coherence"] = df.apply(promoter_gene_coherence, axis=1)

    # ------------------------------------------------------------
    # 7) Chromatin entropy diversity: aggregate *_entropy features
    # ------------------------------------------------------------
    entropy_cols = [c for c in df.columns if c.endswith("_entropy")]
    if entropy_cols:
        df["chromatin_entropy_mean"] = df[entropy_cols].mean(axis=1)
        df["chromatin_entropy_std"] = df[entropy_cols].std(axis=1)

    return check_new_columns(df, prev_cols, "advanced_chromatin_features")


def add_cross_layer_features(df: pd.DataFrame, marks, prefix: str = "cross", bw_norm: str = "logz") -> pd.DataFrame:
    """
    🧬 Cross-layer features between BED and bigWig for each histone mark.

    Expects both:
      • BED layer  : {mark}_tss_signal_mean, {mark}_gene_signal_mean, {mark}_tss_peak_density, {mark}_tss_peak_entropy, ...
      • bigWig layer: {mark}_tss_{bw_norm}_mean, {mark}_tss_entropy, {mark}_gene_{bw_norm}_mean, ...

    What this adds
    --------------
    1) Ratio features: bigWig intensity normalized by BED peak density/entropy.
    2) Difference features: bigWig entropy vs BED peak entropy; bigWig vs BED mean deltas.
    3) Weighted features: density/entropy × bigWig intensity (importance-weighted signal).
    4) Promoter–gene balance on bigWig: delta and ratio at TSS vs gene body.
    5) BED–bigWig interactions: multiplicative coupling (structure × intensity).

    Why this can improve accuracy
    -----------------------------
    - Complementarity: BED encodes discrete peak structure; bigWig encodes continuous signal.
      Crossing them captures cases where strong peaks have weak coverage (or vice versa).
    - Normalization: Ratios reduce batch/scale effects by comparing across layers.
    - Synergy: Multiplicative terms approximate interactions that non-linear models exploit,
      improving separability of active vs repressed promoters.
    - Localization: Promoter–gene contrasts distill initiation vs elongation differences.

    Parameters
    ----------
    df : pd.DataFrame
        Merged table containing both BED and bigWig features.
    marks : Iterable[str]
        Histone/assay names, e.g., ["H3K27ac", "H3K4me3", "H3K27me3", "DNase"].
    prefix : str, default "cross"
        Prefix for newly created cross-layer feature names.
    bw_norm : str, default "logz"
        Normalization tag used in bigWig columns (e.g., "logz", "zscore", "log", "raw").

    Returns
    -------
    pd.DataFrame
        Input `df` augmented with cross-layer features. New numeric columns are
        cleaned (±inf→NaN→0.0) to avoid downstream issues.
    """
    prev_cols = df.columns.copy()

    def safe_div(a, b):
        """Safe division to avoid divide-by-zero blowups."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(b) > 1e-8, a / b, 0.0)

    for mark in marks:
        # --- BED layer columns (structure/peaks) ---
        tss_bed_mean   = f"{mark}_tss_signal_mean"
        gene_bed_mean  = f"{mark}_gene_signal_mean"
        tss_peak_dens  = f"{mark}_tss_peak_density"   if f"{mark}_tss_peak_density"   in df else None
        gene_peak_dens = f"{mark}_gene_peak_density"  if f"{mark}_gene_peak_density"  in df else None
        tss_peak_entr  = f"{mark}_tss_peak_entropy"   if f"{mark}_tss_peak_entropy"   in df else None
        gene_peak_entr = f"{mark}_gene_peak_entropy"  if f"{mark}_gene_peak_entropy"  in df else None

        # --- bigWig layer columns (continuous intensity) ---
        tss_bw_mean    = f"{mark}_tss_{bw_norm}_mean"
        gene_bw_mean   = f"{mark}_gene_{bw_norm}_mean"
        tss_bw_std     = f"{mark}_tss_{bw_norm}_std"
        gene_bw_std    = f"{mark}_gene_{bw_norm}_std"
        tss_bw_entropy = f"{mark}_tss_entropy"
        gene_bw_entropy= f"{mark}_gene_entropy"

        # Skip marks without bigWig layer present
        if not any(c in df.columns for c in (tss_bw_mean, gene_bw_mean)):
            continue

        # ============================================================
        # 1) Ratio-type cross features (scale-normalized intensity)
        # ============================================================
        if tss_bw_mean in df and tss_peak_dens:
            df[f"{prefix}_{mark}_tss_bw_over_peak_density"] = safe_div(df[tss_bw_mean], df[tss_peak_dens])
        if tss_bw_std in df and tss_peak_entr:
            df[f"{prefix}_{mark}_tss_bw_std_over_peak_entropy"] = safe_div(df[tss_bw_std], df[tss_peak_entr])

        if gene_bw_mean in df and gene_peak_dens:
            df[f"{prefix}_{mark}_gene_bw_over_peak_density"] = safe_div(df[gene_bw_mean], df[gene_peak_dens])
        if gene_bw_std in df and gene_peak_entr:
            df[f"{prefix}_{mark}_gene_bw_std_over_peak_entropy"] = safe_div(df[gene_bw_std], df[gene_peak_entr])

        # ============================================================
        # 2) Difference-type cross features (shape mismatches)
        # ============================================================
        if tss_bw_entropy in df and tss_peak_entr:
            df[f"{prefix}_{mark}_tss_entropy_diff"] = df[tss_bw_entropy] - df[tss_peak_entr]
        if tss_bw_mean in df and tss_bed_mean in df:
            df[f"{prefix}_{mark}_tss_bw_vs_bed_mean_diff"] = df[tss_bw_mean] - df[tss_bed_mean]
        if gene_bw_entropy in df and gene_peak_entr:
            df[f"{prefix}_{mark}_gene_entropy_diff"] = df[gene_bw_entropy] - df[gene_peak_entr]

        # ============================================================
        # 3) Weighted cross features (importance-weighted intensity)
        # ============================================================
        if tss_bw_mean in df and tss_peak_dens:
            df[f"{prefix}_{mark}_tss_peak_density_times_bw_mean"] = df[tss_peak_dens] * df[tss_bw_mean]
        if tss_bw_entropy in df and tss_peak_dens:
            df[f"{prefix}_{mark}_tss_peak_density_times_bw_entropy"] = df[tss_peak_dens] * df[tss_bw_entropy]
        if tss_bw_mean in df and tss_peak_entr:
            df[f"{prefix}_{mark}_tss_peak_entropy_times_bw_mean"] = df[tss_peak_entr] * df[tss_bw_mean]

        # ============================================================
        # 4) bigWig promoter vs gene balance (initiation vs elongation)
        # ============================================================
        if tss_bw_mean in df and gene_bw_mean in df:
            df[f"{prefix}_{mark}_bw_promoter_gene_delta"] = df[tss_bw_mean] - df[gene_bw_mean]
            df[f"{prefix}_{mark}_bw_promoter_gene_ratio"] = safe_div(df[tss_bw_mean], df[gene_bw_mean])

        # ============================================================
        # 5) BED–bigWig cross interactions (structure × intensity)
        # ============================================================
        if tss_bed_mean in df and tss_bw_mean in df:
            df[f"{prefix}_{mark}_tss_bw_bed_interaction"] = df[tss_bed_mean] * df[tss_bw_mean]
        if gene_bed_mean in df and gene_bw_mean in df:
            df[f"{prefix}_{mark}_gene_bw_bed_interaction"] = df[gene_bed_mean] * df[gene_bw_mean]

    # ============================================================
    # Cleanup newly created numeric columns: inf→NaN→0.0
    # ============================================================
    new_cols = [c for c in df.columns if c not in prev_cols and df[c].dtype.kind in "fc"]
    df[new_cols] = df[new_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    print(f"✅ [{prefix}] Added {len(new_cols)} cross-layer features.")
    return df



def add_bed_topology_features(df, marks):
    """
    🧩 BED-based topology & spatial features (mark-aware).

    What this adds (per mark ∈ `marks`, region ∈ {gene, tss})
    ---------------------------------------------------------
    - `{prefix}compactness_index`              := coverage_ratio / num_peaks
        Why: Peak packing; compact high-coverage promoters often indicate focused regulation.
    - `{prefix}signal_density_product`         := signal_mean * peak_density
        Why: Joint strength × clustering; highlights strong, clustered activity.
    - `{prefix}entropy_density_ratio`          := peak_entropy / peak_density
        Why: Dispersion per density; separates diffuse vs organized peak landscapes.
    - `{prefix}signal_coefficient_of_variation`:= signal_std / |signal_mean|
        Why: Relative variability; unstable or bursty signals vs consistent ones.
    - `{prefix}avg_peak_dist_per_peak`         := closest_peak_to_TSS / num_peaks
        Why: Proximity normalization; closer, multiple peaks near TSS ↔ active promoters.

    Why this can improve accuracy
    -----------------------------
    These features summarize promoter/gene-body *organization* beyond raw intensity:
    compactness, clustering, dispersion, and proximity are all correlated with promoter
    activity and transcription, offering robust cues complementary to bigWig means.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing BED features across marks.
    marks : list[str]
        e.g., ["H3K27ac", "H3K4me3", "H3K9me3", "DNase"].

    Returns
    -------
    pd.DataFrame
        Input `df` augmented with topology features for each mark/region.
    """
    prev_cols = df.columns.copy()

    for mark in marks:
        for region in ["gene", "tss"]:
            # Dynamic column names
            prefix = f"{mark}_{region}_"
            num_peaks    = f"{prefix}num_peaks"
            cov_ratio    = f"{prefix}coverage_ratio"
            peak_dens    = f"{prefix}peak_density"
            peak_entropy = f"{prefix}peak_entropy"
            signal_mean  = f"{prefix}signal_mean"
            signal_std   = f"{prefix}signal_std"
            closest_peak = f"{prefix}closest_peak_to_TSS"

            # 1) Compactness & density
            if all(c in df.columns for c in (num_peaks, cov_ratio)):
                df[f"{prefix}compactness_index"] = safe_div(df[cov_ratio], df[num_peaks])
            if all(c in df.columns for c in (signal_mean, peak_dens)):
                df[f"{prefix}signal_density_product"] = df[signal_mean] * df[peak_dens]

            # 2) Entropy–density ratio
            if all(c in df.columns for c in (peak_entropy, peak_dens)):
                df[f"{prefix}entropy_density_ratio"] = safe_div(df[peak_entropy], df[peak_dens])

            # 3) Coefficient of variation (relative instability)
            if all(c in df.columns for c in (signal_mean, signal_std)):
                df[f"{prefix}signal_coefficient_of_variation"] = safe_div(
                    df[signal_std], np.abs(df[signal_mean]) + 1e-8
                )

            # 4) Average TSS proximity per peak (normalized by count)
            if all(c in df.columns for c in (closest_peak, num_peaks)):
                df[f"{prefix}avg_peak_dist_per_peak"] = safe_div(df[closest_peak], df[num_peaks])

    return check_new_columns(df, prev_cols, "bed_topology_features_v2")


# ============================================================
#                MAIN PIPELINE FUNCTION
# ============================================================


def run_feature_engineering(merged_dir, cells, marks):
    activating_marks = ["DNase", "H3K27ac", "H3K4me3", "H3K36me3"]
    repressive_marks = ["H3K9me3", "H3K27me3"]

    for cell in cells:
        in_path = os.path.join(merged_dir, f"{cell}_all_logzscore_logzscore.tsv")
        if not os.path.exists(in_path):
            print(f"⚠️ Missing input file: {in_path}")
            continue

        print(f"\n📂 Processing {cell} ...")
        df = pd.read_csv(in_path, sep="\t")
        df = add_gene_structure(df)
        df = add_tss_distance_feature(df)   # 🆕 新增這一行
        df = add_promoter_gene_ratio(df, activating_marks + repressive_marks)
        df = add_activation_balance(df)
        df = add_promoter_entropy(df, activating_marks, repressive_marks)
        df = add_chromatin_indices(df, activating_marks, repressive_marks)
        df = add_strand_features(df)
        df = add_cross_mark_interactions(df, marks)
        df = add_activation_repression_indices(df)
        df = add_axis_and_delta(df, marks)
        df = rank_transform_features(df)
        df = add_bed_topology_features(df, marks)
        df = add_advanced_chromatin_features(df, marks)   # 🧠 加在這裡
        df = add_cross_layer_features(df, marks=marks, prefix="cross", bw_norm="logz")

        out_path = os.path.join(merged_dir, f"{cell}_all_rank_features.tsv")
        df.to_csv(out_path, sep="\t", index=False)
        print(f"✅ Saved engineered features → {out_path}")

    print("\n🎯 Feature engineering complete for all cell lines.")
    return df

# ============================================================
#                EXECUTION EXAMPLE
# ============================================================

if __name__ == "__main__":
    merged_dir = "../preprocessed_data/reference/1. merged data/without_y_100_one_side/"
    cells = ["X1", "X2", "X3"]
    META_COLS = ["gene_name", "chr", "gene_start", "gene_end",
             "TSS_start", "TSS_end", "strand", "gex", "gex_rank"]
    marks = ["DNase", "H3K27ac", "H3K4me3", "H3K27me3", "H3K36me3", "H3K4me1", "H3K9me3"]
    df = run_feature_engineering(merged_dir, cells, marks)
