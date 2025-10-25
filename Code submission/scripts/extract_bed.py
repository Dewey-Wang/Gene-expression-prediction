import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict
import os

# ============================================================
# ⚙️ Normalize utility
# ============================================================
def normalize_array(x: np.ndarray, mode: str = "none", global_mean=None, global_std=None) -> np.ndarray:
    """多模式 normalization"""
    if len(x) == 0:
        return x
    if mode == "none":
        return x
    elif mode == "zscore":
        if global_mean is None or global_std is None:
            raise ValueError("Missing global mean/std for zscore.")
        return (x - global_mean) / (global_std + 1e-8)
    elif mode == "log_zscore":
        x = np.log1p(np.clip(x, a_min=0, a_max=None))
        if global_mean is None or global_std is None:
            global_mean, global_std = np.mean(x), np.std(x)
        return (x - global_mean) / (global_std + 1e-8)
    else:
        raise ValueError(f"Unsupported normalization mode: {mode}")


# ============================================================
# 📏 Helper functions
# ============================================================
@dataclass
class Interval:
    start: int
    end: int

def union_length(intervals: List[Interval]) -> int:
    if not intervals:
        return 0
    intervals.sort(key=lambda iv: (iv.start, iv.end))
    total = 0
    cs, ce = intervals[0].start, intervals[0].end
    for iv in intervals[1:]:
        if iv.start > ce:
            total += ce - cs
            cs, ce = iv.start, iv.end
        elif iv.end > ce:
            ce = iv.end
    total += ce - cs
    return total

def get_tss_region(row, window=1000):
    """根據 strand (+/-) 給出 promoter (TSS ± window) 區域"""
    if row["strand"] == "+":
        start = max(0, row["TSS_start"] - window)
        end = row["TSS_end"]
    else:  # strand == "-"
        start = row["TSS_start"]
        end = row["TSS_end"] + window
    return start, end

def gene_region(row):
    return int(row["gene_start"]), int(row["gene_end"])

def select_mask(peaks_chr: pd.DataFrame, start: int, end: int) -> pd.Series:
    return (peaks_chr["summit"] >= start) & (peaks_chr["summit"] < end)

def shannon_entropy_from_weights(weights: np.ndarray) -> float:
    s = float(weights.sum())
    if s <= 0:
        return 0.0
    p = weights / s
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def spacing_std_from_positions(pos: np.ndarray) -> float:
    if pos.size < 2:
        return 0.0
    diffs = np.diff(np.sort(pos))
    return float(diffs.std(ddof=0))

# ============================================================
# 🧬 主函式
# ============================================================
def compute_features(df_genes: pd.DataFrame,
                     peaks_by_chr: Dict[str, pd.DataFrame],
                     region: str = "gene",
                     tss_window_bp: int = 5000,
                     norm_mode: str = "none") -> pd.DataFrame:
    assert region in {"gene", "tss"}
    rows = []
    peak_densities = []  # 🆕 收集所有 density

    # === 統一 global mean/std ===
    all_signals = np.concatenate([v["signal_value"].to_numpy() for v in peaks_by_chr.values()])
    global_mean = np.mean(all_signals)
    global_std = np.std(all_signals)
    print(f"🌍 Global mean/std (signal): {global_mean:.4f}, {global_std:.4f}")

    for _, g in tqdm(df_genes.iterrows(), total=len(df_genes), desc=f"Computing {region} features"):
        chrom = g["chr"]
        if chrom not in peaks_by_chr:
            rows.append({"gene_name": g["gene_name"], "has_peak": 0})
            continue

        start, end = (gene_region(g) if region == "gene" else get_tss_region(g, window=tss_window_bp))
        peaks_chr = peaks_by_chr[chrom]
        sub = peaks_chr.loc[select_mask(peaks_chr, start, end)]
        if sub.empty:
            rows.append({"gene_name": g["gene_name"], "has_peak": 0})
            continue

        s_val = sub["signal_value"].to_numpy()
        w = sub["width"].to_numpy()
        qv = sub["q_value"].to_numpy()
        sc = sub["score"].to_numpy()
        summits = sub["summit"].to_numpy()

        # === Normalize signal_value ===
        s_val_norm = normalize_array(s_val, mode=norm_mode, global_mean=global_mean, global_std=global_std)

        region_len = max(1, end - start)
        ivs = [Interval(int(s), int(e)) for s, e in zip(sub["start"], sub["end"])]
        cov = union_length(ivs) / region_len
        n = len(sub)
        peak_density = n / region_len
        peak_densities.append(peak_density)

        # === 統計計算統一函式 ===
        def stat_block(arr):
            if arr.size == 0:
                return dict(sum=0.0, mean=0.0, std=0.0, min=0.0, max=0.0, diff=0.0)
            return dict(
                sum=float(np.sum(arr)),
                mean=float(np.mean(arr)),
                std=float(np.std(arr, ddof=0)),
                min=float(np.min(arr)),
                max=float(np.max(arr)),
                diff=float(np.max(arr) - np.min(arr))
            )

        stat_signal = stat_block(s_val_norm)
        stat_width  = stat_block(w)
        stat_qvalue = stat_block(qv)
        stat_score  = stat_block(sc)

        rows.append({
            "gene_name": g["gene_name"],
            "has_peak": 1,
            "num_peaks": n,
            "peak_density": peak_density,  # 🆕 暫時未 normalize
            **{f"signal_{k}": v for k, v in stat_signal.items()},
            **{f"width_{k}": v for k, v in stat_width.items()},
            **{f"qvalue_{k}": v for k, v in stat_qvalue.items()},
            **{f"score_{k}": v for k, v in stat_score.items()},
            "coverage_ratio": cov,
            "peak_entropy": shannon_entropy_from_weights(s_val_norm),
            "spacing_std": spacing_std_from_positions(summits),
            "width_entropy": float(-np.sum((w / w.sum()) * np.log2((w / w.sum()) + 1e-8))) if w.sum() > 0 else 0.0,
            "compactness": cov / n if n > 0 else 0.0,
            "peak_pos_entropy": (
                lambda hist: float(-np.sum(hist * np.log2(hist + 1e-8)))
            )(np.histogram(np.clip((summits - start) / region_len, 0, 1),
                           bins=5, range=(0, 1))[0] / max(1, n))
            if region_len > 0 and len(summits) > 0 else 0.0,
            "closest_peak_to_TSS": float(np.abs(summits - ((g["TSS_start"] + g["TSS_end"]) / 2)).min()),
            "directional_peak_bias": (
                (np.sum(summits > ((g["TSS_start"] + g["TSS_end"]) / 2)) -
                 np.sum(summits < ((g["TSS_start"] + g["TSS_end"]) / 2))) / n if n > 0 else 0.0
            ),
            "signal_weighted_dist_to_TSS": (
                float(np.sum(s_val_norm * np.abs(summits - ((g["TSS_start"] + g["TSS_end"]) / 2)))) / np.sum(s_val_norm)
                if np.sum(s_val_norm) > 0 else 0.0
            ),
        })

    # === 組裝結果表 ===
    df = pd.DataFrame(rows)

    # === 🧮 peak_density 做 0–1 normalization ===
    if "peak_density" in df.columns:
        min_d, max_d = np.nanmin(peak_densities), np.nanmax(peak_densities)
        if max_d > min_d:
            df["peak_density"] = (df["peak_density"] - min_d) / (max_d - min_d)
        else:
            df["peak_density"] = 0.0

    # === 🧹 把所有 NaN 補 0 ===
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0.0)
    df = df.applymap(lambda x: 0.0 if isinstance(x, float) and not np.isfinite(x) else x)

    return df



# ============================================================
#                  Batch integrate multiple marks / cell lines
# ============================================================
def process_cellline(
    cell_name: str,
    marks: List[str],
    base_dir: str,
    ref_path: str,
    tss_window_bp: int = 5000,
    output_dir: str = "../preprocessed_data/reference/0. raw_bed",
    norm_mode: str = "log_zscore",
):
    """
    Process all histone marks for a given cell line (e.g., X1) using a common
    reference table. Supports normalization modes: 'none' / 'zscore' / 'log_zscore'.

    Why this structure:
    - Single pass per cell line to keep feature namespace consistent across marks.
    - Column prefixing prevents collisions when merging per-mark features.
    """
    print(f"\n=== 🧬 Processing Cell Line: {cell_name} ===")
    print(f"Normalization mode: {norm_mode}")

    # --- Load reference table (genes + coordinates) ---
    df_genes = pd.read_csv(ref_path, sep="\t")
    print(f"📖 Loaded reference table: {len(df_genes)} genes")

    # --- Initialize the output DataFrame with core gene metadata ---
    df_all = df_genes[["gene_name", "chr", "strand", "gene_start", "gene_end", "TSS_start", "TSS_end"]].copy()

    # --- Iterate over marks and accumulate features ---
    for mark in marks:
        bed_path = os.path.join(base_dir, f"{mark}-bed", f"{cell_name}.bed")
        if not os.path.exists(bed_path):
            print(f"⚠️ Missing {mark} for {cell_name}: {bed_path}")
            continue

        print(f"\n📘 Reading {mark} ({cell_name}) ...")
        # Use explicit dtypes for reproducibility; large genomic coordinates need int64
        df_peaks = pd.read_csv(
            bed_path,
            sep="\t",
            header=None,
            names=[
                "chr",
                "start",
                "end",
                "name",
                "score",
                "strand",
                "signal_value",
                "p_value",
                "q_value",
                "peak",
            ],
            dtype={
                "chr": str,
                "start": np.int64,
                "end": np.int64,
                "name": str,
                "score": float,
                "strand": str,
                "signal_value": float,
                "p_value": float,
                "q_value": float,
                "peak": np.int64,
            },
        )

        # --- Derive summit/width; group by chromosome (why: faster lookup by chr) ---
        df_peaks["summit"] = df_peaks["start"] + df_peaks["peak"]
        df_peaks["width"] = df_peaks["end"] - df_peaks["start"]
        peaks_by_chr = {c: sub.reset_index(drop=True) for c, sub in df_peaks.groupby("chr", sort=False)}
        print(f"📊 Loaded {len(df_peaks)} peaks, grouped into {len(peaks_by_chr)} chromosomes")

        # --- Compute features near gene bodies and TSS windows ---
        print(f"⚙️ Computing features for {mark} ({cell_name}) ...")
        # compute_features is expected to consume df_genes and per-chr peak dict
        feat_gene = compute_features(
            df_genes, peaks_by_chr, region="gene", tss_window_bp=tss_window_bp, norm_mode=norm_mode
        )
        feat_tss = compute_features(
            df_genes, peaks_by_chr, region="tss", tss_window_bp=tss_window_bp, norm_mode=norm_mode
        )

        # --- Prefix columns to avoid name clashes after merges ---
        feat_gene = feat_gene.rename(columns={c: f"{mark}_gene_{c}" for c in feat_gene.columns if c != "gene_name"})
        feat_tss = feat_tss.rename(columns={c: f"{mark}_tss_{c}" for c in feat_tss.columns if c != "gene_name"})

        # --- Merge per-mark features into the master table on gene_name ---
        df_all = df_all.merge(feat_gene, on="gene_name", how="left")
        df_all = df_all.merge(feat_tss, on="gene_name", how="left")

    # --- Fill missing features with zeros (why: absent marks → neutral contribution) ---
    df_all = df_all.fillna(0.0)

    # --- Build output filename suffix based on normalization mode ---
    norm_suffix = {"none": "raw", "zscore": "zscore", "log_zscore": "logzscore"}.get(norm_mode, norm_mode)

    # --- Persist results ---
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{cell_name}_{norm_suffix}.tsv")
    df_all.to_csv(out_path, sep="\t", index=False)

    print(f"\n✅ Saved: {out_path}")
    print(f"📊 Final shape: {df_all.shape[0]} genes × {df_all.shape[1]} columns")
    return df_all
