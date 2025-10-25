
# === 輔助函式 ===
import numpy as np
from scipy.stats import kurtosis, skew
from numpy.fft import fft


# ============================================================
# ⚙️ 通用 Normalization Function
# ============================================================
def normalize_signal(vals, global_mean=None, global_std=None, mode="none"):
    """
    多模式 normalization 支援：
      - "none": 不做任何處理
      - "zscore": (x - μ) / σ
      - "log_zscore": ((log1p(x)) - μ_log) / σ_log
      - "log_only": 只取 log1p，不標準化
    """
    if mode == "none":
        return vals

    elif mode == "zscore":
        if global_mean is None or global_std is None:
            raise ValueError("Missing global_mean/global_std for zscore normalization.")
        return (vals - global_mean) / (global_std + 1e-8)

    elif mode == "log_zscore":
        if global_mean is None or global_std is None:
            raise ValueError("Missing global_mean/global_std for log_zscore normalization.")
        vals = np.log1p(np.clip(vals, a_min=0, a_max=None))
        return (vals - global_mean) / (global_std + 1e-8)

    elif mode == "log_only":
        return np.log1p(np.clip(vals, a_min=0, a_max=None))

    else:
        raise ValueError(f"Unknown normalization mode: {mode}")


# ============================================================
# 🧬 region_zsignal (with normalization selection)
# ============================================================
def region_zsignal(bw, chrom, start, end, global_mean=None, global_std=None,
                   mark_name=None, cell_name=None, norm_mode="none"):
    """
    Extracts region-level features with multiple normalization modes.
    """
    feature_keys = [
        "mean", "std", "min", "max", "diff",
        "gradient_mean", "slope", "kurtosis", "skewness",
        "entropy", "autocorr", "laplacian"
    ]

    chroms = bw.chroms()
    if chrom not in chroms:
        return {k: 0.0 for k in feature_keys}

    chrom_length = chroms[chrom]
    start, end = max(0, int(start)), min(int(end), chrom_length)
    if end <= start:
        return {k: 0.0 for k in feature_keys}

    vals = np.array(bw.values(chrom, start, end, numpy=True))
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return {k: 0.0 for k in feature_keys}

    # === 🔧 Apply normalization (可動態切換) ===
    vals_norm = normalize_signal(vals, global_mean, global_std, mode=norm_mode)

    # === Feature Extraction ===
    local_mean = np.mean(vals_norm)
    local_std  = np.std(vals_norm)
    local_min  = np.min(vals_norm)
    local_max  = np.max(vals_norm)
    local_diff = local_max - local_min

    # Gradient / slope
    if len(vals_norm) > 1:
        diffs = np.diff(vals_norm)
        gradient_mean = np.mean(np.abs(diffs))
        try:
            slope = np.polyfit(np.arange(len(vals_norm)), vals_norm, 1)[0]
        except Exception:
            slope = 0.0
    else:
        gradient_mean, slope = 0.0, 0.0

    # Shape-based descriptors
    sharpness = kurtosis(vals_norm) if len(vals_norm) > 3 else 0.0
    asymmetry = skew(vals_norm) if len(vals_norm) > 3 else 0.0

    # Entropy
    p = np.abs(vals_norm)
    p_sum = np.sum(p)
    local_entropy = -np.sum((p / (p_sum + 1e-8)) * np.log2(p / (p_sum + 1e-8))) if p_sum > 0 else 0.0

    # Autocorrelation & Laplacian
    autocorr = np.corrcoef(vals_norm[:-1], vals_norm[1:])[0, 1] if len(vals_norm) > 2 else 0.0
    laplacian = np.mean(np.abs(vals_norm[:-2] - 2 * vals_norm[1:-1] + vals_norm[2:])) if len(vals_norm) > 3 else 0.0

    result = {
        "mean": local_mean,
        "std": local_std,
        "min": local_min,
        "max": local_max,
        "diff": local_diff,
        "gradient_mean": gradient_mean,
        "slope": slope,
        "kurtosis": sharpness,
        "skewness": asymmetry,
        "entropy": local_entropy,
        "autocorr": autocorr,
        "laplacian": laplacian,
    }

    # 保險處理 nan/inf
    for k, v in result.items():
        if not np.isfinite(v):
            result[k] = 0.0

    return result


def get_tss_region(row, window=1000):
    """根據 strand (+/-) 給出 promoter (TSS ± window) 區域"""
    if row["strand"] == "+":
        start = max(0, row["TSS_start"] - window)
        end = row["TSS_end"]
    else:  # strand == "-"
        start = row["TSS_start"]
        end = row["TSS_end"] + window
    return start, end
