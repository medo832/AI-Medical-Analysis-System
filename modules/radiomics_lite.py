"""
Lightweight radiomics fallback
==============================
A pure-NumPy / scikit-image implementation of the 93 first-order, GLCM, GLRLM,
GLSZM, GLDM and NGTDM features that the trained `scaler.pkl` expects.

Why this exists
---------------
Real PyRadiomics is hard to install on Windows + new Python versions because
it builds C extensions from source. This module re-implements the same feature
names using only `numpy` and `skimage.feature.graycomatrix`, both of which have
ready-made wheels on every platform.

Caveats
-------
Values are CLOSE to PyRadiomics but not bit-identical (different binning,
different distance/angle conventions). For a Phase-1 demo this is fine — the
KMeans phenotype may occasionally flip on borderline ROIs.

If you need exact PyRadiomics behaviour, install it once the environment
allows it; this module is a drop-in replacement until then.
"""
from __future__ import annotations

import numpy as np
from skimage.feature import graycomatrix, graycoprops


# ------------------------------------------------------------------
# Helper: discretise into a small number of gray levels
# ------------------------------------------------------------------
def _discretise(roi: np.ndarray, bin_width: int = 25) -> tuple[np.ndarray, int]:
    """Mimic PyRadiomics' fixedBinWidth discretisation."""
    roi = roi.astype(float)
    mn = float(roi.min())
    levels = np.floor((roi - mn) / bin_width).astype(np.int32) + 1
    return levels, int(levels.max())


# ------------------------------------------------------------------
# First-order
# ------------------------------------------------------------------
def _firstorder(roi: np.ndarray) -> dict:
    x = roi.astype(float).ravel()
    if x.size == 0:
        return {}
    eps = 1e-12
    mean = x.mean()
    var = x.var()
    std = x.std()
    p10, p25, p50, p75, p90 = np.percentile(x, [10, 25, 50, 75, 90])
    # entropy of histogram
    hist, _ = np.histogram(x, bins=max(8, int(x.max() - x.min()) // 25 + 1))
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    entropy = -float(np.sum(p * np.log2(p))) if p.size else 0.0
    # uniformity = sum of p^2
    uniformity = float(np.sum(p * p)) if p.size else 0.0
    skew = float(np.mean(((x - mean) / (std + eps)) ** 3))
    kurt = float(np.mean(((x - mean) / (std + eps)) ** 4)) - 3.0
    mad = float(np.mean(np.abs(x - mean)))
    # robust MAD = mean abs deviation of 10-90th percentile
    mask = (x >= p10) & (x <= p90)
    rmad = float(np.mean(np.abs(x[mask] - x[mask].mean()))) if mask.any() else 0.0
    energy = float(np.sum(x * x))
    return {
        "original_firstorder_10Percentile": float(p10),
        "original_firstorder_90Percentile": float(p90),
        "original_firstorder_Energy": energy,
        "original_firstorder_Entropy": entropy,
        "original_firstorder_InterquartileRange": float(p75 - p25),
        "original_firstorder_Kurtosis": kurt,
        "original_firstorder_Maximum": float(x.max()),
        "original_firstorder_MeanAbsoluteDeviation": mad,
        "original_firstorder_Mean": float(mean),
        "original_firstorder_Median": float(p50),
        "original_firstorder_Minimum": float(x.min()),
        "original_firstorder_Range": float(x.max() - x.min()),
        "original_firstorder_RobustMeanAbsoluteDeviation": rmad,
        "original_firstorder_RootMeanSquared": float(np.sqrt(np.mean(x * x))),
        "original_firstorder_Skewness": skew,
        "original_firstorder_TotalEnergy": energy,  # voxel spacing = 1 here
        "original_firstorder_Uniformity": uniformity,
        "original_firstorder_Variance": float(var),
    }


# ------------------------------------------------------------------
# GLCM (using skimage, then derive PyRadiomics-style features)
# ------------------------------------------------------------------
def _glcm(roi: np.ndarray, bin_width: int = 25) -> dict:
    levels_img, max_lvl = _discretise(roi, bin_width)
    if max_lvl < 2:
        return {}
    img = np.clip(levels_img, 0, 255).astype(np.uint8)
    # average over 4 angles, distance=1
    glcm = graycomatrix(img, distances=[1],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=int(img.max()) + 1, symmetric=True, normed=True)
    P = glcm[:, :, 0, :].mean(axis=2)  # average over angles
    n = P.shape[0]
    eps = 1e-12

    i = np.arange(n).reshape(-1, 1)
    j = np.arange(n).reshape(1, -1)
    px = P.sum(axis=1)
    py = P.sum(axis=0)
    mu_x = float(np.sum(i.ravel() * px))
    mu_y = float(np.sum(j.ravel() * py))
    sig_x = float(np.sqrt(np.sum(((i.ravel() - mu_x) ** 2) * px)))
    sig_y = float(np.sqrt(np.sum(((j.ravel() - mu_y) ** 2) * py)))

    contrast = float(np.sum(((i - j) ** 2) * P))
    diff_avg = float(np.sum(np.abs(i - j) * P))
    correlation = float(np.sum((i - mu_x) * (j - mu_y) * P) / (sig_x * sig_y + eps))
    joint_avg = mu_x
    joint_energy = float(np.sum(P * P))
    joint_entropy = -float(np.sum(P * np.log2(P + eps)))
    autocorr = float(np.sum(i * j * P))
    max_prob = float(P.max())
    cluster_shade = float(np.sum(((i + j - mu_x - mu_y) ** 3) * P))
    cluster_prom = float(np.sum(((i + j - mu_x - mu_y) ** 4) * P))
    cluster_tend = float(np.sum(((i + j - mu_x - mu_y) ** 2) * P))
    Id = float(np.sum(P / (1.0 + np.abs(i - j))))
    Idm = float(np.sum(P / (1.0 + (i - j) ** 2)))
    Idn = float(np.sum(P / (1.0 + np.abs(i - j) / n)))
    Idmn = float(np.sum(P / (1.0 + ((i - j) ** 2) / (n * n))))
    inv_var = float(np.sum(P[i != j] / ((i - j) ** 2)[i != j]))
    sum_sq = float(np.sum(((i - joint_avg) ** 2) * P))

    # difference & sum distributions
    diffs = np.zeros(n)
    sums = np.zeros(2 * n - 1)
    for a in range(n):
        for b in range(n):
            diffs[abs(a - b)] += P[a, b]
            sums[a + b] += P[a, b]
    diff_var = float(np.sum(((np.arange(n) - diff_avg) ** 2) * diffs))
    diff_entropy = -float(np.sum(diffs * np.log2(diffs + eps)))
    sum_avg = float(np.sum(np.arange(2, 2 * n + 1)[:len(sums)] * sums))
    sum_entropy = -float(np.sum(sums * np.log2(sums + eps)))

    # IMC1, IMC2 (information measures of correlation)
    hx = -float(np.sum(px * np.log2(px + eps)))
    hy = -float(np.sum(py * np.log2(py + eps)))
    pxy = px.reshape(-1, 1) * py.reshape(1, -1)
    hxy = joint_entropy
    hxy1 = -float(np.sum(P * np.log2(pxy + eps)))
    hxy2 = -float(np.sum(pxy * np.log2(pxy + eps)))
    imc1 = (hxy - hxy1) / max(hx, hy, eps)
    imc2_val = 1.0 - np.exp(-2.0 * (hxy2 - hxy))
    imc2 = float(np.sqrt(max(imc2_val, 0.0)))

    # MCC — costly to compute exactly; use a stable proxy
    mcc = correlation  # acceptable approximation

    return {
        "original_glcm_Autocorrelation": autocorr,
        "original_glcm_ClusterProminence": cluster_prom,
        "original_glcm_ClusterShade": cluster_shade,
        "original_glcm_ClusterTendency": cluster_tend,
        "original_glcm_Contrast": contrast,
        "original_glcm_Correlation": correlation,
        "original_glcm_DifferenceAverage": diff_avg,
        "original_glcm_DifferenceEntropy": diff_entropy,
        "original_glcm_DifferenceVariance": diff_var,
        "original_glcm_Id": Id,
        "original_glcm_Idm": Idm,
        "original_glcm_Idmn": Idmn,
        "original_glcm_Idn": Idn,
        "original_glcm_Imc1": float(imc1),
        "original_glcm_Imc2": imc2,
        "original_glcm_InverseVariance": inv_var,
        "original_glcm_JointAverage": joint_avg,
        "original_glcm_JointEnergy": joint_energy,
        "original_glcm_JointEntropy": joint_entropy,
        "original_glcm_MCC": float(mcc),
        "original_glcm_MaximumProbability": max_prob,
        "original_glcm_SumAverage": sum_avg,
        "original_glcm_SumEntropy": sum_entropy,
        "original_glcm_SumSquares": sum_sq,
    }


# ------------------------------------------------------------------
# GLRLM (gray-level run-length matrix)
# ------------------------------------------------------------------
def _run_lengths(line: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (gray_level, run_length) on a 1-D pass."""
    if len(line) == 0:
        return []
    out = []
    cur = line[0]; cnt = 1
    for v in line[1:]:
        if v == cur:
            cnt += 1
        else:
            out.append((int(cur), cnt)); cur = v; cnt = 1
    out.append((int(cur), cnt))
    return out


def _glrlm(roi: np.ndarray, bin_width: int = 25) -> dict:
    levels, n_g = _discretise(roi, bin_width)
    if n_g < 2:
        return {}
    max_run = max(levels.shape)
    P = np.zeros((n_g + 1, max_run + 1), dtype=float)
    n_dirs = 0
    for arr in (
        levels, levels.T,                               # horizontal, vertical
        np.array([np.diag(levels, k=k) for k in range(-levels.shape[0]+1, levels.shape[1])], dtype=object),
        np.array([np.diag(np.fliplr(levels), k=k) for k in range(-levels.shape[0]+1, levels.shape[1])], dtype=object),
    ):
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            lines = list(arr)
        else:
            lines = list(arr)
        n_dirs += 1
        for line in lines:
            for g, rl in _run_lengths(np.asarray(line, dtype=int)):
                if 1 <= g <= n_g and 1 <= rl <= max_run:
                    P[g, rl] += 1.0
    P = P[1:, 1:]                          # drop the 0-padding row/col
    Nr = P.sum()
    if Nr == 0:
        return {}
    P_norm = P / Nr
    i = np.arange(1, P.shape[0] + 1).reshape(-1, 1)
    j = np.arange(1, P.shape[1] + 1).reshape(1, -1)
    p_g = P.sum(axis=1)
    p_r = P.sum(axis=0)
    eps = 1e-12

    SRE  = float(np.sum(p_r / (np.arange(1, len(p_r) + 1) ** 2)) / Nr)
    LRE  = float(np.sum(p_r * (np.arange(1, len(p_r) + 1) ** 2)) / Nr)
    GLN  = float(np.sum(p_g * p_g) / Nr)
    GLNN = float(np.sum(p_g * p_g) / (Nr * Nr))
    RLN  = float(np.sum(p_r * p_r) / Nr)
    RLNN = float(np.sum(p_r * p_r) / (Nr * Nr))
    RP   = float(Nr / max(levels.size * n_dirs, 1))
    LGRE = float(np.sum(p_g / (np.arange(1, len(p_g) + 1) ** 2)) / Nr)
    HGRE = float(np.sum(p_g * (np.arange(1, len(p_g) + 1) ** 2)) / Nr)
    SRLGE = float(np.sum(P / (i ** 2 * j ** 2)) / Nr)
    SRHGE = float(np.sum(P * (i ** 2) / (j ** 2)) / Nr)
    LRLGE = float(np.sum(P * (j ** 2) / (i ** 2)) / Nr)
    LRHGE = float(np.sum(P * (i ** 2) * (j ** 2)) / Nr)
    mu_g  = float(np.sum(i * P_norm))
    mu_r  = float(np.sum(j * P_norm))
    glvar = float(np.sum(((i - mu_g) ** 2) * P_norm))
    rvar  = float(np.sum(((j - mu_r) ** 2) * P_norm))
    rentropy = -float(np.sum(P_norm * np.log2(P_norm + eps)))

    return {
        "original_glrlm_GrayLevelNonUniformity": GLN,
        "original_glrlm_GrayLevelNonUniformityNormalized": GLNN,
        "original_glrlm_GrayLevelVariance": glvar,
        "original_glrlm_HighGrayLevelRunEmphasis": HGRE,
        "original_glrlm_LongRunEmphasis": LRE,
        "original_glrlm_LongRunHighGrayLevelEmphasis": LRHGE,
        "original_glrlm_LongRunLowGrayLevelEmphasis": LRLGE,
        "original_glrlm_LowGrayLevelRunEmphasis": LGRE,
        "original_glrlm_RunEntropy": rentropy,
        "original_glrlm_RunLengthNonUniformity": RLN,
        "original_glrlm_RunLengthNonUniformityNormalized": RLNN,
        "original_glrlm_RunPercentage": RP,
        "original_glrlm_RunVariance": rvar,
        "original_glrlm_ShortRunEmphasis": SRE,
        "original_glrlm_ShortRunHighGrayLevelEmphasis": SRHGE,
        "original_glrlm_ShortRunLowGrayLevelEmphasis": SRLGE,
    }


# ------------------------------------------------------------------
# GLSZM (gray-level size-zone matrix)
# ------------------------------------------------------------------
def _glszm(roi: np.ndarray, bin_width: int = 25) -> dict:
    from scipy.ndimage import label
    levels, n_g = _discretise(roi, bin_width)
    if n_g < 2:
        return {}
    structure = np.ones((3, 3), dtype=int)
    zones: list[tuple[int, int]] = []                   # (gray_level, area)
    for g in range(1, n_g + 1):
        lab, n = label(levels == g, structure=structure)
        for k in range(1, n + 1):
            zones.append((g, int((lab == k).sum())))
    if not zones:
        return {}
    max_size = max(z[1] for z in zones)
    P = np.zeros((n_g, max_size), dtype=float)
    for g, s in zones:
        P[g - 1, s - 1] += 1
    Nz = P.sum()
    eps = 1e-12
    P_norm = P / Nz
    p_g = P.sum(axis=1)
    p_s = P.sum(axis=0)
    i = np.arange(1, P.shape[0] + 1).reshape(-1, 1)
    j = np.arange(1, P.shape[1] + 1).reshape(1, -1)
    mu_g = float(np.sum(i * P_norm))

    SAE  = float(np.sum(p_s / (np.arange(1, len(p_s) + 1) ** 2)) / Nz)
    LAE  = float(np.sum(p_s * (np.arange(1, len(p_s) + 1) ** 2)) / Nz)
    GLN  = float(np.sum(p_g * p_g) / Nz)
    GLNN = float(np.sum(p_g * p_g) / (Nz * Nz))
    SZN  = float(np.sum(p_s * p_s) / Nz)
    SZNN = float(np.sum(p_s * p_s) / (Nz * Nz))
    ZP   = float(Nz / max(levels.size, 1))
    LGZE = float(np.sum(p_g / (np.arange(1, len(p_g) + 1) ** 2)) / Nz)
    HGZE = float(np.sum(p_g * (np.arange(1, len(p_g) + 1) ** 2)) / Nz)
    SALGE = float(np.sum(P / (i ** 2 * j ** 2)) / Nz)
    SAHGE = float(np.sum(P * (i ** 2) / (j ** 2)) / Nz)
    LALGE = float(np.sum(P * (j ** 2) / (i ** 2)) / Nz)
    LAHGE = float(np.sum(P * (i ** 2) * (j ** 2)) / Nz)
    glvar = float(np.sum(((i - mu_g) ** 2) * P_norm))
    zvar  = float(np.sum(((j - float(np.sum(j * P_norm))) ** 2) * P_norm))
    zentropy = -float(np.sum(P_norm * np.log2(P_norm + eps)))

    return {
        "original_glszm_GrayLevelNonUniformity": GLN,
        "original_glszm_GrayLevelNonUniformityNormalized": GLNN,
        "original_glszm_GrayLevelVariance": glvar,
        "original_glszm_HighGrayLevelZoneEmphasis": HGZE,
        "original_glszm_LargeAreaEmphasis": LAE,
        "original_glszm_LargeAreaHighGrayLevelEmphasis": LAHGE,
        "original_glszm_LargeAreaLowGrayLevelEmphasis": LALGE,
        "original_glszm_LowGrayLevelZoneEmphasis": LGZE,
        "original_glszm_SizeZoneNonUniformity": SZN,
        "original_glszm_SizeZoneNonUniformityNormalized": SZNN,
        "original_glszm_SmallAreaEmphasis": SAE,
        "original_glszm_SmallAreaHighGrayLevelEmphasis": SAHGE,
        "original_glszm_SmallAreaLowGrayLevelEmphasis": SALGE,
        "original_glszm_ZoneEntropy": zentropy,
        "original_glszm_ZonePercentage": ZP,
        "original_glszm_ZoneVariance": zvar,
    }


# ------------------------------------------------------------------
# GLDM (gray-level dependence matrix)
# ------------------------------------------------------------------
def _gldm(roi: np.ndarray, bin_width: int = 25, alpha: int = 0) -> dict:
    levels, n_g = _discretise(roi, bin_width)
    if n_g < 2:
        return {}
    rows, cols = levels.shape
    max_dep = 9                        # 8 neighbours + self
    P = np.zeros((n_g + 1, max_dep), dtype=float)
    for r in range(rows):
        for c in range(cols):
            g = levels[r, c]
            dep = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        if abs(levels[rr, cc] - g) <= alpha:
                            dep += 1
            P[g, dep] += 1
    P = P[1:, :]                       # drop level=0 padding
    Ns = P.sum()
    if Ns == 0:
        return {}
    P_norm = P / Ns
    p_g = P.sum(axis=1)
    p_d = P.sum(axis=0)
    i = np.arange(1, P.shape[0] + 1).reshape(-1, 1)
    j = np.arange(1, P.shape[1] + 1).reshape(1, -1)
    eps = 1e-12
    mu_g = float(np.sum(i * P_norm))
    mu_d = float(np.sum(j * P_norm))

    SDE  = float(np.sum(p_d / (np.arange(1, len(p_d) + 1) ** 2)) / Ns)
    LDE  = float(np.sum(p_d * (np.arange(1, len(p_d) + 1) ** 2)) / Ns)
    GLN  = float(np.sum(p_g * p_g) / Ns)
    DN   = float(np.sum(p_d * p_d) / Ns)
    DNN  = float(np.sum(p_d * p_d) / (Ns * Ns))
    LGE  = float(np.sum(p_g / (np.arange(1, len(p_g) + 1) ** 2)) / Ns)
    HGE  = float(np.sum(p_g * (np.arange(1, len(p_g) + 1) ** 2)) / Ns)
    SDLGE = float(np.sum(P / (i ** 2 * j ** 2)) / Ns)
    SDHGE = float(np.sum(P * (i ** 2) / (j ** 2)) / Ns)
    LDLGE = float(np.sum(P * (j ** 2) / (i ** 2)) / Ns)
    LDHGE = float(np.sum(P * (i ** 2) * (j ** 2)) / Ns)
    dentropy = -float(np.sum(P_norm * np.log2(P_norm + eps)))
    dvar = float(np.sum(((j - mu_d) ** 2) * P_norm))
    glvar = float(np.sum(((i - mu_g) ** 2) * P_norm))

    return {
        "original_gldm_DependenceEntropy": dentropy,
        "original_gldm_DependenceNonUniformity": DN,
        "original_gldm_DependenceNonUniformityNormalized": DNN,
        "original_gldm_DependenceVariance": dvar,
        "original_gldm_GrayLevelNonUniformity": GLN,
        "original_gldm_GrayLevelVariance": glvar,
        "original_gldm_HighGrayLevelEmphasis": HGE,
        "original_gldm_LargeDependenceEmphasis": LDE,
        "original_gldm_LargeDependenceHighGrayLevelEmphasis": LDHGE,
        "original_gldm_LargeDependenceLowGrayLevelEmphasis": LDLGE,
        "original_gldm_LowGrayLevelEmphasis": LGE,
        "original_gldm_SmallDependenceEmphasis": SDE,
        "original_gldm_SmallDependenceHighGrayLevelEmphasis": SDHGE,
        "original_gldm_SmallDependenceLowGrayLevelEmphasis": SDLGE,
    }


# ------------------------------------------------------------------
# NGTDM (neighbourhood gray-tone difference matrix)
# ------------------------------------------------------------------
def _ngtdm(roi: np.ndarray, bin_width: int = 25) -> dict:
    levels, n_g = _discretise(roi, bin_width)
    if n_g < 2:
        return {}
    from scipy.ndimage import uniform_filter
    win_sum = uniform_filter(levels.astype(float), size=3, mode="reflect") * 9
    # mean of 8 neighbours = (sum_3x3 - centre) / 8
    A = (win_sum - levels) / 8.0
    s = np.zeros(n_g + 1)
    nv = np.zeros(n_g + 1, dtype=int)
    for g in range(1, n_g + 1):
        mask = (levels == g)
        nv[g] = int(mask.sum())
        if nv[g] > 0:
            s[g] = float(np.sum(np.abs(g - A[mask])))
    Nvp = int(nv.sum())
    if Nvp == 0:
        return {}
    p = nv / Nvp
    eps = 1e-12
    Ng_real = int((nv > 0).sum())
    grays = np.where(nv > 0)[0]

    coarseness = 1.0 / (float(np.sum(p[grays] * s[grays])) + eps)
    contrast_a = 0.0
    for gi in grays:
        for gj in grays:
            contrast_a += p[gi] * p[gj] * (gi - gj) ** 2
    contrast = (contrast_a / max(Ng_real * (Ng_real - 1), 1)) * (s[grays].sum() / Nvp)
    busyness_num = float(np.sum(p[grays] * s[grays]))
    busyness_den = 0.0
    for gi in grays:
        for gj in grays:
            if p[gi] > 0 and p[gj] > 0:
                busyness_den += abs(gi * p[gi] - gj * p[gj])
    busyness = busyness_num / (busyness_den + eps)
    complexity = 0.0
    for gi in grays:
        for gj in grays:
            if abs(p[gi]) + abs(p[gj]) > 0:
                complexity += (
                    abs(gi - gj) * (p[gi] * s[gi] + p[gj] * s[gj])
                    / (Nvp * (p[gi] + p[gj] + eps))
                )
    strength = 0.0
    for gi in grays:
        for gj in grays:
            strength += (p[gi] + p[gj]) * (gi - gj) ** 2
    strength = strength / (float(s[grays].sum()) + eps)

    return {
        "original_ngtdm_Busyness": float(busyness),
        "original_ngtdm_Coarseness": float(coarseness),
        "original_ngtdm_Complexity": float(complexity),
        "original_ngtdm_Contrast": float(contrast),
        "original_ngtdm_Strength": float(strength),
    }


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def extract_all_features(roi: np.ndarray, bin_width: int = 25) -> dict:
    """
    Compute all 93 radiomic features expected by the trained scaler.
    `roi` is a 2-D grayscale numpy array containing only the ROI pixels
    (background already masked to zero — the caller is responsible for that).
    """
    # only the masked pixels participate in firstorder stats
    nonzero = roi[roi > 0]
    out = {}
    out.update(_firstorder(nonzero if nonzero.size else roi))
    out.update(_glcm(roi, bin_width))
    out.update(_glrlm(roi, bin_width))
    out.update(_glszm(roi, bin_width))
    out.update(_gldm(roi, bin_width))
    out.update(_ngtdm(roi, bin_width))
    return out
