#!/usr/bin/env python3
"""
Scoring + thresholding + metrics for the telemetry anomaly layer.

Window scores -> per-point scores -> threshold -> P/R/F1/FPR vs ground truth.

Phase 1 threshold is FIXED (constant in time): a CFAR-style med + k*MAD on the
score distribution. This is deliberately the SAME robust threshold the RF temporal
layer uses (src/multiframe_cuc.py), i.e. the non-periodic instance of the unified
framework. Phase 2 will add the paper's time-varying period-based EWMA band and
compare the two (reproducing their adaptive-vs-fixed improvement).
"""
import numpy as np


def windows_to_points(scores, w, T):
    """
    Spread each length-w window score over the points it covers and take the max
    (a point is as anomalous as the most-anomalous window containing it). Window i
    covers points [i, i+w-1]; there are L = T-w+1 windows.
    """
    pts = np.zeros(T)
    L = len(scores)
    for i in range(L):
        s = scores[i]
        seg = slice(i, i + w)
        np.maximum(pts[seg], s, out=pts[seg])
    return pts


def fixed_threshold(scores, k=4.0):
    """Constant CFAR threshold: median + k * (robust sigma). Returns the scalar.

    Falls back to the std when the MAD is ~0 (a near-binary residual where most
    windows reconstruct perfectly), so the threshold does not collapse to 0.
    """
    med = np.median(scores)
    mad = 1.4826 * np.median(np.abs(scores - med))
    spread = mad if mad > 1e-6 else float(np.std(scores))
    return med + k * spread


def estimate_period(values, min_p=5, max_p=None):
    """First dominant autocorrelation peak of the train signal -> period p."""
    x = np.asarray(values, dtype=float) - np.mean(values)
    n = len(x)
    if max_p is None:
        max_p = n // 2
    ac = np.correlate(x, x, mode="full")[n - 1:]
    ac = ac / (ac[0] + 1e-12)
    window = ac[min_p:max_p]
    if window.size == 0:
        return min_p
    return min_p + int(np.argmax(window))


def ewma_detect(point_scores, period, lam=0.8, eta=1.0, gamma=0.5, floor=None):
    """
    Period-based EWMA two-sided detector (paper Algorithm 3, eqs 10-12).

    U_t = lam * U_{t-p} + (1-lam) * a_t  (references the PREVIOUS PERIOD, not t-1),
    flag if a_t > U_t(1+eta) + floor   [novel pattern, high residual]
        or a_t < U_t(1-gamma) - floor  [over-easy reconstruction, e.g. frozen],
    and on a flag reset U_t <- U_{t-p} so the anomaly does not poison the average.

    `floor` is an ADDITIVE margin on the multiplicative band. The paper's band is
    purely multiplicative (floor=0), which degenerates when the residual baseline
    approaches 0 (a near-perfectly reconstructed signal): U_t -> 0 makes the band
    collapse and flags baseline noise. The additive floor is precisely the RF
    layer's CFAR margin -- floor=0 reproduces the paper, floor>0 is the robust
    generalization (see docs/framework_chapter_outline.md). floor=None auto-sets
    it to half the robust sigma of the score series.

    Returns (pred_mask, tau_up, tau_down, U).
    """
    a = np.asarray(point_scores, dtype=float)
    n = len(a)
    if floor is None:
        # Half the robust spread of the score series. The MAD is EXACTLY zero
        # whenever most windows reconstruct perfectly (the ADMM E-step zeroes
        # normal columns outright), which is the near-binary regime this floor
        # exists to handle -- without the fallback the floor would be 0 and the
        # band would collapse to the purely multiplicative rule it repairs.
        # Mirrors the same guard in fixed_threshold().
        med = np.median(a)
        mad = 1.4826 * np.median(np.abs(a - med))
        spread = mad if mad > 1e-6 else float(np.std(a))
        floor = 0.5 * spread
    U = np.zeros(n)
    up = np.zeros(n)
    down = np.zeros(n)
    pred = np.zeros(n, dtype=bool)
    for t in range(n):
        ref = U[t - period] if t >= period else (U[t - 1] if t > 0 else a[t])
        U[t] = lam * ref + (1.0 - lam) * a[t]
        up[t] = U[t] * (1.0 + eta) + floor
        down[t] = U[t] * (1.0 - gamma) - floor
        if a[t] > up[t] or a[t] < down[t]:
            pred[t] = True
            U[t] = ref                      # do not let the anomaly skew the trend
    return pred, up, down, U


def evaluate_mask(pred, label_mask):
    """Point-wise confusion + P/R/F1/FPR from an explicit prediction mask."""
    truth = label_mask
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    tn = int(np.sum(~pred & ~truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1, "fpr": fpr}


def evaluate(point_scores, threshold, label_mask):
    """Point-wise confusion + P/R/F1/FPR from a constant (fixed) threshold."""
    return evaluate_mask(point_scores > threshold, label_mask)


def _segments(mask):
    """Contiguous True runs of a boolean mask as [(start, end_inclusive), ...]."""
    runs = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def point_adjust(pred, label_mask):
    """
    Point-adjusted predictions (Xu et al. 2018; the SMAP/MSL community standard
    used by LSTM-NDT etc.): if ANY point inside a true anomaly segment is flagged,
    the WHOLE segment counts as detected. Lets a detector that catches part of an
    event be credited for the event, which is how the baselines we compare against
    were scored. Returns an adjusted prediction mask.
    """
    adj = pred.copy()
    for s, e in _segments(label_mask):
        if pred[s:e + 1].any():
            adj[s:e + 1] = True
    return adj


def evaluate_pa(pred, label_mask):
    """Point-ADJUSTED confusion + P/R/F1/FPR."""
    return evaluate_mask(point_adjust(pred, label_mask), label_mask)
