#!/usr/bin/env python3
"""
Preprocessing for the sparse telemetry detector.

Collective anomalies live in SUBSEQUENCES, not single samples, so we cannot judge
points in isolation. Following the paper (Fig. 2), we slide a window of length w
(shift 1) over the series and stack each window as a column -> a sample matrix
S in R^{w x L}, L = T - w + 1. The dictionary then models the local DYNAMICS of a
window rather than a single value.

Standardisation uses TRAIN statistics only (the test split must not leak into the
normality model).
"""
import numpy as np


def standardizer(train_values):
    """Return (mean, std) from the train split; std floored away from 0."""
    mu = float(np.mean(train_values))
    sd = float(np.std(train_values))
    return mu, max(sd, 1e-8)


def sliding_window(values, w, mu=0.0, sd=1.0):
    """
    Stack length-w windows (shift 1) as columns of S in R^{w x L}.
    Values are standardised with the supplied (mu, sd) first.
    """
    x = (np.asarray(values, dtype=float) - mu) / sd
    T = len(x)
    if T < w:
        raise ValueError(f"series length {T} shorter than window {w}")
    L = T - w + 1
    # stride trick would alias the source buffer; an explicit copy is safer here.
    S = np.empty((w, L), dtype=float)
    for i in range(L):
        S[:, i] = x[i:i + w]
    return S
