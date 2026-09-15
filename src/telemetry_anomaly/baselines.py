#!/usr/bin/env python3
"""
Baseline detectors for the Phase-3 comparison.

PCA reconstruction is the natural baseline for the sparse layer: it is the SAME
reconstruction-residual framework, but with a DENSE, LINEAR, ORTHOGONAL model
(the top-k principal components of the train windows) instead of an OVERCOMPLETE,
SPARSE one (K-SVD). Comparing the two isolates exactly what sparsity +
overcompleteness buy -- both score a window by how badly the normal model
reconstructs it. Pure NumPy, no extra dependency (sklearn is absent here).

Published LSTM-NDT / OCSVM numbers on SMAP/MSL (Hundman et al. 2018; the paper's
baselines) are cited directly in the thesis table; PCA is the one we run ourselves
on identical channels and scoring so the comparison is controlled.
"""
import numpy as np


def pca_fit(S_train, n_components=10):
    """Top-k principal directions of the (w-dim) train windows. Returns (mean, U_k)."""
    mean = S_train.mean(axis=1, keepdims=True)
    U, _s, _Vt = np.linalg.svd(S_train - mean, full_matrices=False)
    k = min(n_components, U.shape[1])
    return mean, U[:, :k]


def pca_scores(S, mean, U):
    """Per-window anomaly score = ||residual after projecting onto the PCA subspace||."""
    Xc = S - mean
    resid = Xc - U @ (U.T @ Xc)
    return np.linalg.norm(resid, axis=0)
