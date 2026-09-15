#!/usr/bin/env python3
"""
Sparse representation core (Phase 1: dictionary + OMP reconstruction).

The normality model is a dictionary D whose columns ("atoms") are normal local
patterns. A test window y is sparse-coded as y ~ D x (only T0 atoms active); the
RESIDUAL ||y - D x|| is the anomaly evidence (eq. 9): a window built from normal
dynamics reconstructs well (small residual); an anomalous window does not.

Phase 1 keeps the dictionary NAIVE -- a random, l2-normalised subset of the train
windows -- and codes with Orthogonal Matching Pursuit (OMP). This is the exact
sparse-coding step the paper uses inside K-SVD; Phase 2 swaps the naive dictionary
for a K-SVD-LEARNED one and OMP for the ADMM joint (X, E) solve. Keeping it pure
NumPy avoids a heavy dependency and matches the custom ADMM coming in Phase 2.
"""
import numpy as np


def l2_normalize_columns(D):
    norms = np.linalg.norm(D, axis=0)
    norms[norms < 1e-12] = 1.0
    return D / norms


def build_dictionary(S_train, n_atoms=256, seed=0):
    """Naive dictionary: a random l2-normalised subset of the train windows."""
    L = S_train.shape[1]
    rng = np.random.default_rng(seed)
    k = min(n_atoms, L)
    cols = rng.choice(L, size=k, replace=False)
    return l2_normalize_columns(S_train[:, cols].copy())


def omp(D, y, n_nonzero=4, tol=1e-6):
    """
    Orthogonal Matching Pursuit: approximate argmin ||y - D x|| s.t. ||x||_0 <= k.
    Returns (x, residual_vector). D columns are assumed l2-normalised.

    Already-selected atoms are masked out each step so the support always grows
    to n_nonzero (the K-SVD update needs a full, non-degenerate support).
    """
    n_atoms = D.shape[1]
    residual = y.astype(float).copy()
    idx = []
    x = np.zeros(n_atoms)
    coef = None
    for _ in range(n_nonzero):
        proj = np.abs(D.T @ residual)               # correlate atoms with residual
        if idx:
            proj[idx] = -np.inf                      # never reselect an active atom
        idx.append(int(np.argmax(proj)))
        Ds = D[:, idx]
        coef, *_ = np.linalg.lstsq(Ds, y, rcond=None)   # re-fit on the active set
        residual = y - Ds @ coef
        if np.linalg.norm(residual) < tol:
            break
    if idx and coef is not None:
        x[idx] = coef
    return x, residual


def reconstruction_scores(D, S, n_nonzero=4):
    """Per-window anomaly score = residual l2-norm ||y - D x||_2 (eq. 9)."""
    L = S.shape[1]
    scores = np.empty(L)
    for i in range(L):
        _, r = omp(D, S[:, i], n_nonzero=n_nonzero)
        scores[i] = np.linalg.norm(r)
    return scores


# ---------------------------------------------------------------------------
# Phase 2 -- K-SVD dictionary learning (Algorithm 1) and ADMM solve (Algorithm 2)
# Equations verified in docs/ksvd_admm_notes.md.
# ---------------------------------------------------------------------------
def ksvd_dictionary(S_train, n_atoms=256, n_nonzero=4, n_iter=15,
                    seed=0, max_train=2000, verbose=False):
    """
    Learn a dictionary D from the (anomaly-free) train windows via K-SVD.

    Alternates: (i) sparse-code every sample with OMP, then (ii) update each atom
    d_k and its coefficient row from the rank-1 SVD of the representation error on
    just the samples that use d_k. Unused atoms are re-seeded onto the
    worst-reconstructed sample so the dictionary does not collapse.

    On long channels the train windows are randomly subsampled to max_train before
    learning (the per-window OMP loop is the cost; the dictionary does not need
    every overlapping window). Set max_train=None to use all.
    """
    rng = np.random.default_rng(seed)
    if max_train is not None and S_train.shape[1] > max_train:
        S_train = S_train[:, rng.choice(S_train.shape[1], size=max_train, replace=False)]
    w, L = S_train.shape
    K = min(n_atoms, L)
    D = l2_normalize_columns(S_train[:, rng.choice(L, size=K, replace=False)].copy())

    for it in range(n_iter):
        # (i) sparse coding
        X = np.zeros((K, L))
        for i in range(L):
            X[:, i], _ = omp(D, S_train[:, i], n_nonzero=n_nonzero)
        # (ii) atom-by-atom dictionary update
        for k in range(K):
            using = np.nonzero(X[k, :])[0]
            if using.size == 0:
                err = np.sum((S_train - D @ X) ** 2, axis=0)
                worst = int(np.argmax(err))
                D[:, k] = S_train[:, worst] / (np.linalg.norm(S_train[:, worst]) + 1e-12)
                continue
            X[k, using] = 0.0                                  # remove atom k
            E_k = S_train[:, using] - D @ X[:, using]          # error w/o atom k (E_k^R)
            U, s, Vt = np.linalg.svd(E_k, full_matrices=False)
            D[:, k] = U[:, 0]                                  # d_k <- u_1
            X[k, using] = s[0] * Vt[0, :]                      # x_T^k <- sigma_1 * v_1
        if verbose:
            rmse = np.linalg.norm(S_train - D @ X) / np.sqrt(L)
            print(f"   K-SVD iter {it + 1:>2}/{n_iter}  RMSE/col {rmse:.4f}")
    return D


def admm_residual_scores(D, S, alpha=1.0, beta=2.6, mu0=1.0, rho=1.25,
                         mu_max=1e4, n_iter=100, tol=1e-4):
    """
    Solve eq. 6  min (1/2)||S - DX - E||_F^2 + alpha||X||_{1,1} + beta||E||_{2,1}
    by ADMM (Algorithm 2). E is the outlier/residual matrix; the per-window
    anomaly score is ||E(:,j)||_2 (eq. 9).

    rho>1 escalates the penalty (paper); we cap it at mu_max for numerical
    stability and stop early on a small primal residual (the 3-block split has
    no strict convergence guarantee -- see docs/ksvd_admm_notes.md).
    """
    w, L = S.shape
    M = D.shape[1]
    X = np.zeros((M, L)); Z = np.zeros((M, L)); Mult = np.zeros((M, L))
    E = np.zeros((w, L))
    mu = mu0
    DtD = D.T @ D
    DtS = D.T @ S
    I = np.eye(M)
    for _ in range(n_iter):
        # X-update: (D^T D + mu I) X = D^T(S - E) + Mult + mu Z
        X = np.linalg.solve(DtD + mu * I, DtS - D.T @ E + Mult + mu * Z)
        # Z-update: element-wise soft-threshold at alpha/mu
        P = X - Mult / mu
        Z = np.sign(P) * np.maximum(np.abs(P) - alpha / mu, 0.0)
        # E-update: column-wise block soft-threshold at beta on Q = S - D X
        Q = S - D @ X
        norms = np.linalg.norm(Q, axis=0) + 1e-12
        E = Q * np.maximum(0.0, 1.0 - beta / norms)
        # dual + penalty
        primal = Z - X
        Mult = Mult + mu * primal
        if np.linalg.norm(primal) / (np.linalg.norm(X) + 1e-12) < tol:
            break
        mu = min(mu * rho, mu_max)
    return np.linalg.norm(E, axis=0), E
