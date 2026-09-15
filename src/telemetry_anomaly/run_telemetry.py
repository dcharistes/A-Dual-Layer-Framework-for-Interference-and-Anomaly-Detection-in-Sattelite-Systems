#!/usr/bin/env python3
"""
Layer-2 scoreboard: sparse-representation anomaly detection over telemetry
channels. Mirrors the RF side's run_experiments.py.

Two configurations of the same framework:
  Phase 1 : --dict naive --score omp  --threshold fixed
            (naive dictionary + OMP residual + constant CFAR threshold)
  Phase 2 : --dict ksvd  --score admm --threshold ewma   (DEFAULT)
            (K-SVD dictionary + ADMM (X,E) solve + period-EWMA two-sided threshold)

--compare runs Phase-2 reconstruction under BOTH thresholds (fixed vs EWMA) to
reproduce the paper's adaptive-vs-fixed improvement.

If the data directory holds no channels, a synthetic periodic benchmark is
generated automatically so the pipeline runs end to end.

Run (from project root):
    PYTHONPATH=src python3 -m telemetry_anomaly.run_telemetry
    PYTHONPATH=src python3 -m telemetry_anomaly.run_telemetry --compare
    PYTHONPATH=src python3 -m telemetry_anomaly.run_telemetry --data data/telemetry/telemanom
"""
import argparse
import os

import numpy as np

from telemetry_anomaly import data_loader as dl
from telemetry_anomaly import preprocessing as pp
from telemetry_anomaly import sparse_model as sm
from telemetry_anomaly import detector as det

DEFAULT_DATA = os.path.join("data", "telemetry", "synthetic")


def window_scores(ch, args):
    """Standardise -> window -> build model -> per-window residual scores."""
    mu, sd = pp.standardizer(ch.train_values)
    S_tr = pp.sliding_window(ch.train_values, args.window, mu, sd)
    S_te = pp.sliding_window(ch.test_values, args.window, mu, sd)

    if args.dict == "ksvd":
        D = sm.ksvd_dictionary(S_tr, n_atoms=args.atoms, n_nonzero=args.sparsity,
                               n_iter=args.ksvd_iter, seed=args.seed)
    else:
        D = sm.build_dictionary(S_tr, n_atoms=args.atoms, seed=args.seed)

    if args.score == "admm":
        scores, _ = sm.admm_residual_scores(D, S_te, alpha=args.alpha,
                                            beta=args.beta)
    else:
        scores = sm.reconstruction_scores(D, S_te, n_nonzero=args.sparsity)
    return scores


def detect(point_scores, ch, args, threshold):
    if threshold == "ewma":
        p = det.estimate_period(ch.train_values)
        pred, _, _, _ = det.ewma_detect(point_scores, p, lam=args.lam,
                                        eta=args.eta, gamma=args.gamma)
        return det.evaluate_mask(pred, ch.label_mask())
    thr = det.fixed_threshold(point_scores, k=args.k)
    return det.evaluate(point_scores, thr, ch.label_mask())


def scoreboard(channels, args, threshold, title):
    print(f"\n== {title} ==")
    print(f"  {'channel':<10}{'P':>8}{'R':>8}{'F1':>8}{'FPR':>8}")
    print("  " + "-" * 42)
    agg = {kk: 0 for kk in ("tp", "fp", "fn", "tn")}
    f1s = []
    for ch in channels:
        pts = det.windows_to_points(_cached_scores[ch.cid], args.window,
                                    len(ch.test_values))
        m = detect(pts, ch, args, threshold)
        for kk in agg:
            agg[kk] += m[kk]
        f1s.append(m["f1"])
        print(f"  {ch.cid:<10}{m['precision']:>8.3f}{m['recall']:>8.3f}"
              f"{m['f1']:>8.3f}{m['fpr']:>8.4f}")
    tp, fp, fn, tn = agg["tp"], agg["fp"], agg["fn"], agg["tn"]
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    FPR = fp / (fp + tn) if fp + tn else 0.0
    print("  " + "-" * 42)
    print(f"  {'micro-avg':<10}{P:>8.3f}{R:>8.3f}{F1:>8.3f}{FPR:>8.4f}")
    print(f"  {'macro-F1':<10}{np.mean(f1s):>8.3f}")


_cached_scores = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--dict", choices=["naive", "ksvd"], default="ksvd")
    ap.add_argument("--score", choices=["omp", "admm"], default="admm")
    ap.add_argument("--threshold", choices=["fixed", "ewma"], default="ewma")
    ap.add_argument("--compare", action="store_true",
                    help="score once, evaluate under BOTH fixed and ewma thresholds")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--atoms", type=int, default=128)
    ap.add_argument("--sparsity", type=int, default=4, help="OMP T0")
    ap.add_argument("--ksvd-iter", type=int, default=12, dest="ksvd_iter")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=2.6)
    ap.add_argument("--k", type=float, default=4.0, help="fixed-threshold k * MAD")
    ap.add_argument("--lam", type=float, default=0.8, help="EWMA lambda")
    ap.add_argument("--eta", type=float, default=1.0, help="EWMA upper scale")
    ap.add_argument("--gamma", type=float, default=0.5, help="EWMA lower scale")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not (os.path.exists(os.path.join(args.data, "labels.json")) or
            os.path.exists(os.path.join(args.data, "labeled_anomalies.csv"))):
        print(f"[*] no telemetry in {args.data} -- generating synthetic benchmark")
        dl.generate_synthetic(args.data)

    channels = dl.load(args.data)
    print(f"[*] Layer-2 sparse anomaly detection  |  {len(channels)} channels  |  "
          f"dict={args.dict} score={args.score} w={args.window} atoms={args.atoms} "
          f"T0={args.sparsity}")

    # score once (the expensive part), reuse across thresholds
    for ch in channels:
        _cached_scores[ch.cid] = window_scores(ch, args)

    if args.compare:
        scoreboard(channels, args, "fixed",
                   f"FIXED threshold (k={args.k})")
        scoreboard(channels, args, "ewma",
                   f"EWMA threshold (lam={args.lam}, eta={args.eta}, gamma={args.gamma})")
        print("\n[compare] same reconstruction, two thresholds -- the EWMA "
              "two-sided band is the paper's adaptive strategy.")
    else:
        label = (f"EWMA (lam={args.lam}, eta={args.eta}, gamma={args.gamma})"
                 if args.threshold == "ewma" else f"FIXED (k={args.k})")
        scoreboard(channels, args, args.threshold, label)


if __name__ == "__main__":
    main()
