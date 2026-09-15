#!/usr/bin/env python3
"""
Phase-3 evaluation: K-SVD-ADMM telemetry anomaly detector vs a PCA-reconstruction
baseline, on the real SMAP/MSL benchmark (or the synthetic set).

For each channel and method we compute per-window residual scores, map them to
points, then choose a per-method operating point by sweeping the threshold and
keeping the best point-ADJUSTED F1 (the SMAP/MSL community scoring used by the
published baselines). We report BOTH point-wise and point-adjusted P/R/F1/FPR,
micro-averaged over channels, plus per-channel CSVs and example score-vs-time
figures (paper Fig. 3 style).

Run (from project root):
    PYTHONPATH=src python3 -m telemetry_anomaly.evaluate --data data/telemetry/telemanom
    PYTHONPATH=src python3 -m telemetry_anomaly.evaluate --data data/telemetry/telemanom \
        --channels P-1,S-1,E-1,T-1 --figures
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from telemetry_anomaly import data_loader as dl
from telemetry_anomaly import preprocessing as pp
from telemetry_anomaly import sparse_model as sm
from telemetry_anomaly import baselines as bl
from telemetry_anomaly import detector as det

FIG_DIR = os.path.join("results", "figures", "telemetry")
CSV_DIR = os.path.join("results", "metrics", "telemetry")


def channel_scores(ch, args):
    """Per-point anomaly scores for both methods on one channel."""
    mu, sd = pp.standardizer(ch.train_values)
    S_tr = pp.sliding_window(ch.train_values, args.window, mu, sd)
    S_te = pp.sliding_window(ch.test_values, args.window, mu, sd)
    T = len(ch.test_values)

    D = sm.ksvd_dictionary(S_tr, n_atoms=args.atoms, n_nonzero=args.sparsity,
                           n_iter=args.ksvd_iter, seed=args.seed)
    ksvd_w, _ = sm.admm_residual_scores(D, S_te, alpha=args.alpha, beta=args.beta)

    mean, U = bl.pca_fit(S_tr, n_components=args.pca_components)
    pca_w = bl.pca_scores(S_te, mean, U)

    return {"KSVD-ADMM": det.windows_to_points(ksvd_w, args.window, T),
            "PCA":       det.windows_to_points(pca_w, args.window, T)}


def best_operating_point(point_scores, label_mask, n_grid=40):
    """Sweep the threshold; keep the one with best point-ADJUSTED F1."""
    lo, hi = np.percentile(point_scores, 50), point_scores.max()
    best = {"f1": -1.0}
    for thr in np.linspace(lo, hi, n_grid):
        pred = point_scores > thr
        m_pa = det.evaluate_pa(pred, label_mask)
        if m_pa["f1"] > best["f1"]:
            m_pw = det.evaluate_mask(pred, label_mask)
            best = {"thr": float(thr), "f1": m_pa["f1"], "pa": m_pa, "pw": m_pw}
    return best


def micro(confusions):
    tp = sum(c["tp"] for c in confusions); fp = sum(c["fp"] for c in confusions)
    fn = sum(c["fn"] for c in confusions); tn = sum(c["tn"] for c in confusions)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    FPR = fp / (fp + tn) if fp + tn else 0.0
    return P, R, F1, FPR


def fig_channel(ch, scores, ops, path):
    methods = list(scores.keys())
    fig, axes = plt.subplots(len(methods) + 1, 1, figsize=(11, 2.2 * (len(methods) + 1)),
                             sharex=True)
    axes[0].plot(ch.test_values, color="steelblue", lw=0.7)
    axes[0].set_ylabel("telemetry")
    for s, e in ch.anomaly_sequences:
        for ax in axes:
            ax.axvspan(s, e, color="orange", alpha=0.25)
    for ax, mth in zip(axes[1:], methods):
        ax.plot(scores[mth], color="firebrick", lw=0.7, label=f"{mth} score")
        ax.axhline(ops[mth]["thr"], color="black", ls="--", lw=0.8, label="threshold")
        ax.set_ylabel(mth); ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("time index")
    fig.suptitle(f"{ch.cid} ({ch.spacecraft}) -- score vs time, anomalies shaded")
    fig.tight_layout()
    fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("data", "telemetry", "synthetic"))
    ap.add_argument("--channels", default="", help="comma list; default all")
    ap.add_argument("--max-channels", type=int, default=0, dest="max_channels")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--atoms", type=int, default=128)
    ap.add_argument("--sparsity", type=int, default=4)
    ap.add_argument("--ksvd-iter", type=int, default=10, dest="ksvd_iter")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=2.6)
    ap.add_argument("--pca-components", type=int, default=10, dest="pca_components")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(CSV_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)
    channels = dl.load(args.data)
    if args.channels:
        want = set(args.channels.split(","))
        channels = [c for c in channels if c.cid in want]
    if args.max_channels:
        channels = channels[:args.max_channels]

    methods = ["KSVD-ADMM", "PCA"]
    conf_pw = {m: [] for m in methods}      # list of (spacecraft, confusion)
    conf_pa = {m: [] for m in methods}
    rows = []
    print(f"[*] Phase-3 evaluation  |  {len(channels)} channels  |  "
          f"w={args.window} atoms={args.atoms} pca_k={args.pca_components}\n")
    for n, ch in enumerate(channels, 1):
        if len(ch.test_values) < args.window + 5 or not ch.anomaly_sequences:
            continue
        scores = channel_scores(ch, args)
        lbl_mask = ch.label_mask()
        ops = {}
        for m in methods:
            op = best_operating_point(scores[m], lbl_mask)
            ops[m] = op
            conf_pw[m].append((ch.spacecraft, op["pw"]))
            conf_pa[m].append((ch.spacecraft, op["pa"]))
            pw, pa = op["pw"], op["pa"]
            rows.append([ch.cid, ch.spacecraft, m,
                         round(pw["precision"], 3), round(pw["recall"], 3),
                         round(pw["f1"], 3),
                         round(pa["precision"], 3), round(pa["recall"], 3),
                         round(pa["f1"], 3), round(pa["fpr"], 4),
                         pw["tp"], pw["fp"], pw["fn"], pw["tn"],
                         pa["tp"], pa["fp"], pa["fn"], pa["tn"]])
        print(f"  [{n:>2}/{len(channels)}] {ch.cid:<6} "
              + "  ".join(f"{m} PA-F1={ops[m]['pa']['f1']:.2f}" for m in methods))
        if args.figures and n <= 6:
            fig_channel(ch, scores, ops, os.path.join(FIG_DIR, f"{ch.cid}.png"))

    # per-channel CSV (with counts so any subset can be micro-aggregated later)
    with open(os.path.join(CSV_DIR, "per_channel.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["chan", "spacecraft", "method", "pw_P", "pw_R", "pw_F1",
                     "pa_P", "pa_R", "pa_F1", "pa_FPR",
                     "pw_tp", "pw_fp", "pw_fn", "pw_tn",
                     "pa_tp", "pa_fp", "pa_fn", "pa_tn"])
        wr.writerows(rows)

    # summary comparison table, per subset (ALL / SMAP / MSL)
    def subset(confs, sc):
        return [c for s, c in confs if sc == "ALL" or s == sc]

    print("\n  == micro-averaged comparison ==")
    print(f"  {'subset':<6}{'method':<12}{'pw_P':>7}{'pw_R':>7}{'pw_F1':>7}   "
          f"{'pa_P':>7}{'pa_R':>7}{'pa_F1':>7}{'pa_FPR':>8}")
    print("  " + "-" * 76)
    summary = []
    for sc in ("ALL", "SMAP", "MSL"):
        for m in methods:
            if not subset(conf_pa[m], sc):
                continue
            pP, pR, pF, pFPR = micro(subset(conf_pw[m], sc))
            aP, aR, aF, aFPR = micro(subset(conf_pa[m], sc))
            print(f"  {sc:<6}{m:<12}{pP:>7.3f}{pR:>7.3f}{pF:>7.3f}   "
                  f"{aP:>7.3f}{aR:>7.3f}{aF:>7.3f}{aFPR:>8.4f}")
            summary.append([sc, m, round(pP, 3), round(pR, 3), round(pF, 3),
                            round(aP, 3), round(aR, 3), round(aF, 3), round(aFPR, 4)])
        print("  " + "-" * 76)
    with open(os.path.join(CSV_DIR, "summary.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["subset", "method", "pw_P", "pw_R", "pw_F1",
                     "pa_P", "pa_R", "pa_F1", "pa_FPR"])
        wr.writerows(summary)
    print(f"\n[+] CSVs -> {CSV_DIR}/summary.csv, per_channel.csv")
    if args.figures:
        print(f"[+] figures -> {FIG_DIR}/<chan>.png")


if __name__ == "__main__":
    main()
