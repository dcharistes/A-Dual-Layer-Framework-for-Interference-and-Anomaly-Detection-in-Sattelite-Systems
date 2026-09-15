#!/usr/bin/env python3
"""
Phase-3 comparison figure + table: our two methods (K-SVD-ADMM, PCA) against a
PUBLISHED reference baseline (LSTM-NDT) on SMAP/MSL.

Reads results/metrics/telemetry/per_channel.csv (written by evaluate.py, with
per-channel confusion counts) and micro-aggregates by subset (ALL / SMAP / MSL).
Draws two panels:
  (left)  "the scoring metric decides the winner": point-wise vs point-adjusted
          F1 for both methods on ALL channels.
  (right) "vs published": point-adjusted F1 per spacecraft for both methods plus
          the LSTM-NDT reference.

The LSTM-NDT numbers are point-adjusted F1 as reported in the literature
(values: Su et al., "Robust Anomaly Detection for Multivariate Time Series through
Stochastic Recurrent Neural Networks", KDD 2019, Table 4 -- a re-evaluation of
Hundman et al. 2018 / telemanom). They are CONTEXT only: threshold selection and
preprocessing differ from ours, so treat them as a reference band, not a
controlled head-to-head. The only controlled comparison here is K-SVD-ADMM vs PCA
(identical channels, scoring and operating-point protocol).

Run (after evaluate.py):
    PYTHONPATH=src python3 -m telemetry_anomaly.make_comparison
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_DIR = os.path.join("results", "metrics", "telemetry")
FIG_DIR = os.path.join("results", "figures", "telemetry")

# Published reference (point-adjusted F1), cited above. Context, not a controlled run.
LSTM_NDT = {"SMAP": {"P": 0.8965, "R": 0.8846, "F1": 0.8905},
            "MSL":  {"P": 0.5934, "R": 0.5374, "F1": 0.5640}}
REF_NAME = "LSTM-NDT (publ.)"


def _f1(tp, fp, fn):
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return 2 * P * R / (P + R) if P + R else 0.0


def aggregate(rows, method, sc):
    """Micro point-wise and point-adjusted F1 for a method over a subset."""
    sel = [r for r in rows if r["method"] == method
           and (sc == "ALL" or r["spacecraft"] == sc)]
    pw = _f1(sum(int(r["pw_tp"]) for r in sel), sum(int(r["pw_fp"]) for r in sel),
             sum(int(r["pw_fn"]) for r in sel))
    pa = _f1(sum(int(r["pa_tp"]) for r in sel), sum(int(r["pa_fp"]) for r in sel),
             sum(int(r["pa_fn"]) for r in sel))
    return pw, pa


def main():
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(CSV_DIR, "per_channel.csv")
    # tolerate human-aligned CSVs (spaces after commas) -> strip keys and values
    rows = [{k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
            for r in csv.DictReader(open(path))]
    if not rows or "pw_tp" not in rows[0]:
        raise SystemExit("per-channel CSV lacks count columns; re-run evaluate.py.")
    methods = ["KSVD-ADMM", "PCA"]

    # --- console table ---
    print(f"{'subset':<6}{'method':<16}{'pw_F1':>8}{'pa_F1':>8}")
    print("-" * 40)
    agg = {}
    for sc in ("ALL", "SMAP", "MSL"):
        for m in methods:
            pw, pa = aggregate(rows, m, sc)
            agg[(sc, m)] = (pw, pa)
            print(f"{sc:<6}{m:<16}{pw:>8.3f}{pa:>8.3f}")
        if sc in LSTM_NDT:
            print(f"{sc:<6}{REF_NAME:<16}{'--':>8}{LSTM_NDT[sc]['F1']:>8.3f}")
    print("-" * 40)

    # comparison CSV (incl. published reference rows)
    with open(os.path.join(CSV_DIR, "comparison.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["subset", "method", "pw_F1", "pa_F1", "source"])
        for sc in ("ALL", "SMAP", "MSL"):
            for m in methods:
                pw, pa = agg[(sc, m)]
                wr.writerow([sc, m, round(pw, 3), round(pa, 3), "this work"])
            if sc in LSTM_NDT:
                wr.writerow([sc, REF_NAME, "", round(LSTM_NDT[sc]["F1"], 3),
                             "Su et al. 2019 (point-adjusted)"])

    # --- figure ---
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

    # left: metric decides the winner (ALL channels)
    x = np.arange(len(methods)); bw = 0.35
    pw_vals = [agg[("ALL", m)][0] for m in methods]
    pa_vals = [agg[("ALL", m)][1] for m in methods]
    axL.bar(x - bw / 2, pw_vals, bw, label="point-wise F1", color="steelblue")
    axL.bar(x + bw / 2, pa_vals, bw, label="point-adjusted F1", color="indianred")
    for xi, v in zip(x - bw / 2, pw_vals):
        axL.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    for xi, v in zip(x + bw / 2, pa_vals):
        axL.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    axL.set_xticks(x); axL.set_xticklabels(methods)
    axL.set_ylim(0, 1.05); axL.set_ylabel("F1"); axL.legend(fontsize=9)
    axL.set_title("Scoring metric decides the winner (all channels)\n"
                  "K-SVD-ADMM: precise localization | PCA: segment coverage")
    axL.grid(axis="y", alpha=0.3)

    # right: point-adjusted F1 per spacecraft, vs published LSTM-NDT
    subs = ["SMAP", "MSL"]
    series = {"KSVD-ADMM": [agg[(s, "KSVD-ADMM")][1] for s in subs],
              "PCA":       [agg[(s, "PCA")][1] for s in subs],
              REF_NAME:    [LSTM_NDT[s]["F1"] for s in subs]}
    colors = {"KSVD-ADMM": "seagreen", "PCA": "indianred", REF_NAME: "gray"}
    xs = np.arange(len(subs)); bw2 = 0.26
    for i, (name, vals) in enumerate(series.items()):
        off = (i - 1) * bw2
        axR.bar(xs + off, vals, bw2, label=name, color=colors[name],
                hatch="//" if name == REF_NAME else None)
        for xi, v in zip(xs + off, vals):
            axR.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    axR.set_xticks(xs); axR.set_xticklabels(subs)
    axR.set_ylim(0, 1.05); axR.set_ylabel("point-adjusted F1"); axR.legend(fontsize=9)
    axR.set_title("Point-adjusted F1 vs published reference\n"
                  "(LSTM-NDT: Su et al. 2019 — context, protocol differs)")
    axR.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "comparison.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"[+] figure -> {out}")
    print(f"[+] csv    -> {CSV_DIR}/comparison.csv")


if __name__ == "__main__":
    main()
