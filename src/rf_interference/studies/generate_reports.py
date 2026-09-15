#!/usr/bin/env python3
"""
Generate all figures and metric CSVs for the 10 experiments.

Figures (./results/figures/):
  - <exp>_spectrum.png        per experiment: raw + denoised spectrum, estimated
                              noise floor, authorized carriers (green) and
                              detected CUC hosts (red), with the TRUE CUC
                              locations marked (detected vs missed).
  - _methodology_E01.png      how vote-driven segmentation works: denoised
                              spectrum + start/end votes + detected segments.
  - _roc.png                  recall vs false-alarms across sensitivity presets.
  - _summary_outcomes.png     per-experiment detected / missed / false-alarm bars.

Metrics (./results/metrics/):
  - summary_metrics.csv       one row per experiment (segmentation + CUC metrics).
  - cuc_detail.csv            one row per true CUC (detected?, localization error).
  - sensitivity_roc.csv       one row per sensitivity preset (pooled TP/FN/FP/PRF).
  - roc_by_experiment.csv     experiment x preset detection counts.
  - carriers/<exp>.csv        every detected carrier with its classification.

Run (from the project root, after generating datasets):
    PYTHONPATH=src python3 src/generate_reports.py
"""
import csv
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rf_interference import wavelet_dsp
from rf_interference import carrier_cuc_analysis as cca
from rf_interference import multiframe_cuc as mf
from rf_interference.psd import welch_psd_db

EXP_DIR = "./data/experiments"
OUT_FIG = "./results/figures/rf"
OUT_CSV = "./results/metrics/rf"
NUM_CHUNKS = 400
WEIGHT_MODE = "entropy"
MATCH_TOL_MHZ = 1.0
DEFAULT_SENS = "conservative"
PRESETS = list(cca.SENSITIVITY_PRESETS)
TEMPORAL_N = 2048   # frames for the temporal baseline-differencing layer


# --------------------------------------------------------------------------- #
# Core analysis
# --------------------------------------------------------------------------- #
def analyze(manifest, sensitivity):
    """Run the full detector; return everything needed for plots + metrics."""
    freqs, power_db = welch_psd_db(manifest["dataset"], num_chunks=NUM_CHUNKS)
    clean, v_start, v_end = wavelet_dsp.detect_boundaries_advanced(
        freqs, power_db, weight_mode=WEIGHT_MODE, verbose=False)
    floor = cca._estimate_tilt_floor(clean)
    carriers = cca.extract_and_analyze_carriers(
        freqs, power_db, v_start, v_end, clean,
        sensitivity=sensitivity, verbose=False)
    return dict(freqs=freqs, power_db=power_db, clean=clean, floor=floor,
                v_start=v_start, v_end=v_end, carriers=carriers)


def score(manifest, carriers):
    """CUC + segmentation metrics for one experiment."""
    flagged = [(c["bump_freq"], c) for c in carriers if c["spoofed_flag"]]
    true_cucs = manifest["cucs"]
    gt_legit = [c for c in manifest["carriers"] if not c["is_cuc"]]

    # CUC matching
    detected, missed, loc_err = [], [], []
    for cuc in true_cucs:
        fc = cuc["fc_mhz"]
        hit = min((b for b, _ in flagged), key=lambda b: abs(b - fc), default=None)
        if hit is not None and abs(hit - fc) <= MATCH_TOL_MHZ:
            detected.append((cuc, hit)); loc_err.append(abs(hit - fc))
        else:
            missed.append(cuc)
    false_pos = [b for b, _ in flagged
                 if all(abs(b - c["fc_mhz"]) > MATCH_TOL_MHZ for c in true_cucs)]

    # Carrier segmentation coverage: a legit carrier is "found" if its centre
    # lies inside some detected segment (handles merged overlapping carriers).
    spans = [(c["start_idx"], c["end_idx"]) for c in carriers]
    freqs = None  # filled by caller-independent check below via manifest? -> use idx-free
    return dict(detected=detected, missed=missed, false_pos=false_pos,
                loc_err=loc_err, n_cucs=len(true_cucs),
                gt_legit=gt_legit, flagged=flagged, spans=spans)


def coverage(freqs, carriers, gt_legit):
    covered = 0
    ranges = [(freqs[c["start_idx"]], freqs[c["end_idx"]]) for c in carriers]
    for g in gt_legit:
        fc = g["fc_mhz"]
        if any(a <= fc <= b for a, b in ranges):
            covered += 1
    return covered, len(gt_legit)


def prf(tp, fn, fp):
    rec = tp / (tp + fn) if (tp + fn) else None
    prec = tp / (tp + fp) if (tp + fp) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
    return rec, prec, f1


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_spectrum(man, A, sc, cov, path):
    f, raw, clean, floor = A["freqs"], A["power_db"], A["clean"], A["floor"]
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(f, raw, color="lightgray", alpha=0.7, label="Raw PSD")
    ax.plot(f, clean, color="dodgerblue", lw=1.2, label="Wavelet-denoised")
    ax.plot(f, floor, color="dimgray", ls=":", lw=1.2, label="Noise floor (tilt)")

    seen = set()
    for c in A["carriers"]:
        spoof = c["spoofed_flag"]
        color = "red" if spoof else "mediumseagreen"
        lab = "Detected CUC host" if spoof else "Authorized carrier"
        ax.axvspan(f[c["start_idx"]], f[c["end_idx"]], color=color, alpha=0.22,
                   label=lab if lab not in seen else None)
        seen.add(lab)

    # Headroom so the CUC labels sit inside the axes, clear of the title.
    lo, hi = floor.min() - 2, raw.max()
    ax.set_ylim(lo, hi + 0.18 * (hi - lo))

    # True CUC markers (green = detected, red = missed)
    det_fc = {round(cuc["fc_mhz"], 3) for cuc, _ in sc["detected"]}
    ytext = hi + 0.04 * (hi - lo)
    for cuc in man["cucs"]:
        fc = cuc["fc_mhz"]
        found = round(fc, 3) in det_fc
        col = "green" if found else "crimson"
        ax.axvline(fc, color=col, ls="--", lw=1.5)
        ax.annotate(f"CUC {'OK' if found else 'MISS'}\n{fc:+.1f}",
                    xy=(fc, ytext), color=col, fontsize=8, ha="center",
                    va="bottom", fontweight="bold")

    tp, fn, fp = len(sc["detected"]), len(sc["missed"]), len(sc["false_pos"])
    ax.set_title(f"{man['name']}  |  {man['desc']}\n"
                 f"carriers covered {cov[0]}/{cov[1]}  |  "
                 f"CUC detected {tp}/{sc['n_cucs']}  missed {fn}  false+ {fp}",
                 fontsize=10)
    ax.set_xlabel("Frequency (MHz)"); ax.set_ylabel("Power (dB)")
    ax.grid(True, alpha=0.3); ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(f[0], f[-1])
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def fig_methodology(man, A, path):
    f, clean, floor = A["freqs"], A["clean"], A["floor"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    ax0, ax1 = axes
    ax0.plot(f, clean, color="dodgerblue", label="Wavelet-denoised")
    ax0.plot(f, floor, color="dimgray", ls=":", label="Noise floor")
    for c in A["carriers"]:
        col = "red" if c["spoofed_flag"] else "mediumseagreen"
        ax0.axvspan(f[c["start_idx"]], f[c["end_idx"]], color=col, alpha=0.20)
    ax0.set_ylabel("Power (dB)"); ax0.grid(True, alpha=0.3)
    ax0.legend(loc="lower right", fontsize=8)
    ax0.set_title(f"Vote-driven segmentation -- {man['name']}\n"
                  "top: denoised spectrum + detected segments;  "
                  "bottom: cross-scale start/end edge votes", fontsize=10)

    ax1.plot(f, A["v_start"], color="seagreen", label="V_start (rising edge)")
    ax1.plot(f, -A["v_end"], color="indianred", label="V_end (falling edge)")
    ax1.axhline(0.4, color="seagreen", ls="--", lw=0.8)
    ax1.axhline(-0.4, color="indianred", ls="--", lw=0.8)
    ax1.set_ylabel("vote V(f)"); ax1.set_xlabel("Frequency (MHz)")
    ax1.grid(True, alpha=0.3); ax1.legend(loc="lower right", fontsize=8)
    ax1.set_xlim(f[0], f[-1])
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def fig_roc(roc_rows, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for r in roc_rows:
        ax.scatter(r["FP"], r["recall"] * 100, s=90, zorder=3)
        ax.annotate(f"{r['preset']}\n({r['TP']}/{r['n_cucs']}, {r['FP']} FA)",
                    (r["FP"], r["recall"] * 100), textcoords="offset points",
                    xytext=(8, -4), fontsize=9)
    xs = [r["FP"] for r in roc_rows]; ys = [r["recall"] * 100 for r in roc_rows]
    ax.plot(xs, ys, color="gray", ls="--", lw=1, zorder=2)
    ax.set_xlabel("False alarms (count, 10 scenarios)")
    ax.set_ylabel("CUC recall (%)")
    ax.set_title("Detector operating curve across sensitivity presets")
    ax.grid(True, alpha=0.3); ax.set_ylim(0, 100); ax.set_xlim(-0.5, max(xs) + 1.5)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def fig_summary(rows, path):
    names = [r["experiment"] for r in rows]
    tp = [r["cuc_TP"] for r in rows]; fn = [r["cuc_FN"] for r in rows]
    fp = [r["cuc_FP"] for r in rows]
    x = np.arange(len(names)); w = 0.6
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x, tp, w, label="CUC detected (TP)", color="mediumseagreen")
    ax.bar(x, fn, w, bottom=tp, label="CUC missed (FN)", color="gold")
    ax.bar(x, fp, w, bottom=[t + n for t, n in zip(tp, fn)],
           label="False alarms (FP)", color="indianred")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("count"); ax.set_title(
        f"Per-experiment detection outcomes (sensitivity={DEFAULT_SENS})")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def temporal_analyze(man, N=TEMPORAL_N):
    """Temporal layer per experiment: difference current vs clean-baseline twin.
    Returns the residual + classified detections + scores, or None if no baseline."""
    base = man.get("baseline_dataset")
    if not base or not os.path.exists(base):
        return None
    freqs, cur = mf.integrated_psd(man["dataset"], N)
    _, bas = mf.integrated_psd(base, N)
    dets, resid, thr, _ = mf.detect_cuc_baseline(freqs, cur, bas)
    cucs = [c["fc_mhz"] for c in man["cucs"]]
    for d in dets:
        d["is_tp"] = any(abs(d["freq_mhz"] - fc) <= MATCH_TOL_MHZ for fc in cucs)
    detected_fcs = {fc for fc in cucs
                    if any(abs(d["freq_mhz"] - fc) <= MATCH_TOL_MHZ for d in dets)}
    tp = len(detected_fcs)
    fp = sum(1 for d in dets if not d["is_tp"])
    return dict(freqs=freqs, resid=resid, thr=thr, dets=dets, cucs=cucs,
                detected_fcs=detected_fcs, n_cucs=len(cucs),
                TP=tp, FN=len(cucs) - tp, FP=fp)


def fig_residual(man, T, path):
    """Per-experiment temporal view: baseline-difference residual with detections."""
    f, resid, thr = T["freqs"], T["resid"], T["thr"]
    lo, hi = min(resid.min(), -0.2), max(resid.max(), thr) + 0.2
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(f, resid, color="purple", lw=1.0,
            label="baseline-difference residual (band-integrated)")
    ax.axhline(thr, color="orange", ls="--", lw=1.2, label="detection threshold")
    ax.axhline(0, color="k", lw=0.5)
    seen = set()
    for d in T["dets"]:
        col = "green" if d["is_tp"] else "red"
        lab = "detected CUC (TP)" if d["is_tp"] else "false positive"
        yv = resid[int(np.argmin(np.abs(f - d["freq_mhz"])))]
        ax.plot(d["freq_mhz"], yv, "v", color=col, ms=11,
                label=lab if lab not in seen else None)
        seen.add(lab)
    ytext = hi - 0.06 * (hi - lo)
    for fc in T["cucs"]:
        found = fc in T["detected_fcs"]
        col = "green" if found else "crimson"
        ax.axvline(fc, color=col, ls=":", lw=1.3)
        ax.annotate(f"CUC {'OK' if found else 'MISS'}\n{fc:+.1f}", xy=(fc, ytext),
                    color=col, fontsize=8, ha="center", va="top", fontweight="bold")
    ax.set_xlim(f[0], f[-1]); ax.set_ylim(lo, hi)
    ax.set_xlabel("Frequency (MHz)"); ax.set_ylabel("Excess over baseline (dB)")
    ax.set_title(f"{man['name']} — temporal baseline-difference residual (N={TEMPORAL_N})  |  "
                 f"CUC detected {T['TP']}/{T['n_cucs']}  missed {T['FN']}  false+ {T['FP']}\n"
                 f"(assumes a current, CUC-free reference capture of the same band)",
                 fontsize=10)
    ax.grid(True, alpha=0.3); ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def fig_layers(rows, path):
    """Grouped bars: snapshot TP vs temporal TP per experiment (out of n_cucs)."""
    names = [r["experiment"] for r in rows]
    x = np.arange(len(names)); w = 0.38
    sTP = [r["snap_TP"] for r in rows]; tTP = [r["temp_TP"] for r in rows]
    ncu = [r["n_cucs"] for r in rows]
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - w/2, sTP, w, label="snapshot bump (no prior)", color="steelblue")
    ax.bar(x + w/2, tTP, w, label="temporal baseline-diff (needs clean ref)",
           color="mediumseagreen")
    ax.plot(x, ncu, "k_", ms=18, mew=2, label="CUCs present")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("CUCs detected")
    ax.set_title("Snapshot vs temporal layer — CUC detection per experiment\n"
                 "(temporal layer assumes a current, CUC-free reference + long "
                 "stationary dwell; snapshot needs no prior)", fontsize=10)
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUT_FIG, exist_ok=True)
    os.makedirs(os.path.join(OUT_CSV, "carriers"), exist_ok=True)
    mans = [json.load(open(m)) for m in sorted(glob.glob(os.path.join(EXP_DIR, "E*.json")))]
    if not mans:
        raise SystemExit("No manifests; run experiment_scenarios.py first.")

    summary_rows, cuc_detail_rows = [], []
    temp_results = {}

    for man in mans:
        A = analyze(man, DEFAULT_SENS)
        sc = score(man, A["carriers"])
        cov = coverage(A["freqs"], A["carriers"], sc["gt_legit"])

        # ---- per-experiment SNAPSHOT spectrum figure ----
        fig_spectrum(man, A, sc, cov, os.path.join(OUT_FIG, f"{man['name']}_spectrum.png"))

        # ---- per-experiment TEMPORAL baseline-difference residual figure ----
        T = temporal_analyze(man)
        temp_results[man["name"]] = T
        if T is not None:
            fig_residual(man, T, os.path.join(OUT_FIG, f"{man['name']}_residual.png"))

        # ---- per-experiment carrier CSV ----
        f = A["freqs"]
        with open(os.path.join(OUT_CSV, "carriers", f"{man['name']}.csv"), "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["fc_mhz", "bw_mhz", "start_mhz", "end_mhz", "variance",
                         "flagged", "bump_freq_mhz", "classification",
                         "nearest_truth_fc_mhz", "nearest_truth_is_cuc"])
            for c in A["carriers"]:
                ctr = c["fc"]
                near = min(man["carriers"], key=lambda g: abs(g["fc_mhz"] - ctr))
                if c["spoofed_flag"]:
                    is_tp = any(abs(c["bump_freq"] - t["fc_mhz"]) <= MATCH_TOL_MHZ
                                for t in man["cucs"])
                    cls = "CUC_detected_TP" if is_tp else "false_positive_FP"
                else:
                    cls = "authorized"
                wr.writerow([f"{c['fc']:.3f}", f"{c['bw']:.3f}",
                             f"{f[c['start_idx']]:.3f}", f"{f[c['end_idx']]:.3f}",
                             f"{c['variance']:.3f}", int(c["spoofed_flag"]),
                             "" if c["bump_freq"] is None else f"{c['bump_freq']:.3f}",
                             cls, f"{near['fc_mhz']:.3f}", int(near["is_cuc"])])

        # ---- CUC detail rows ----
        det_map = {round(cuc["fc_mhz"], 3): b for cuc, b in sc["detected"]}
        for cuc in man["cucs"]:
            fc = cuc["fc_mhz"]; b = det_map.get(round(fc, 3))
            cuc_detail_rows.append([man["name"], f"{fc:.3f}", f"{cuc['power_db']:.1f}",
                                    f"{cuc['bw_mhz']:.3f}", int(b is not None),
                                    "" if b is None else f"{b:.3f}",
                                    "" if b is None else f"{abs(b - fc):.3f}"])

        # ---- summary metrics row ----
        tp, fn, fp = len(sc["detected"]), len(sc["missed"]), len(sc["false_pos"])
        rec, prec, f1 = prf(tp, fn, fp)
        summary_rows.append(dict(
            experiment=man["name"], n_gt_carriers=len(sc["gt_legit"]),
            n_segments=len(A["carriers"]), carriers_covered=cov[0],
            coverage_pct=100.0 * cov[0] / cov[1] if cov[1] else 100.0,
            n_cucs=sc["n_cucs"], cuc_TP=tp, cuc_FN=fn, cuc_FP=fp,
            recall_pct=None if rec is None else 100 * rec,
            precision_pct=None if prec is None else 100 * prec,
            f1=f1, mean_loc_err_mhz=float(np.mean(sc["loc_err"])) if sc["loc_err"] else None))

    # ---- methodology figure (baseline) ----
    base = next(m for m in mans if m["name"].startswith("E01"))
    fig_methodology(base, analyze(base, DEFAULT_SENS),
                    os.path.join(OUT_FIG, "_methodology_E01.png"))

    # ---- sensitivity sweep (ROC) ----
    roc_rows = []
    roc_by_exp = []
    for sens in PRESETS:
        tp = fn = fp = ncuc = 0
        for man in mans:
            A = analyze(man, sens); sc = score(man, A["carriers"])
            etp, efn, efp = len(sc["detected"]), len(sc["missed"]), len(sc["false_pos"])
            tp += etp; fn += efn; fp += efp; ncuc += sc["n_cucs"]
            roc_by_exp.append([man["name"], sens, etp, efn, efp])
        rec, prec, f1 = prf(tp, fn, fp)
        p = cca.SENSITIVITY_PRESETS[sens]
        roc_rows.append(dict(preset=sens, TP=tp, FN=fn, FP=fp, n_cucs=ncuc,
                             recall=rec or 0.0, precision=prec or 0.0, f1=f1 or 0.0,
                             **p))

    fig_roc(roc_rows, os.path.join(OUT_FIG, "_roc.png"))
    fig_summary(summary_rows, os.path.join(OUT_FIG, "_summary_outcomes.png"))

    # ---- snapshot vs temporal (baseline-differencing) layer comparison ----
    layer_rows = []
    snap_by_name = {r["experiment"]: r for r in summary_rows}  # conservative snapshot
    for man in mans:
        s = snap_by_name[man["name"]]
        t = temp_results[man["name"]]
        layer_rows.append(dict(
            experiment=man["name"], n_cucs=s["n_cucs"],
            snap_TP=s["cuc_TP"], snap_FP=s["cuc_FP"],
            temp_TP=(t["TP"] if t else 0), temp_FN=(t["FN"] if t else s["n_cucs"]),
            temp_FP=(t["FP"] if t else 0), has_baseline=int(t is not None)))
    fig_layers(layer_rows, os.path.join(OUT_FIG, "_layers_comparison.png"))

    # ---- CSVs ----
    def w_csv(name, header, rows):
        with open(os.path.join(OUT_CSV, name), "w", newline="") as fh:
            wr = csv.writer(fh); wr.writerow(header); wr.writerows(rows)

    def fmt(v):
        return "" if v is None else (f"{v:.2f}" if isinstance(v, float) else v)

    w_csv("summary_metrics.csv",
          ["experiment", "n_gt_carriers", "n_segments", "carriers_covered",
           "coverage_pct", "n_cucs", "cuc_TP", "cuc_FN", "cuc_FP",
           "recall_pct", "precision_pct", "f1", "mean_loc_err_mhz"],
          [[r["experiment"], r["n_gt_carriers"], r["n_segments"], r["carriers_covered"],
            fmt(r["coverage_pct"]), r["n_cucs"], r["cuc_TP"], r["cuc_FN"], r["cuc_FP"],
            fmt(r["recall_pct"]), fmt(r["precision_pct"]), fmt(r["f1"]),
            fmt(r["mean_loc_err_mhz"])] for r in summary_rows])

    w_csv("cuc_detail.csv",
          ["experiment", "true_fc_mhz", "power_db", "bw_mhz", "detected",
           "bump_freq_mhz", "loc_err_mhz"], cuc_detail_rows)

    w_csv("sensitivity_roc.csv",
          ["preset", "height_db", "prom_db", "wmin_mhz", "wmax_mhz",
           "TP", "FN", "FP", "n_cucs", "recall_pct", "precision_pct", "f1"],
          [[r["preset"], r["height_db"], r["prom_db"], r["wmin_mhz"], r["wmax_mhz"],
            r["TP"], r["FN"], r["FP"], r["n_cucs"], f"{100*r['recall']:.1f}",
            f"{100*r['precision']:.1f}", f"{r['f1']:.3f}"] for r in roc_rows])

    w_csv("roc_by_experiment.csv",
          ["experiment", "preset", "TP", "FN", "FP"], roc_by_exp)

    w_csv("layer_comparison.csv",
          ["experiment", "n_cucs", "snapshot_TP", "snapshot_FP",
           "temporal_TP", "temporal_FN", "temporal_FP", "has_baseline",
           f"temporal_N={TEMPORAL_N}"],
          [[r["experiment"], r["n_cucs"], r["snap_TP"], r["snap_FP"],
            r["temp_TP"], r["temp_FN"], r["temp_FP"], r["has_baseline"], TEMPORAL_N]
           for r in layer_rows])

    # ---- console recap ----
    print(f"[+] Figures  -> {OUT_FIG}/  ({len(mans)} spectra + methodology + roc + summary)")
    print(f"[+] Metrics  -> {OUT_CSV}/  (summary_metrics, cuc_detail, sensitivity_roc, "
          f"roc_by_experiment, layer_comparison, carriers/*.csv)")
    print("\nPooled CUC metrics by sensitivity:")
    for r in roc_rows:
        print(f"  {r['preset']:<12} TP={r['TP']:>2}/{r['n_cucs']}  FN={r['FN']:>2}  "
              f"FP={r['FP']:>2}  recall={100*r['recall']:5.1f}%  "
              f"precision={100*r['precision']:5.1f}%  F1={r['f1']:.2f}")
    sT = sum(r["snap_TP"] for r in layer_rows); tT = sum(r["temp_TP"] for r in layer_rows)
    nC = sum(r["n_cucs"] for r in layer_rows)
    tF = sum(r["temp_FP"] for r in layer_rows); sF = sum(r["snap_FP"] for r in layer_rows)
    print(f"\nLayer comparison (CUCs={nC}):  snapshot {sT}/{nC} ({sF} FP)  ->  "
          f"temporal {tT}/{nC} ({tF} FP, N={TEMPORAL_N})")
    print("  NOTE: temporal layer assumes a current, CUC-free reference capture + long")
    print("  stationary dwell; a CUC already in the baseline cancels out, and reference")
    print("  drift would surface as false alarms. Snapshot layer needs no such prior.")


if __name__ == "__main__":
    main()
