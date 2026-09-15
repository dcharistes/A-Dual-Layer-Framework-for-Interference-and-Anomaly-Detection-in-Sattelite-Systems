#!/usr/bin/env python3
"""
Stress the temporal baseline-differencing layer against a MISMATCHED reference.

The temporal layer's 0-false-alarm score assumes the clean baseline matches the
current band exactly. Operationally the baseline comes from an earlier time and
will have drifted. This script quantifies how much drift the layer tolerates
before the un-cancelled host structure turns into false alarms -- i.e. how much
"host structure" enters the real floor as a function of reference mismatch.

Three drifts, applied in closed form to the baseline PSD (no regeneration):
  - GAIN  drift  (+dG dB uniform)            : a flat level change.
  - TILT  drift  (+dT dB end-to-end ramp)    : transponder gain-slope change.
  - FREQ  drift  (+dF kHz LO/Doppler shift)  : carriers move between captures.

For each magnitude we record whether the real CUC is still detected and how many
false alarms the drift creates, then plot false-alarms-vs-drift per type.

Run:  PYTHONPATH=src python3 src/baseline_stress.py
"""
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rf_interference import multiframe_cuc as mf

OUT_FIG = "./results/figures/rf"
OUT_CSV = "./results/metrics/rf"
SCENARIO = "E04_uniform_control"   # 1 CUC @ +2, 8 identical authorized carriers
N = 2048
TOL = 1.0
HALF_BW_MHZ = 20.0


def perturb(freqs, base_db, gain_db=0.0, tilt_db=0.0, freq_khz=0.0):
    out = base_db + gain_db
    if tilt_db:
        out = out + tilt_db * (freqs / (2 * HALF_BW_MHZ))   # -dT/2..+dT/2
    if freq_khz:
        df_mhz = freq_khz / 1000.0
        out = np.interp(freqs - df_mhz, freqs, out)         # shift carriers by +dF
    return out


def score(freqs, cur, base_db, cuc_fcs, robust=False):
    dets, _, _, _ = mf.detect_cuc_baseline(freqs, cur, base_db, robust=robust)
    detected = sum(any(abs(d["freq_mhz"] - fc) <= TOL for d in dets) for fc in cuc_fcs)
    fp = sum(all(abs(d["freq_mhz"] - fc) > TOL for fc in cuc_fcs) for d in dets)
    return detected, fp


def plan_change_study(freqs, cur, base, cuc_fcs):
    """PLAN changes (a carrier's power changed, or a carrier removed) between
    baseline and now. Unlike gain/tilt/freq drift these are LOCALIZED and are
    genuine anomalies relative to the authorized plan -- robust mode should NOT
    (and does not) suppress them. We perturb one authorized carrier (-6 MHz)."""
    df = freqs[1] - freqs[0]
    fc0 = -6.0; halfbw = 0.675   # the 1 Msym carrier at -6 MHz
    band = (freqs > fc0 - halfbw) & (freqs < fc0 + halfbw)
    floor = np.median(base[base < np.percentile(base, 40)])
    rows = []
    print("\nPlan-change sensitivity (authorized carrier @ -6 MHz):")
    # The layer flags ADDED power (the CUC / unauthorized-emission model). Model
    # "carrier stronger now than at baseline" by lowering the baseline band.
    for dp in [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]:
        b = base.copy(); b[band] -= dp
        dets, _, _, _ = mf.detect_cuc_baseline(freqs, cur, b, robust=True)
        flagged = any(abs(d["freq_mhz"] - fc0) <= TOL for d in dets)
        rows.append(["power_increase_dB", dp, int(flagged)])
        print(f"   carrier +{dp:.1f} dB vs baseline -> flagged: {flagged}")
    b = base.copy(); b[band] = floor      # carrier absent in baseline
    dets, _, _, _ = mf.detect_cuc_baseline(freqs, cur, b, robust=True)
    rows.append(["carrier_appeared", "n/a",
                 int(any(abs(d["freq_mhz"] - fc0) <= TOL for d in dets))])
    print(f"   carrier present now, absent at baseline -> flagged: {rows[-1][2]==1}")
    print("   (correct anomaly detections, not false alarms -- the layer flags")
    print("    ADDED power, so the baseline must track the authorized-carrier plan;")
    print("    a carrier that only got WEAKER than baseline is not flagged.)")
    with open(os.path.join(OUT_CSV, "baseline_stress_planchange.csv"), "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["change", "magnitude", "flagged"]); wr.writerows(rows)


def main():
    os.makedirs(OUT_FIG, exist_ok=True); os.makedirs(OUT_CSV, exist_ok=True)
    man = json.load(open(f"./data/experiments/{SCENARIO}.json"))
    cuc_fcs = [c["fc_mhz"] for c in man["cucs"]]
    freqs, cur = mf.integrated_psd(man["dataset"], N)
    _, base = mf.integrated_psd(man["baseline_dataset"], N)

    sweeps = {
        "gain (dB)":  ("gain_db",  [0, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0]),
        "tilt (dB)":  ("tilt_db",  [0, 0.2, 0.5, 1.0, 2.0, 4.0, 6.0]),
        "freq (kHz)": ("freq_khz", [0, 1, 2, 3, 5, 10, 20, 50]),
    }
    results = {}
    rows = []
    for label, (kw, vals) in sweeps.items():
        raw, rob = ([], []), ([], [])
        for v in vals:
            b = perturb(freqs, base, **{kw: v})
            d0, f0 = score(freqs, cur, b, cuc_fcs, robust=False)
            d1, f1 = score(freqs, cur, b, cuc_fcs, robust=True)
            raw[0].append(d0); raw[1].append(f0)
            rob[0].append(d1); rob[1].append(f1)
            rows.append([label, v, d0, f0, d1, f1, len(cuc_fcs)])
        results[label] = (vals, raw, rob)

    # ---- figure: false alarms vs drift (3 panels), raw vs robust ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (label, (vals, raw, rob)) in zip(axes, results.items()):
        ax.plot(vals, raw[1], "o-", color="indianred", label="false alarms (raw)")
        ax.plot(vals, rob[1], "s--", color="seagreen", label="false alarms (robust)")
        ax.set_xlabel(f"baseline {label} drift"); ax.set_ylabel("false alarms")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(f"Temporal layer tolerance to baseline drift ({SCENARIO}, N={N})\n"
                 "RAW vs ROBUST (frequency-align + linear detrend). Robust mode "
                 "absorbs gain/tilt and aligns frequency, hugely widening tolerance.",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_FIG, "_baseline_stress.png"), dpi=110)
    plt.close(fig)

    with open(os.path.join(OUT_CSV, "baseline_stress.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["drift_type", "magnitude", "raw_cuc_det", "raw_FP",
                     "robust_cuc_det", "robust_FP", "n_cucs"])
        wr.writerows(rows)

    # ---- console recap: first-false-alarm onset, raw vs robust ----
    print(f"Baseline-drift tolerance ({SCENARIO}):  raw  ->  robust")
    for label, (vals, raw, rob) in results.items():
        on_raw = next((v for v, f in zip(vals, raw[1]) if f > 0), None)
        on_rob = next((v for v, f in zip(vals, rob[1]) if f > 0), None)
        lost_raw = next((v for v, d in zip(vals, raw[0]) if d < len(cuc_fcs)), None)
        lost_rob = next((v for v, d in zip(vals, rob[0]) if d < len(cuc_fcs)), None)
        print(f"  {label:<12} first FP: {on_raw} -> {on_rob}   |   "
              f"CUC lost: {lost_raw} -> {lost_rob}")
    plan_change_study(freqs, cur, base, cuc_fcs)

    print(f"\n[+] figure -> {OUT_FIG}/_baseline_stress.png")
    print(f"[+] csv    -> {OUT_CSV}/baseline_stress.csv, baseline_stress_planchange.csv")


if __name__ == "__main__":
    main()
