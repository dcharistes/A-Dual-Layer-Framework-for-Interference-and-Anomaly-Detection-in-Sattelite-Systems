#!/usr/bin/env python3
"""
Run the dual-layer detector across all generated experiments and score it
against each scenario's ground-truth manifest.

For every experiment it reports:
  - carriers segmented,
  - CUCs detected / total (a CUC counts as detected if a flagged segment's
    bump frequency lands within MATCH_TOL_MHZ of the true CUC centre),
  - false positives (flagged segments not matching any true CUC).

Generate the datasets first:
    python3 transponder_gnuradio/experiment_scenarios.py
Then score (from the project root):
    PYTHONPATH=src python3 src/run_experiments.py
"""
import glob
import json
import os

from rf_interference import wavelet_dsp
from rf_interference import carrier_cuc_analysis as cca
from rf_interference import multiframe_cuc as mf
from rf_interference.psd import welch_psd_db

EXP_DIR = "./data/experiments"
MATCH_TOL_MHZ = 1.0
WEIGHT_MODE = "entropy"
NUM_CHUNKS = 400   # more Welch averaging -> suppresses estimation-variance bumps
TEMPORAL_N = 2048  # frames integrated for the baseline-differencing layer


DEFAULT_SENS = "conservative"
PRESETS = list(cca.SENSITIVITY_PRESETS)   # conservative, balanced, high

# Cache the (expensive) spectrum + wavelet votes per dataset; only the cheap
# bump pass changes between sensitivity presets.
_SPEC_CACHE = {}


def _spectrum(dataset):
    if dataset not in _SPEC_CACHE:
        freqs, power_db = welch_psd_db(dataset, num_chunks=NUM_CHUNKS)
        clean, v_start, v_end = wavelet_dsp.detect_boundaries_advanced(
            freqs, power_db, weight_mode=WEIGHT_MODE, verbose=False)
        _SPEC_CACHE[dataset] = (freqs, power_db, clean, v_start, v_end)
    return _SPEC_CACHE[dataset]


def evaluate(manifest, sensitivity):
    freqs, power_db, clean, v_start, v_end = _spectrum(manifest["dataset"])
    carriers = cca.extract_and_analyze_carriers(
        freqs, power_db, v_start, v_end, clean,
        sensitivity=sensitivity, verbose=False)

    flagged = [c["bump_freq"] for c in carriers if c["spoofed_flag"]]
    true_cucs = [c["fc_mhz"] for c in manifest["cucs"]]

    detected, missed = [], []
    for fc in true_cucs:
        hit = next((b for b in flagged if abs(b - fc) <= MATCH_TOL_MHZ), None)
        (detected if hit is not None else missed).append(fc)
    false_pos = [b for b in flagged
                 if all(abs(b - fc) > MATCH_TOL_MHZ for fc in true_cucs)]

    return {
        "n_carriers": len(carriers), "n_cucs": len(true_cucs),
        "detected": detected, "missed": missed, "false_pos": false_pos,
        "flagged": flagged,
    }


def evaluate_temporal(manifest, N=TEMPORAL_N):
    """Temporal layer: detect CUCs by differencing the N-frame-averaged current
    spectrum against the N-frame-averaged clean baseline twin. Returns None if
    no baseline capture is available for this scenario."""
    base_path = manifest.get("baseline_dataset")
    if not base_path or not os.path.exists(base_path):
        return None
    freqs, cur = mf.integrated_psd(manifest["dataset"], N)
    _, base = mf.integrated_psd(base_path, N)
    dets, _, _, _ = mf.detect_cuc_baseline(freqs, cur, base)
    true_cucs = [c["fc_mhz"] for c in manifest["cucs"]]
    detected, missed = [], []
    for fc in true_cucs:
        hit = any(abs(d["freq_mhz"] - fc) <= MATCH_TOL_MHZ for d in dets)
        (detected if hit else missed).append(fc)
    false_pos = [d for d in dets
                 if all(abs(d["freq_mhz"] - fc) > MATCH_TOL_MHZ for fc in true_cucs)]
    return {"n_cucs": len(true_cucs), "detected": detected,
            "missed": missed, "false_pos": false_pos}


def main():
    manifests = sorted(glob.glob(os.path.join(EXP_DIR, "E*.json")))
    if not manifests:
        raise SystemExit(f"No manifests in {EXP_DIR}. "
                         "Run transponder_gnuradio/experiment_scenarios.py first.")
    mans = [json.load(open(m)) for m in manifests]

    # ---- Per-experiment scoreboard at the operational default ----
    print("=" * 84)
    print(f" DUAL-LAYER DETECTOR -- SCOREBOARD  (weights={WEIGHT_MODE}, "
          f"sensitivity={DEFAULT_SENS}, match tol {MATCH_TOL_MHZ} MHz)")
    print("=" * 84)
    print(f"{'experiment':<26}{'carriers':>9}{'CUC det':>9}{'missed':>8}"
          f"{'false+':>8}   result")
    print("-" * 84)
    tot_cuc = tot_det = tot_fp = 0
    rows = []
    for man in mans:
        r = evaluate(man, DEFAULT_SENS)
        tot_cuc += r["n_cucs"]; tot_det += len(r["detected"]); tot_fp += len(r["false_pos"])
        verdict = "PASS" if (not r["missed"] and not r["false_pos"]) else \
                  ("MISS" if r["missed"] else "FALSE+")
        print(f"{man['name']:<26}{r['n_carriers']:>9}"
              f"{len(r['detected']):>5}/{r['n_cucs']:<3}{len(r['missed']):>8}"
              f"{len(r['false_pos']):>8}   {verdict}")
        rows.append((man, r))
    print("-" * 84)
    print(f"{'TOTAL':<26}{'':>9}{tot_det:>5}/{tot_cuc:<3}"
          f"{tot_cuc - tot_det:>8}{tot_fp:>8}")

    # ---- ROC-style sweep across sensitivity presets ----
    print("\n" + "=" * 84)
    print(" SENSITIVITY SWEEP (ROC) -- all 10 scenarios pooled")
    print("=" * 84)
    print(f"{'preset':<14}{'CUC detected':>14}{'missed':>9}{'false+':>9}"
          f"{'recall':>9}{'  detail (missed CUCs)'}")
    print("-" * 84)
    for sens in PRESETS:
        det = miss = fp = ncuc = 0
        missed_locs = []
        for man in mans:
            r = evaluate(man, sens)
            det += len(r["detected"]); miss += len(r["missed"]); fp += len(r["false_pos"])
            ncuc += r["n_cucs"]
            missed_locs += [f"{man['name'].split('_')[0]}@{m:+.0f}" for m in r["missed"]]
        recall = det / ncuc if ncuc else 0.0
        print(f"{sens:<14}{det:>9}/{ncuc:<4}{miss:>9}{fp:>9}{recall:>8.0%}"
              f"  {', '.join(missed_locs)}")
    print("-" * 84)
    print(" conservative: ops default (no false alarms) | "
          "high: recovers narrowband-host CUCs at the cost of false alarms")

    # ---- Combined: snapshot layer vs temporal baseline-differencing layer ----
    print("\n" + "=" * 84)
    print(f" COMBINED LAYERS  --  snapshot bump ({DEFAULT_SENS})  vs  "
          f"temporal baseline-diff (N={TEMPORAL_N})")
    print("=" * 84)
    print(f"{'':<26}{'snapshot':>14}{'':>4}{'temporal':>14}")
    print(f"{'experiment':<26}{'det  miss  FP':>16}{'det  miss  FP':>18}")
    print("-" * 84)
    s_tot = [0, 0, 0]; t_tot = [0, 0, 0]; ncuc = 0
    for man in mans:
        s = evaluate(man, DEFAULT_SENS)
        t = evaluate_temporal(man)
        ncuc += s["n_cucs"]
        sd, sm, sf = len(s["detected"]), len(s["missed"]), len(s["false_pos"])
        s_tot = [s_tot[0]+sd, s_tot[1]+sm, s_tot[2]+sf]
        if t is None:
            tcol = "   (no baseline) "
        else:
            td, tm, tf = len(t["detected"]), len(t["missed"]), len(t["false_pos"])
            t_tot = [t_tot[0]+td, t_tot[1]+tm, t_tot[2]+tf]
            tcol = f"{td:>5}{tm:>6}{tf:>6}"
        print(f"{man['name']:<26}{sd:>5}{sm:>6}{sf:>6}    {tcol}")
    print("-" * 84)
    print(f"{'TOTAL  (CUCs='+str(ncuc)+')':<26}"
          f"{s_tot[0]:>5}{s_tot[1]:>6}{s_tot[2]:>6}    "
          f"{t_tot[0]:>5}{t_tot[1]:>6}{t_tot[2]:>6}")
    print("-" * 84)
    print(" CAVEAT: the snapshot layer needs NO prior -- one capture, catches strong CUCs.")
    print(" The temporal layer requires a CURRENT, CUC-FREE reference capture of the same")
    print(" band + a long stationary dwell; a pre-existing or persistent CUC present in")
    print(" the baseline cancels out and is missed, and reference/channel drift shows up")
    print(" as false alarms. Its 0-FP score assumes a drift-free reference. The two")
    print(" columns therefore have DIFFERENT operational prerequisites.")


if __name__ == "__main__":
    main()
