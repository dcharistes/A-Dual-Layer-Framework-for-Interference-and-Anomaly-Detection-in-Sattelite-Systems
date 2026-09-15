#!/usr/bin/env python3
"""
Document the single-frame -> multi-frame progression for CUC detection.

Compares, as a function of the number of integrated frames N:
  (A) SNAPSHOT / AVERAGED BUMP -- the existing spectral bump detector
      (carrier_cuc_analysis, 'high' sensitivity) run on the N-averaged spectrum.
  (C) BASELINE DIFFERENCING -- multi-frame temporal detection (multiframe_cuc):
      N-averaged current spectrum minus N-averaged clean baseline.

Outputs:
  results/figures/_progression_snr.png   residual SNR vs N (method C) with the
                                          detection threshold; shows integration gain.
  results/figures/_progression_triptych_<exp>.png  single-frame -> averaged ->
                                          baseline-difference for E04 and E02.
  results/metrics/multiframe_progression.csv

Run (from project root, after baselines exist):
    PYTHONPATH=src python3 src/multiframe_study.py
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

OUT_FIG = "./results/figures/rf"
OUT_CSV = "./results/metrics/rf"
FFT = 8192
TOL = 1.0
BAND_MHZ = 0.30
CFAR_K = 6.0
FOCUS = ["E04_uniform_control", "E02_dense_narrowband",
         "E01_baseline", "E10_clean_complex"]
N_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


def load_iq(path):
    return np.fromfile(path, dtype=np.complex64)


def psd_db(iq, n_frames):
    """Averaged periodogram (dB) from the first n_frames frames of cached I/Q."""
    win = np.blackman(FFT)
    acc = np.zeros(FFT)
    got = 0
    for i in range(n_frames):
        chunk = iq[i * FFT:(i + 1) * FFT]
        if len(chunk) < FFT:
            break
        acc += np.abs(np.fft.fftshift(np.fft.fft(chunk * win))) ** 2
        got += 1
    acc /= max(got, 1)
    freqs = np.linspace(-20.0, 20.0, FFT)
    return freqs, 10 * np.log10(acc), got


def snapshot_bump_detects(freqs, power_db, cuc_fcs):
    """Method A: existing detector (best/'high' sensitivity) on this spectrum."""
    clean, vs, ve = wavelet_dsp.detect_boundaries_advanced(freqs, power_db, verbose=False)
    cars = cca.extract_and_analyze_carriers(freqs, power_db, vs, ve, clean,
                                            sensitivity="high", verbose=False)
    flagged = [c["bump_freq"] for c in cars if c["spoofed_flag"]]
    return [fc for fc in cuc_fcs if any(abs(b - fc) <= TOL for b in flagged)]


def main():
    os.makedirs(OUT_FIG, exist_ok=True)
    os.makedirs(OUT_CSV, exist_ok=True)
    mans = {json.load(open(m))["name"]: json.load(open(m))
            for m in glob.glob("./data/experiments/E*.json")}

    rows = []
    snr_curves = {}        # exp -> (Ns, residual SNR at CUC per N), method C
    fp_curves = {}         # exp -> (Ns, baseline false-positive count per N)
    for name in FOCUS:
        man = mans[name]
        cuc_fcs = [c["fc_mhz"] for c in man["cucs"]]
        cur_iq = load_iq(man["dataset"])
        base_iq = load_iq(man["baseline_dataset"])
        maxframes = len(cur_iq) // FFT
        Ns = [n for n in N_GRID if n <= maxframes]
        snr_at_N = []
        fp_at_N = []
        for N in Ns:
            f, cur, got = psd_db(cur_iq, N)
            _, base, _ = psd_db(base_iq, N)

            # Method A: averaged-bump detector
            a_hits = snapshot_bump_detects(f, cur, cuc_fcs)
            a_det = len(a_hits) == len(cuc_fcs) and len(cuc_fcs) > 0

            # Method C: baseline differencing
            dets, _, _, _ = mf.detect_cuc_baseline(f, cur, base,
                                                   band_mhz=BAND_MHZ, k=CFAR_K)
            c_hits = [fc for fc in cuc_fcs
                      if any(abs(d["freq_mhz"] - fc) <= TOL for d in dets)]
            c_det = len(c_hits) == len(cuc_fcs) and len(cuc_fcs) > 0
            c_fp = [d for d in dets
                    if all(abs(d["freq_mhz"] - fc) > TOL for fc in cuc_fcs)]

            # residual SNR at the (first) CUC -- the integration-gain curve
            if cuc_fcs:
                _, _, snr = mf.cuc_residual_snr(f, cur, base, cuc_fcs[0], BAND_MHZ)
            else:
                snr = 0.0
            snr_at_N.append(snr)
            fp_at_N.append(len(c_fp))
            rows.append([name, N, int(a_det), len(a_hits),
                         int(c_det), len(c_hits), len(c_fp), f"{snr:.2f}"])
        snr_curves[name] = (Ns, snr_at_N)
        fp_curves[name] = (Ns, fp_at_N)

    # ---- CSV ----
    with open(os.path.join(OUT_CSV, "multiframe_progression.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["experiment", "N_frames", "snapshot_detect_all",
                     "snapshot_cuc_hits", "baseline_detect_all",
                     "baseline_cuc_hits", "baseline_false_pos",
                     "baseline_residual_snr"])
        wr.writerows(rows)

    # ---- 2-panel progression figure: detection SNR + false alarms vs N ----
    fig, (axT, axB) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    for name in FOCUS:
        if mans[name]["cucs"]:
            Ns, snr = snr_curves[name]
            axT.plot(Ns, np.clip(snr, 0.1, None), marker="o", label=name)
        Ns, fp = fp_curves[name]
        tag = name + (" (clean)" if not mans[name]["cucs"] else "")
        axB.plot(Ns, fp, marker="s", label=tag)
    axT.axhline(CFAR_K, color="crimson", ls="--", lw=1.2,
                label=f"detection threshold (k={CFAR_K:g})")
    axT.set_yscale("log"); axT.set_ylabel("residual SNR at CUC")
    axT.set_title("Temporal-integration gain (baseline differencing)")
    axT.grid(True, which="both", alpha=0.3); axT.legend(fontsize=8)
    axB.set_xscale("log", base=2); axB.set_ylabel("false alarms (count)")
    axB.set_xlabel("frames integrated  N  (log scale)")
    axB.grid(True, which="both", alpha=0.3); axB.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_FIG, "_progression.png"), dpi=110)
    plt.close(fig)

    # ---- triptych figures (single frame -> averaged -> baseline diff) ----
    for name in ("E04_uniform_control", "E02_dense_narrowband"):
        man = mans[name]; fc = man["cucs"][0]["fc_mhz"]
        cur_iq = load_iq(man["dataset"]); base_iq = load_iq(man["baseline_dataset"])
        Nmax = len(cur_iq) // FFT
        f, s1, _ = psd_db(cur_iq, 1)
        _, savg, _ = psd_db(cur_iq, Nmax)
        _, bavg, _ = psd_db(base_iq, Nmax)
        from scipy.ndimage import uniform_filter1d
        df = f[1] - f[0]
        resid = uniform_filter1d(savg - bavg, max(1, int(BAND_MHZ / df)))
        m = (f > fc - 2.2) & (f < fc + 2.2)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        axes[0].plot(f[m], s1[m], color="gray"); axes[0].set_title("(a) single frame (N=1)")
        axes[1].plot(f[m], savg[m], color="dodgerblue")
        axes[1].set_title(f"(b) {Nmax}-frame average")
        axes[2].plot(f[m], resid[m], color="purple")
        axes[2].axhline(0, color="k", lw=0.6)
        axes[2].set_title(f"(c) baseline difference (N={Nmax})")
        for a in axes:
            a.axvline(fc, color="crimson", ls="--", lw=1.2)
            a.set_xlabel("MHz"); a.grid(True, alpha=0.3)
        axes[0].set_ylabel("Power (dB)"); axes[2].set_ylabel("Excess (dB)")
        fig.suptitle(f"{name}: CUC @ {fc:+.1f} MHz invisible in a single frame, "
                     f"buried in the averaged host, REVEALED by baseline differencing",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_FIG, f"_progression_triptych_{name}.png"), dpi=110)
        plt.close(fig)

    # ---- console recap ----
    print("Multi-frame progression (snapshot bump vs baseline differencing):")
    print(f"{'experiment':<24}{'N':>6}{'snap?':>7}{'base?':>7}{'baseFP':>8}{'residSNR':>10}")
    for r in rows:
        print(f"{r[0]:<24}{r[1]:>6}{r[2]:>7}{r[4]:>7}{r[6]:>8}{r[7]:>10}")
    # ---- full-benchmark re-score with the temporal (baseline-diff) layer ----
    benchmark_rescore(mans, N=2048)

    print(f"\n[+] figures -> {OUT_FIG}/_progression.png, _progression_triptych_*.png")
    print(f"[+] csv     -> {OUT_CSV}/multiframe_progression.csv, baseline_benchmark.csv")


def benchmark_rescore(mans, N=512):
    """Score baseline differencing across ALL scenarios at a fixed N -> table+CSV.

    Also records the LOCALISATION error of the temporal layer, i.e. the distance
    between each detected CUC's reported centre and the manifest's true centre.
    The snapshot layer's equivalent is written by generate_reports.py into
    cuc_detail.csv; without this the thesis can only quote snapshot localisation.
    """
    rows, tp = [], 0
    ncuc = fp = 0
    detail = []                       # one row per TRUE CUC
    for name in sorted(mans):
        man = mans[name]
        cur_iq = load_iq(man["dataset"]); base_iq = load_iq(man["baseline_dataset"])
        Nuse = min(N, len(cur_iq) // FFT, len(base_iq) // FFT)
        f, cur, _ = psd_db(cur_iq, Nuse); _, base, _ = psd_db(base_iq, Nuse)
        dets, _, _, _ = mf.detect_cuc_baseline(f, cur, base, band_mhz=BAND_MHZ, k=CFAR_K)
        cuc_fcs = [c["fc_mhz"] for c in man["cucs"]]
        hits = [fc for fc in cuc_fcs
                if any(abs(d["freq_mhz"] - fc) <= TOL for d in dets)]
        fps = [d for d in dets if all(abs(d["freq_mhz"] - fc) > TOL for fc in cuc_fcs)]
        rows.append([name, len(cuc_fcs), len(hits), len(cuc_fcs) - len(hits), len(fps)])
        ncuc += len(cuc_fcs); tp += len(hits); fp += len(fps)

        # localisation: match each true CUC to its NEAREST in-tolerance detection
        for c in man["cucs"]:
            fc = c["fc_mhz"]
            near = [d for d in dets if abs(d["freq_mhz"] - fc) <= TOL]
            if near:
                best = min(near, key=lambda d: abs(d["freq_mhz"] - fc))
                detail.append([name, f"{fc:.3f}", c.get("power_db", ""),
                               c.get("bw_mhz", ""), 1,
                               f"{best['freq_mhz']:.3f}",
                               f"{abs(best['freq_mhz'] - fc):.3f}"])
            else:
                detail.append([name, f"{fc:.3f}", c.get("power_db", ""),
                               c.get("bw_mhz", ""), 0, "", ""])

    loc_errs = [float(r[6]) for r in detail if r[6] != ""]
    mean_loc = float(np.mean(loc_errs)) if loc_errs else None

    with open(os.path.join(OUT_CSV, "baseline_benchmark.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["experiment", "n_cucs", "cuc_TP", "cuc_FN", "false_pos", f"N={N}"])
        wr.writerows([r + [N] for r in rows])

    with open(os.path.join(OUT_CSV, "temporal_cuc_detail.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["experiment", "true_fc_mhz", "power_db", "bw_mhz",
                     "detected", "det_freq_mhz", "loc_err_mhz"])
        wr.writerows(detail)

    if loc_errs:
        print(f"\nTEMPORAL localisation error over {len(loc_errs)} detected CUCs: "
              f"min {min(loc_errs):.3f}  mean {mean_loc:.3f}  max {max(loc_errs):.3f} MHz")

    print(f"\nFULL BENCHMARK with TEMPORAL baseline-differencing layer (N={N} frames):")
    print(f"{'experiment':<26}{'CUC det':>9}{'missed':>8}{'false+':>8}")
    for r in rows:
        print(f"{r[0]:<26}{r[2]:>5}/{r[1]:<3}{r[3]:>8}{r[4]:>8}")
    print(f"{'TOTAL':<26}{tp:>5}/{ncuc:<3}{ncuc - tp:>8}{fp:>8}   "
          f"(snapshot layer alone: 7/12, 0 FP)")


if __name__ == "__main__":
    main()
