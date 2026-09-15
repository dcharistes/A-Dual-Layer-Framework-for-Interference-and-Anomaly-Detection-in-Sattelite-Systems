# A Dual-Layer Framework for Interference and Anomaly Detection in Satellite Systems

## Overview

This repository contains the code, results, and documentation for my diploma thesis.
The thesis directory is:
[`docs/A_Dual_Layer_Interference_Detection_System_for_Sattelite_Communications_thesis.pdf`](docs/A_Dual_Layer_Interference_Detection_System_for_Sattelite_Communications_thesis.pdf).

The thesis develops a **dual-domain detection system** for Frequency Division Multiple
Access (FDMA) satellite communication networks. A satellite mission can fail from
two directions at once: **from outside**, when an interfering RF emission corrupts
the communication link, and **from inside**, when a subsystem fault shows up as
anomalous housekeeping telemetry.

* **Layer 1 — RF interference detection** watches the transponder **spectrum**:
  multi-scale wavelet carrier segmentation plus Carrier-under-Carrier (CUC)
  detection, extended with a temporal baseline-differencing stage [1].
  Details and reasoning: `src/rf_interference/README.md`.
* **Layer 2 — telemetry anomaly detection** watches spacecraft **health**: sparse
  representation (K-SVD dictionary learning + ADMM sparse reconstruction) on
  housekeeping telemetry channels [2].
  Details and reasoning: `docs/ksvd_admm_notes.md`, `src/telemetry_anomaly/README.md`.

## The working principle: reconstruction-residual detection

Both layers face the same obstacle: **a fixed threshold on the raw signal does not
work**, because the *normal* signal is itself large, structured, and inhomogeneous.

The way out, in both domains, is to stop thresholding the **signal** and start
thresholding what a model of normality **cannot explain about the signal**. Every
detector in this thesis is the same five-step pipeline:

1. **Observation** `y` — a slice of the signal to be judged
   (one averaged spectrum / one telemetry window).
2. **Normality model** `M` — a representation of what "normal" looks like,
   built from reference data known to be clean.
3. **Reconstruction** `y_hat = R(y; M)` — the model's best explanation of the
   observation.
4. **Residual** `r = y - y_hat` — whatever the normal model could *not* explain.
   This, not the raw signal, is the detection evidence.
5. **Adaptive threshold** `tau` — derived from the residual's own statistics,
   so the false-alarm behaviour is controlled without hand-tuned absolute limits.
   Flag where `r > tau`, and localise there.

## Headline results

* **Layer 1 (10-scenario benchmark, 12 CUC events):** snapshot wavelet detector
  alone: 7/12 detected, 0 false positives (conservative preset). Adding the
  temporal baseline-differencing stage: **12/12 detected, 0 false positives**
  (N=2048 frames, 0.20 dB margin), CUC localisation within 0.002–0.066 MHz.
  Caveat that travels with the number: the temporal stage needs a clean baseline
  twin and flags *any* added power, so the baseline must track the authorised
  carrier plan.
* **Layer 2 (SMAP/MSL, 81 channels):** K-SVD-ADMM vs. controlled PCA baseline —
  point-wise F1 **0.522 vs 0.437** (sparse wins: better localisation, lower FPR);
  point-adjusted F1 0.786 vs **0.841** (PCA wins: broad firing + inflation-prone
  metric). Published LSTM-NDT (different protocol, context only): SMAP 0.89 /
  MSL 0.564 point-adjusted; both methods are below LSTM-NDT on SMAP (0.785 and
  0.845) and above it on MSL (0.788 and 0.814).

## Running

All commands are run **from the project root**; the pure-Python detectors need
`PYTHONPATH=src`.

**Layer 1 — RF interference**
```bash
# generate the benchmark datasets (needs the GNU Radio env)
PYTHONPATH=transponder_gnuradio python3 transponder_gnuradio/experiment_scenarios.py
# single-capture snapshot demo
PYTHONPATH=src python3 -m rf_interference.main
# 10-scenario benchmark scoreboard, then figures + CSVs
PYTHONPATH=src python3 -m rf_interference.studies.run_experiments
PYTHONPATH=src python3 -m rf_interference.studies.generate_reports
```

**Layer 2 — telemetry anomaly** (SMAP/MSL benchmark or an auto-generated
synthetic set; see `src/telemetry_anomaly/README.md` for data download)
```bash
PYTHONPATH=src python3 -m telemetry_anomaly.run_telemetry            # quick scoreboard
PYTHONPATH=src python3 -m telemetry_anomaly.evaluate --data data/telemetry/telemanom
PYTHONPATH=src python3 -m telemetry_anomaly.make_comparison          # comparison figure/table
```

## References

[1] A. V. Sai, S. M. M. N., R. D. N., and S. K. L., "An Automated Interference Detection System for FDMA Based Satellite Communication Networks Using Multi-Scale Wavelet Technique," in 2025 IEEE Space, Aerospace and Defence Conference (SPACE), Bangalore, India, 2025, pp. 1-6.

[2] J. He, Z. Cheng, Z. Xu, B. Li, H. Liu, and B. Guo, "Application of sparse representation method based on K-SVD-ADMM in anomaly detection of satellite telemetry," in 2022 Global Reliability and Prognostics and Health Management (PHM-Yantai), Yantai, China, 2022, pp. 1-7, doi: 10.1109/PHM-Yantai55411.2022.9941750.
