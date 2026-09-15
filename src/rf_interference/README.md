# Layer 1 — RF interference detection (wavelet + temporal baseline)

First domain of the dual-domain monitoring system: detection of interference —
specifically the **Carrier-under-Carrier (CUC)** threat — in an FDMA satellite
transponder, after Sai et al. (IEEE SPACE 2025). The telemetry layer watches
spacecraft health; this layer watches the spectrum. Both are instances of one
reconstruction-residual framework — see the root `README.md` and
`docs/framework_chapter_outline.md`.

## The problem, and why the obvious approach fails

An FDMA transponder carries several legitimate carriers side by side, with very
different bandwidths (hundreds of kHz to a few MHz) and power levels. The naive
detector — one energy threshold over the band — fails immediately: set it low
and every weak legitimate carrier is a false alarm; set it high and an
interferer next to a strong carrier is missed. Detection has to be *per
carrier*, which means the band must first be **segmented into carriers**, and
each carrier judged against its own statistics.

The threat model makes it harder still. A CUC interferer transmits *underneath*
an authorised carrier: same centre region, narrower bandwidth, less power. It
creates **no new spectral edge** and **no guard-band energy** — the two things
segmentation can see — so it is invisible to segmentation by construction. Its
only signature is a small localised **bump in power density** on the host
carrier's flat top: the host's PSD is flat there (root-raised-cosine shaping),
and the CUC's power, concentrated in a narrower band, lifts the density locally
by `Δ = 10·log10(1 + (P_c/P_h)·(R_h/R_c))` dB. For a realistic power ratio and a
~3:1 bandwidth ratio this is **under 1 dB** — the detection problem is "find a
sub-dB bump on a plateau."

## Stage 1 — snapshot detector (single capture, no prior knowledge)

Pipeline (`wavelet_dsp.py` → `carrier_cuc_analysis.py`):

1. **Welch PSD**: average 400 frames of 8192-point Blackman-windowed FFTs of the
   complex-baseband capture. Averaging kills the *statistical* ripple of any
   single FFT (a single periodogram has ~5.6 dB of it — unusable).
2. **Wavelet edge detection**: a multi-scale continuous wavelet transform
   (first-derivative-of-Gaussian atoms, 9 dyadic scales) turns carrier edges
   into signed extrema. Small scales localise edges precisely but respond to
   noise; large scales are robust but blurry — so per-scale *votes* for
   start/end edges are combined with a **data-adaptive entropy weighting**
   (scales whose edge evidence is concentrated get more weight). The reference
   paper uses a fixed triangular mid-scale weighting; a controlled ablation
   (`studies/ablation_weighting.py`) found the two **equivalent on this band** —
   an honest negative result we report rather than overstate.
3. **Segmentation + per-carrier CUC test**: a robust iterative-linear fit
   removes the transponder's tilt/noise-floor, occupancy segmentation with
   vote-refined edges yields the carrier list (100% coverage on all benchmark
   scenarios), and a width-banded bump test runs on each carrier's flat top,
   with three sensitivity presets (conservative / balanced / high = fixed
   absolute height+prominence thresholds).

**Snapshot result: 7 of 12 benchmark CUCs at zero false positives**
(conservative preset; 8 and 9 of 12 at the looser presets, at the cost of FPs).

## Why the snapshot layer *cannot* catch the rest — the key insight

The five missed CUCs are weak (~0.5–0.9 dB) or sit in narrow hosts. The first
instinct — "average more frames" — **does not work**, and understanding why
shaped the whole design:

Welch averaging only removes *statistical* fluctuation. What remains after
heavy averaging is the host carrier's own **deterministic** spectral shape
(RRC roll-off, implementation ripple), which is the same order (~0.5 dB) as the
bump we are hunting. Averaging shrinks the noise and the bump-to-shape contrast
*stays flat* — we measured exactly this before building stage 2: bump-SNR vs.
number of frames plateaus. No single-capture detector, however long it
integrates, separates a sub-dB bump from sub-dB deterministic structure.

The deterministic part, however, is **reproducible** — and anything
reproducible can be *cancelled by subtraction* instead of averaged away.

## Stage 2 — temporal baseline-differencing (`multiframe_cuc.py`)

Difference the N-frame-averaged current spectrum against an equally-averaged
**clean baseline** (a CUC-free capture of the same carrier plan and channel,
different payload data). The host's deterministic shape is identical in both
and cancels; the CUC's density excess remains sitting on a ~0.05 dB residual
floor. In framework terms this is reconstruction-residual detection with the
simplest possible normality model — one fixed template:

* model `M` = the clean baseline PSD (frequency-aligned, linearly detrended);
* reconstruction = the baseline itself (nothing to fit);
* residual = `current − baseline` in dB, per carrier flat top;
* threshold = **CFAR**: `median + max(k·MAD, min_excess)` over the flat-top
  residual — adaptive to the local floor, with an additive margin (0.20 dB)
  that sets the sensitivity floor.

**Result: 12/12 CUCs, 0 false positives** (N=2048 frames), localisation within
0.002–0.066 MHz. Robustness measured by `studies/baseline_stress.py`: tolerates
>4 dB gain error, tilt through the full 6 dB tested, >50 kHz frequency drift between baseline and
capture (the align+detrend step buys the drift tolerance; raw differencing
broke at 2 kHz).

**A methodological lesson worth telling:** the last missed CUC (E06, 19 dB
below its host) initially looked like a fundamental limit. It wasn't — it was a
*simulation artifact*. The generator repeated a 10 kB payload, making carriers
periodic and leaving a deterministic ~0.25 dB comb in the residual that buried
the bump. Regenerating all datasets with non-repeating payloads (which is also
strictly more realistic) dropped the floor to ~0.05 dB and the CUC cleared the
threshold. The general point: **deterministic structure does not average down**
— the same failure mode the telemetry layer hits with periodic telemetry, and
the reason both layers reference their residual to a structure-sharing
baseline.

**Honest caveats that travel with the numbers:** the temporal stage needs a
clean baseline twin (operationally: a commissioning capture, refreshed when the
carrier plan changes), and it flags *any* added power — an authorised carrier
that is new or ~0.2 dB stronger than the baseline is flagged too. That is a
feature for a monitoring system (the baseline must track the authorised plan),
but it must be stated.

## Two layers, one scoreboard

| | detects | needs | benchmark |
|---|---|---|---|
| Snapshot (stage 1) | strong/wide CUCs, instantly, no prior | one capture | 7/12, 0 FP |
| Temporal (stage 2) | weak/narrow CUCs | clean baseline + dwell | 12/12, 0 FP |

They are complementary by design: the snapshot layer is the fast, assumption-free
first line; the temporal layer is the sensitive, baseline-referenced second line.
The temporal layer is also the **degenerate single-atom instance** of the
framework whose rich end is the telemetry layer's learned dictionary — see
`docs/framework_chapter_outline.md`.

## Modules

| file | role | framework piece |
|------|------|-----------------|
| `psd.py`                  | shared Welch PSD helper (8192-pt, Blackman)      | observation |
| `wavelet_dsp.py`          | denoise + multi-scale CWT edge votes + weighting | segmentation front end |
| `carrier_cuc_analysis.py` | tilt removal, segmentation, snapshot bump test   | snapshot detector |
| `multiframe_cuc.py`       | baseline differencing + CFAR                     | model + residual + threshold |
| `main.py`                 | single-capture snapshot demo                     | orchestrator |
| `studies/run_experiments.py`   | 10-scenario combined scoreboard             | evaluation |
| `studies/generate_reports.py`  | figures + CSVs (residuals, layer comparison)| evaluation |
| `studies/multiframe_study.py`  | detection-vs-N progression study            | evaluation |
| `studies/baseline_stress.py`   | baseline drift/gain/tilt tolerance          | evaluation |
| `studies/ablation_weighting.py`| entropy vs triangular weighting A/B         | ablation |

## Run

```bash
# from the project root; datasets first (needs the GNU Radio 3.10 env)
PYTHONPATH=transponder_gnuradio python3 transponder_gnuradio/experiment_scenarios.py

PYTHONPATH=src python3 -m rf_interference.main                       # snapshot demo
PYTHONPATH=src python3 -m rf_interference.studies.run_experiments    # scoreboard
PYTHONPATH=src python3 -m rf_interference.studies.generate_reports   # figures + CSVs
```

Outputs land in `results/figures/rf/` and `results/metrics/rf/`.
