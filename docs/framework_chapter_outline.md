# Unified Reconstruction-Residual Framework — chapter outline

Purpose of this chapter: make the RF interference layer and the telemetry anomaly
layer read as **two instances of one principle**, not two stapled-together papers.

## Backbone claim

> Every detector in this thesis reduces to the same recipe: **model "normal,"
> reconstruct the observation from that model, and flag where the reconstruction
> residual is anomalous under an adaptive threshold.** The detectors differ only
> in (i) the *richness* of the normality model and (ii) the *assumptions* baked
> into the threshold.

## Two design axes

1. **Model richness** — single fixed reference  →  overcomplete *learned* dictionary.
2. **Threshold prior** — none / data-adaptive CFAR  →  periodicity-aware EWMA.

Every detector is a point in this 2-D design space.

---

## Section plan

### X.1 Motivation
Two surveillance problems on one spacecraft link:
- unauthorized / interfering RF emissions (the interference problem), and
- health / behavioural faults in housekeeping telemetry (the anomaly problem).

Argue they are the *same statistical problem* in two domains — which is why one
framework spans both, and why the thesis broadens from "interference detection"
to "interference **and** anomaly detection."

### X.2 The general framework (define once, formally)
- reference data → **normality model** `M`
- **reconstruction** `y_hat = R(y; M)`
- **residual** `r = y - y_hat` (or a norm thereof)
- **adaptive threshold** `tau`, derived from the residual's own statistics
- **decision**: flag where `r > tau`

State the two design axes (above) explicitly here.

### X.3 Instance 1 — RF interference (temporal baseline-differencing layer)
Map `src/multiframe_cuc.py` onto the framework:
- `M` = one clean baseline PSD (a CUC-free twin capture)
- `R` = frequency-aligned + linearly-detrended baseline
- `r` = `current_PSD - baseline_PSD`
- `tau` = CFAR, `med + max(k*MAD, min_excess_db)`

Key point: this is the **degenerate case** of sparse-dictionary reconstruction —
a dictionary of a **single atom** with the sparse code fixed to 1.

### X.4 Instance 2 — Telemetry anomaly (K-SVD-ADMM sparse layer)
Map `src/telemetry_anomaly/` onto the framework:
- `M` = K-SVD-learned overcomplete dictionary `D` (normal local patterns)
- `R` = sparse reconstruction `D*X` (OMP in Phase 1, ADMM in Phase 2)
- `r` = residual matrix `E`; per-window score `||E(:,j)||_2`
- `tau` = period-based EWMA band (`U_t(1+eta)`, `U_t(1-gamma)`)

This is the **rich-model end** of the same axis.

### X.5 The unifying view (the intellectual contribution)
One figure + one table placing all detectors on the `model richness x threshold
prior` plane. Then the three points that lift this above "two implemented papers":
1. The RF **CFAR** (const false alarm rate) threshold is the **non-periodic generalization** of the
   telemetry paper's **periodic EWMA** threshold — we independently produced the
   threshold variant the telemetry domain needs when periodicity fails.
2. Both domains hit the **same failure mode**: deterministic / periodic structure
   inflates the residual. We solved it on the RF side with non-repeating data +
   CFAR; the telemetry paper solves it with period-referenced EWMA. Same wall,
   two doors.
3. Baseline differencing = sparse reconstruction with `|dictionary| = 1`. The two
   layers are the endpoints of a single continuum, not separate techniques.

### X.6 Scope and honest boundaries
- The **snapshot wavelet / segmentation layer** (`wavelet_dsp.py` +
  `carrier_cuc_analysis.py`) is a **different detector class** — edge / energy
  segmentation, model-free. Say so plainly; do **not** force it into the
  reconstruction-residual framework.
- **Cross-domain fusion** (correlating an RF interference event with a telemetry
  fault) is named as **future work**, not claimed — it needs jointly-coupled data
  we do not have.

---

## Placement in the thesis
This chapter **precedes** the Layer-2 implementation/evaluation chapter, so the
reader meets the unifying principle before the K-SVD/ADMM machinery. The Layer-1
chapters can be lightly retro-fitted with a forward reference ("see Ch. X: this is
the single-atom instance of the framework").

## Mapping table (drop straight into the chapter)

| Aspect            | Layer 1 — temporal (RF)             | Layer 2 — K-SVD-ADMM (telemetry)     |
|-------------------|-------------------------------------|--------------------------------------|
| Domain            | PSD over frequency                  | sensor value over time               |
| Normality model M | single clean baseline PSD           | learned overcomplete dictionary D    |
| Reconstruction R  | aligned + detrended baseline        | sparse code `D*X` (OMP / ADMM)       |
| Residual r        | `current - baseline`                | residual matrix `E`                  |
| Score             | band-integrated residual            | `||E(:,j)||_2`                       |
| Threshold tau     | CFAR `med + max(k*MAD, min_excess)` | period EWMA `U_t(1±{eta,gamma})`     |
| Model richness    | 1 atom (degenerate)                 | K atoms (rich)                       |
| Threshold prior   | none / CFAR (non-periodic)          | periodicity-aware                    |
