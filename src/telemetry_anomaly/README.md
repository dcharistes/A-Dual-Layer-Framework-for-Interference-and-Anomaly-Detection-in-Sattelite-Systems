# Layer 2 — Telemetry anomaly detection (sparse representation)

Second domain of the dual-domain monitoring system: anomaly detection on
satellite **housekeeping telemetry** via sparse representation, after He et al.
(K-SVD-ADMM, 2022). The RF interference layer watches the spectrum; this layer
watches spacecraft health. Both are instances of one reconstruction-residual
framework — see `docs/framework_chapter_outline.md` and `docs/ksvd_admm_notes.md`.

## Run

```bash
# from the project root
PYTHONPATH=src python3 -m telemetry_anomaly.run_telemetry           # synthetic demo
PYTHONPATH=src python3 -m telemetry_anomaly.run_telemetry --data data/telemetry/telemanom
```

With no data present, a synthetic periodic benchmark is generated into
`data/telemetry/synthetic/` automatically.

## Modules

| file | role | framework piece |
|------|------|-----------------|
| `data_loader.py`   | load SMAP/MSL or synthetic channels; synthetic generator | reference data |
| `preprocessing.py` | standardise + sliding window -> sample matrix S | observation |
| `sparse_model.py`  | dictionary + OMP reconstruction residual | normality model + R |
| `detector.py`      | window->point scores, threshold, P/R/F1/FPR | residual + threshold |
| `run_telemetry.py` | per-channel scoreboard | orchestrator |

## Phases (all done — full story in `docs/ksvd_admm_notes.md`)

- **Phase 1:** naive dictionary (random train windows) + OMP + fixed CFAR
  threshold, validated on the synthetic generator. Caught spikes and frequency
  shifts; *missed the frozen-sensor channel* (low residual, one-sided
  threshold) — independently reproducing the paper's motivation for a
  two-sided band.
- **Phase 2:** true K-SVD dictionary + ADMM joint (X, E) solve + period-EWMA
  two-sided threshold. Fixed the frozen-sensor miss. Key finding: the paper's
  multiplicative EWMA band degenerates when normal data reconstructs
  near-perfectly (score baseline -> 0); fixed with an **additive floor** — the
  RF layer's CFAR margin, which is the bridge to the unified framework.
- **Phase 3:** full SMAP/MSL evaluation (81 channels) vs a controlled PCA
  reconstruction baseline + published LSTM-NDT context. K-SVD-ADMM wins
  point-wise F1 (0.522 vs 0.437 — tighter localisation, lower FPR); PCA wins
  point-adjusted F1 (0.841 vs 0.786 — broad firing under an inflation-prone
  metric). The scoring metric decides the winner, so both are reported.
  Extra entry points: `evaluate.py` (full run) and `make_comparison.py`
  (figure/table from the per-channel CSV, no re-eval).

## Real data (SMAP/MSL)

Fetch the NASA telemanom release (https://github.com/khundman/telemanom, `data.zip`)
and unpack so a directory contains `train/`, `test/`, `labeled_anomalies.csv`.
Point `--data` at it. The bucket key is gated from CI; fetch it interactively
(in the Claude Code prompt, prefix a `curl`/`wget` with `!` to run it in-session).
