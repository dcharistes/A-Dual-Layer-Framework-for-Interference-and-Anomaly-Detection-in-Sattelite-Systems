"""
Telemetry anomaly detection layer (Layer 2 of the dual-domain system).

Sparse-representation anomaly detection on satellite housekeeping telemetry,
after He et al. (K-SVD-ADMM, 2022). This is the rich-model instance of the
unified reconstruction-residual framework (see docs/framework_chapter_outline.md);
the RF temporal layer (src/multiframe_cuc.py) is the single-atom instance.

Phase 1 (this code): dictionary + OMP reconstruction + residual score + a fixed
(constant-in-time) CFAR threshold, validated on a synthetic periodic generator.
Phase 2 will add true K-SVD + ADMM and the period-based EWMA threshold
(see docs/ksvd_admm_notes.md).

Run convention (from project root):
    PYTHONPATH=src python3 -m telemetry_anomaly.run_telemetry
"""
