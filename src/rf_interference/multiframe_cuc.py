#!/usr/bin/env python3
"""
Multi-frame temporal CUC detection by clean-baseline differencing.

WHY THIS EXISTS (the single-frame -> multi-frame progression):
  The single-snapshot bump detector (carrier_cuc_analysis.py) finds a CUC as a
  localized power-density bump on a host's flat top. That works only when the
  bump clearly exceeds the host's own structure. For a weak, only-slightly-
  narrower CUC (E02: 200 ksym in a 600 ksym host; E04: 300 ksym in a 1 Msym
  host) the bump is only ~0.5-0.9 dB -- the SAME order as the host's
  deterministic RRC shape variation. Measured fact: neither more Welch
  averaging NOR a CFAR threshold separates them, because both the bump and the
  host shape are DETERMINISTIC and survive averaging together.

  The fix is temporal: difference the N-frame-averaged CURRENT spectrum against
  an N-frame-averaged CLEAN BASELINE (same authorized carriers, CUC absent).
  The host shape cancels; only the CUC's power-density excess remains. The
  residual floor falls ~1/sqrt(N) (host-shape estimation error + host-CUC
  cross-terms average out) while the CUC excess is constant -- so detectability
  GROWS with integration time. This mirrors the reference paper's 24 h
  noise-floor baseline (eq. 15) and is the temporal half of the dual-layer
  scheme.
"""
import numpy as np
from scipy.ndimage import uniform_filter1d

from rf_interference.psd import welch_psd_db


def integrated_psd(filename, num_frames):
    """N-frame averaged power spectrum (dB). Thin wrapper over the Welch PSD."""
    return welch_psd_db(filename, num_chunks=num_frames)


def _frequency_align(freqs, cur_db, base_db, max_khz=60.0, step_khz=0.5):
    """Shift the baseline in frequency to best match the current spectrum over
    the occupied bins (handles LO/Doppler drift between captures). Returns the
    aligned baseline. Robust criterion: minimise the MAD of the difference."""
    occ = cur_db > (cur_db.max() - 20.0)   # carrier bins (within 20 dB of peak)
    best_v, best_s = np.inf, 0.0
    for sk in np.arange(-max_khz, max_khz + step_khz, step_khz):
        sh = np.interp(freqs - sk / 1000.0, freqs, base_db)
        d = (cur_db - sh)[occ]
        v = np.median(np.abs(d - np.median(d)))
        if v < best_v:
            best_v, best_s = v, sk
    return np.interp(freqs - best_s / 1000.0, freqs, base_db)


def detect_cuc_baseline(freqs, cur_db, base_db, band_mhz=0.30,
                        k=6.0, min_excess_db=0.20, robust=True):
    """
    robust=True (default) adds drift mitigation before differencing, for a
    baseline taken at a DIFFERENT TIME (the operational case): frequency-align
    the baseline to the current spectrum, then linearly detrend the difference
    (removing residual gain + tilt drift). A localized CUC bump survives the
    linear detrend; broad reference drift does not. Measured tolerance vs raw:
    frequency drift 2 kHz -> >50 kHz, tilt 0.5 dB -> ~2 dB, gain already >4 dB;
    matched baselines still score 12/12 at 0 FP. Set robust=False for the
    idealized identical-reference case.
    """
    """
    Detect CUCs as localized POSITIVE excesses in the baseline-differenced
    spectrum (current - clean baseline, both N-frame averaged).

    The residual is band-integrated over ~CUC bandwidth (a rectangular matched
    filter) to beat per-bin noise, then thresholded at the larger of:
      - k * robust-noise(residual)      (CFAR: adapts to the integrated floor)
      - min_excess_db                   (absolute floor, guards against tiny
                                         host-residual artefacts at huge N)

    Detects both a CUC buried in a host (small ~0.5 dB excess) and a standalone
    unauthorized carrier in a guard band (large excess where the baseline is
    just noise).

    Returns (detections, residual, threshold, noise):
      detections = [{freq_mhz, excess_db, snr}], residual = band-integrated diff.
    """
    df = freqs[1] - freqs[0]
    w = max(1, int(band_mhz / df))
    if robust:
        base_db = _frequency_align(freqs, cur_db, base_db)
        diff = cur_db - base_db
        x = np.arange(len(diff))
        diff = diff - np.polyval(np.polyfit(x, diff, 1), x)   # remove gain+tilt
    else:
        diff = cur_db - base_db
    resid = uniform_filter1d(diff, w)

    med = np.median(resid)
    mad = 1.4826 * np.median(np.abs(resid - med)) + 1e-9
    thr = med + max(k * mad, min_excess_db)

    above = resid > thr
    dets, i, n = [], 0, len(resid)
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < n and above[j]:
            j += 1
        loc = i + int(np.argmax(resid[i:j]))
        dets.append({"freq_mhz": float(freqs[loc]),
                     "excess_db": float(resid[loc] - med),
                     "snr": float((resid[loc] - med) / mad)})
        i = j
    return dets, resid, thr, mad


def cuc_residual_snr(freqs, cur_db, base_db, cuc_fc_mhz, band_mhz=0.30):
    """Diagnostic: band-integrated residual excess at a known CUC and the
    robust noise of the residual away from it -> (excess_db, noise_db, snr)."""
    df = freqs[1] - freqs[0]
    w = max(1, int(band_mhz / df))
    resid = uniform_filter1d(cur_db - base_db, w)
    i = int(np.argmin(np.abs(freqs - cuc_fc_mhz)))
    half = max(1, w // 2)
    excess = resid[i - half:i + half + 1].mean() - np.median(resid)
    g, h = int(0.6 / df), int(1.1 / df)
    off = np.concatenate([resid[max(0, i - h):i - g], resid[i + g:i + h]])
    noise = off.std() if off.size else float("nan")
    return excess, noise, excess / (noise + 1e-9)
