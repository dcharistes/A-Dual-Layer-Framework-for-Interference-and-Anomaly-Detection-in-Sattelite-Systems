import numpy as np
from scipy.signal import find_peaks


# CUC bump-detector operating points, calibrated on the 10-scenario benchmark
# (run_experiments.py). Each is a (height, prominence, width-band) tuple:
#   height_db  -- bump must rise this far above the local flat-top median
#   prom_db    -- minimum peak prominence
#   wmin/wmax  -- bump width must fall in this band (MHz); rejects sharp
#                 estimation ripple and broad merged-carrier steps
# Benchmark scores (12 CUCs across 10 scenarios):
#   conservative -> 7 detected, 0 false alarms   (operational default)
#   balanced     -> 8 detected, 2 false alarms
#   high         -> 9 detected, 4 false alarms   (recovers narrowband-host CUCs)
SENSITIVITY_PRESETS = {
    "conservative": dict(height_db=0.9, prom_db=0.80, wmin_mhz=0.16, wmax_mhz=0.70),
    "balanced":     dict(height_db=0.7, prom_db=0.45, wmin_mhz=0.12, wmax_mhz=0.90),
    "high":         dict(height_db=0.5, prom_db=0.45, wmin_mhz=0.12, wmax_mhz=0.90),
}


def _estimate_tilt_floor(power_db, iters=6, keep_db=3.0):
    """
    Robust noise-floor estimate via an iteratively-reweighted LINEAR fit.

    The previous windowed-percentile approach failed badly in the crowded
    centre of the band: a 1200-bin window sits almost entirely on neighbouring
    carriers, so even a low percentile lands on a carrier shoulder and reports
    a "floor" of 40-45 dB where the true floor is ~19 dB. That wiped out
    carriers (e.g. 4.5 MHz) and shredded the wide carrier at -7.5 MHz.

    A single straight line models the gentle transponder gain tilt (a few dB
    end-to-end). We fit it only to the bins that currently sit near the floor,
    re-selecting those bins each iteration, so carrier energy is rejected no
    matter how crowded the band is.
    """
    n = len(power_db)
    x = np.arange(n)
    fit = np.full(n, np.percentile(power_db, 15))
    for _ in range(iters):
        below = power_db < fit + keep_db
        if below.sum() < 2:
            break
        m, b = np.polyfit(x[below], power_db[below], 1)
        fit = m * x + b
    return fit


def _merge_close_runs(runs, gap_bins):
    if not runs:
        return runs
    merged = [list(runs[0])]
    for s, e in runs[1:]:
        if s - merged[-1][1] <= gap_bins:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [tuple(r) for r in merged]


def _refine_edges_with_votes(start, end, vf, search):
    """Snap an energy-derived boundary to the nearest strong wavelet vote."""
    n = len(vf)
    def snap(b):
        lo, hi = max(0, b - search), min(n, b + search)
        loc = vf[lo:hi]
        return b if loc.max() <= 0 else lo + int(np.argmax(loc))
    return snap(start), snap(end)


def extract_and_analyze_carriers(frequencies, power_db, v_start, v_end, clean_power,
                                 sensitivity="conservative", verbose=True):
    """
    Two-pass CUC detection:
      Pass 1 -- segment all carriers (energy occupancy gives the run skeleton;
                the wavelet cross-scale start/end VOTES snap each boundary to
                the true edge) and record each one's flat-top variance.
      Pass 2 -- flag a carrier as a CUC host when a LOCALIZED bump sits on its
                denoised flat top (a weak narrow signal buried in the host).

    NOTE on segmentation: deriving boundaries PURELY from the votes (the
    paper's E(f)=1 where V(f)>=Tv, eq. 9) was tried and proved non-robust on
    this adverse band -- the 'gaus1' response fires repeatedly along every RRC
    roll-off and on flat-top ripple, yielding ~60 candidate edges for 12
    carriers with unbalanced start/end counts, so any start/end pairing
    desynchronises. We therefore use energy occupancy for the reliable run
    skeleton and let the votes REFINE (snap) the boundaries.
    """
    if verbose:
        print("[*] Occupancy Segmentation (vote-refined edges) + Bump CUC Analysis...")

    df_mhz = frequencies[1] - frequencies[0]
    floor = _estimate_tilt_floor(clean_power)
    margin_db = 6.0
    # Combined edge votes (either polarity) used only to snap boundaries.
    vf = np.maximum(v_start, v_end)

    occupied = clean_power > (floor + margin_db)
    e = np.diff(occupied.astype(int))
    starts = list(np.where(e == 1)[0] + 1)
    stops = list(np.where(e == -1)[0] + 1)
    if occupied[0]:
        starts = [0] + starts
    if occupied[-1]:
        stops = stops + [len(occupied)]
    runs = _merge_close_runs([list(r) for r in zip(starts, stops)],
                             int(0.15 / df_mhz))

    # ---- Pass 1: segment + measure ----
    cand = []
    for raw_s, raw_e in runs:
        search = int(0.30 / df_mhz)
        s_idx, en_idx = _refine_edges_with_votes(raw_s, raw_e, vf, search)
        if en_idx <= s_idx:
            s_idx, en_idx = raw_s, raw_e
        bw = frequencies[en_idx] - frequencies[s_idx]
        if bw < 0.20 or bw > 12.0: # needs justification
            continue
        m = max(1, int((en_idx - s_idx) * 0.15))
        seg = slice(s_idx + m, en_idx - m)
        ft_clean = clean_power[seg]
        ft_raw = power_db[seg]
        if ft_clean.size < 8:
            continue
        lf = np.mean(floor[seg])
        if np.mean(ft_clean) < lf + margin_db:
            continue
        cand.append({
            "fc": frequencies[s_idx] + bw / 2, "bw": bw,
            "variance": float(np.var(ft_raw)),
            "start_idx": s_idx, "end_idx": en_idx,
            "seg": seg, "ft_clean": ft_clean, "local_floor": lf,
        })

    if not cand:
        if verbose:
            print("[*] No carriers segmented.")
        return []

    # ---- Pass 2: localized-bump CUC test ----
    # A CUC ("carrier under carrier") buries a weak, narrow signal inside a
    # legitimate host. Its signature is a LOCALIZED power excess sitting on the
    # host's otherwise-flat top -- a bump whose WIDTH matches the buried carrier
    # and that rises ~1 dB above the local flat-top level.
    #
    # Detector (tuned on the 10-scenario benchmark, see run_experiments.py):
    #   height     >= median(flat_top) + BUMP_HEIGHT_DB   (host-relative excess)
    #   prominence >= BUMP_PROM_DB
    #   width       in [BUMP_WMIN_MHZ, BUMP_WMAX_MHZ]
    # The WIDTH BAND is the key discriminator: it rejects the narrow estimation
    # ripple of clean wide carriers (too sharp) and merged-carrier steps (too
    # broad), both of which otherwise rival a real CUC in height/prominence.
    # The old `bw > 1.0 MHz` host gate is dropped, so CUCs inside narrowband
    # hosts are now examined too.
    #
    # NOTE: a global flat-top *variance* outlier test was tried and removed.
    # The CUC hosts are NOT variance outliers (their flat-top variance sits
    # inside the clean-carrier population). Varia
    # nce instead just flags the
    # weak narrow carriers, whose dB trace is noisy near the floor -- pure
    # false positives. Variance is still reported below as a diagnostic only.
    variances = np.array([c["variance"] for c in cand])
    vmed = np.median(variances)

    if sensitivity not in SENSITIVITY_PRESETS:
        raise ValueError(f"sensitivity must be one of {list(SENSITIVITY_PRESETS)}")
    preset = SENSITIVITY_PRESETS[sensitivity]
    BUMP_HEIGHT_DB, BUMP_PROM_DB = preset["height_db"], preset["prom_db"]
    BUMP_WMIN_MHZ, BUMP_WMAX_MHZ = preset["wmin_mhz"], preset["wmax_mhz"]
    wmin_bins = max(2, int(BUMP_WMIN_MHZ / df_mhz))

    carriers = []
    for c in cand:
        # Host-relative bump on the denoised flat top, width-banded.
        bump = False
        bump_freq = None
        bump_info = ""
        ft = c["ft_clean"]
        med = np.median(ft)
        pk, props = find_peaks(ft, height=med + BUMP_HEIGHT_DB,
                               prominence=BUMP_PROM_DB, width=wmin_bins)
        widths_mhz = props["widths"] * df_mhz if len(pk) else np.array([])
        ok = [i for i in range(len(pk))
              if BUMP_WMIN_MHZ <= widths_mhz[i] <= BUMP_WMAX_MHZ]
        if ok:
            bump = True
            k = ok[int(np.argmax(props["prominences"][ok]))]
            trim = max(1, int((c["end_idx"] - c["start_idx"]) * 0.15))
            bidx = c["start_idx"] + trim + pk[k]
            bump_freq = float(frequencies[bidx])
            bump_info = f" | CUC bump @ {bump_freq:+.2f} MHz"

        is_spoofed = bump

        c["spoofed_flag"] = is_spoofed
        c["bump_freq"] = bump_freq
        carriers.append({k: c[k] for k in
                         ("fc", "bw", "variance", "spoofed_flag",
                          "start_idx", "end_idx", "bump_freq")})

        if verbose:
            status = "SPOOF" if is_spoofed else "Clean"
            print(f"    -> @ {c['fc']:>6.2f} MHz (BW {c['bw']:>4.2f}) | "
                  f"floor {c['local_floor']:>4.1f} | var {c['variance']:>5.2f} "
                  f"| {status}{bump_info}")

    if verbose:
        n_spoof = sum(c["spoofed_flag"] for c in carriers)
        print(f"[*] Segmented {len(carriers)} carriers "
              f"(variance median={vmed:.2f}); {n_spoof} CUC-flagged.")
    return carriers
