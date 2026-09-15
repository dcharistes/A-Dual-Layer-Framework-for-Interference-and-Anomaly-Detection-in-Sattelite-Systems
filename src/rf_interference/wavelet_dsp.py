# wavelet denoise + multi-scale entropy-weighted edge detection
# HARDENED VERSION: handles wide carriers and a tilted noise floor.
import numpy as np
import pywt
from scipy.signal import find_peaks


def wavelet_denoise(power_db, wavelet='sym4', level=3):
    """
    DWT soft-thresholding denoiser. Flattens the noise floor without
    blurring carrier edges. Replaces the 5-point moving average from the paper.
    """
    coeffs = pywt.wavedec(power_db, wavelet, mode='per', level=level)

    # Noise sigma from the finest detail coeffs (cD1), robust via MAD.
    sigma = (1 / 0.6745) * np.median(np.abs(coeffs[-1] - np.median(coeffs[-1])))

    # Universal (VisuShrink) threshold.
    uthresh = sigma * np.sqrt(2 * np.log(len(power_db)))

    # Soft-threshold all detail levels; keep the approximation intact.
    coeffs[1:] = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]

    clean_power = pywt.waverec(coeffs, wavelet, mode='per')
    return clean_power[:len(power_db)]


def detect_boundaries_advanced(frequencies, raw_power, weight_mode='entropy',
                               verbose=True):
    """
    Multi-scale CWT edge detection with cross-scale weighted voting.

    HARDENING CHANGES vs. the original:
      - Extended scales 0..8 (was 0..6) so WIDE-carrier roll-offs get a
        sharp, low-entropy response at some scale instead of only smearing.
      - Softer entropy weighting (1/sqrt(H) instead of 1/H) so the
        large-scale votes that detect wide carriers aren't crushed.

    weight_mode selects how the per-scale votes are combined into V(f):
      'entropy'    -- data-adaptive weights 1/sqrt(H_j) (this thesis).
                      Scales whose coefficient energy is FOCUSED (low Shannon
                      entropy = a real, localized edge) are trusted more.
      'triangular' -- the reference paper's fixed weights (eq. 7):
                      w_j = 1 - |j - jmid|/jmax , jmid = jmax/2 .
                      Provided for a direct A/B comparison.
    """
    if verbose:
        print(f"Running Advanced Wavelet Boundary Detection (weights={weight_mode})...")

    clean_power = wavelet_denoise(raw_power)

    # --- CWT scales: extended to cover wide carriers ---
    # s_j = 2^(j/2). j up to 8 gives s_8 = 16 (was capped at 8 with j_max=6).
    n_scales = 9
    j_values = np.arange(0, n_scales)
    scales = 2 ** (j_values * 0.5)
    coeffs_cwt, _ = pywt.cwt(clean_power, scales, 'gaus1')

    num_bins = len(frequencies)
    # Separate SIGNED edge maps: a 'gaus1' coefficient is negative at a rising
    # power transition (carrier start) and positive at a falling one (carrier
    # end). Voting on the signed sign -- instead of |coeff| -- preserves edge
    # polarity, which is what lets us pair a start with its end (the paper's
    # separate start/end edges, Fig. 7e).
    B_start = np.zeros((n_scales, num_bins))   # rising edges  (coeff < -Tj)
    B_end = np.zeros((n_scales, num_bins))     # falling edges (coeff > +Tj)
    entropies = np.zeros(n_scales)

    j_max = n_scales - 1   # 8
    k_0 = 2.0

    for j in range(n_scales):
        cj = coeffs_cwt[j]

        # --- Robust per-scale edge threshold (signed coeff has ~zero median) ---
        Dj = 1.4826 * np.median(np.abs(cj - np.median(cj)))
        kj = k_0 * (0.8 + 0.4 * ((2 ** j) / (2 ** j_max)))
        Tj = kj * Dj
        B_start[j] = (cj < -Tj).astype(int)
        B_end[j] = (cj > Tj).astype(int)

        # --- Shannon entropy of the coefficient energy (focus measure) ---
        energy = cj ** 2
        p = (energy + 1e-10) / np.sum(energy + 1e-10)
        entropies[j] = -np.sum(p * np.log(p))

    if weight_mode == 'triangular':
        # Reference paper, eq. 7: fixed triangular weights peaking at mid-scale.
        jmid = j_max / 2.0
        weights = 1.0 - np.abs(j_values - jmid) / j_max
    else:
        # Softer entropy weighting: 1/sqrt(H) compresses the focused-vs-diffuse
        # ratio so wide-carrier (large-scale) votes survive into V_f.
        weights = 1.0 / np.sqrt(entropies)
    weights = weights / np.sum(weights)

    if verbose:
        print(f"Cross-scale weights ({weight_mode}):")
        for j, w in enumerate(weights):
            print(f"  Scale {j} (s={scales[j]:>5.2f}): weight={w:.3f}  (entropy={entropies[j]:.2f})")

    # Cross-scale weighted voting, eq. 8, done separately per polarity.
    V_start = np.average(B_start, axis=0, weights=weights)
    V_end = np.average(B_end, axis=0, weights=weights)
    return clean_power, V_start, V_end
