#!/usr/bin/env python3
"""
Shared Welch power-spectral-density helper for the RF interference layer.

This is the single source of the averaged-periodogram PSD used everywhere on the
RF side (main.py, multiframe_cuc.py, run_experiments.py, generate_reports.py,
ablation_weighting.py). It previously lived inside ablation_weighting.py, which
obscured the fact that it is a core shared utility rather than part of that study.
"""
import numpy as np

FFT_SIZE = 8192
NUM_CHUNKS = 100
SAMP_HALF_MHZ = 20.0


def welch_psd_db(filename, fft_size=FFT_SIZE, num_chunks=NUM_CHUNKS):
    """Averaged-periodogram power spectrum in dB (matches main.py).

    Reads a complex64 GNU Radio capture, Welch-averages num_chunks Blackman-windowed
    FFT frames, and returns (frequencies in MHz over [-20, 20], power in dB).
    """
    raw_iq = np.fromfile(filename, dtype=np.complex64)
    window = np.blackman(fft_size)
    acc = np.zeros(fft_size)
    n = 0
    for i in range(num_chunks):
        chunk = raw_iq[i * fft_size:(i + 1) * fft_size]
        if len(chunk) < fft_size:
            break
        acc += np.abs(np.fft.fftshift(np.fft.fft(chunk * window))) ** 2
        n += 1
    acc /= n
    freqs = np.linspace(-SAMP_HALF_MHZ, SAMP_HALF_MHZ, fft_size)
    return freqs, 10 * np.log10(acc)
