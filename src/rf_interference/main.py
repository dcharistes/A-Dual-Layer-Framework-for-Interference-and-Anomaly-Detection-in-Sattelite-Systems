import numpy as np
import matplotlib.pyplot as plt
from rf_interference import wavelet_dsp
from rf_interference import carrier_cuc_analysis

# 1. Read the raw binary I/Q data exported from GNU Radio
# GNU Radio's default output is 32-bit float for I and 32-bit float for Q (complex64)
# e.g. raw_iq[0] = I₀ + jQ₀ sample 0
raw_iq = np.fromfile('./data/thesis_dataset_cuc_anomaly.dat', dtype=np.complex64)

# 2. Convert I/Q time-domain data into a Frequency-domain Power Spectrum
# We take a chunk of data (e.g., 8192 samples) to match your resolution bandwidth (RBW)
fft_size = 8192

# 3. Create the frequency X-axis array (spanning -20MHz to +20MHzfor 40e6 samp_rate)
frequencies = np.linspace(-20.0, 20.0, fft_size) # res bw 4.833kHz. any adjacent detail less than 4.833kHz is merged in the same bin.
num_chunks = 400  # Welch averaging frames (more frames -> lower-variance PSD)
actual_chunks = 0
avg_power_linear = np.zeros(fft_size)

# Apply a window function to prevent edge artifacts. tapering.
# the 8192 sample chunk is not a period of a signal.
# it is cutted out from a stream. so spectral leakage mst be mitigated in the fft
# near edge samples go to 0, there is some reduce in the resolution due to sidelobe supress.
# main lobe only. so the cuc is speactrally visible on the flat top of carrier 13,
# no side lobe contanmination.(just a bump)
window = np.blackman(fft_size)

# 100 chunks of 8192 samples each. the blackman window is applied in each 8192 chunk.
print(f"[*] Calculating Welch's Power Spectral Density across {num_chunks} frames...")
for i in range(num_chunks):
    # Pull a chunk, apply window, FFT, and get linear power
    chunk = raw_iq[i*fft_size : (i+1)*fft_size]
    if len(chunk) < fft_size:
        print(f"[!] Only {actual_chunks} complete frames available. Stopping early.")
        break
    spectrum = np.fft.fftshift(np.fft.fft(chunk * window))
    avg_power_linear += np.abs(spectrum)**2
    actual_chunks += 1

# Divide by total frames to get the true average
avg_power_linear /= actual_chunks

# Convert the mathematically smoothed linear average into Logarithmic dB
power_spectrum_db = 10 * np.log10(avg_power_linear)

# 2. Run the DSP pipeline (cross-scale voting gives separate start/end edges)
clean_power, v_start, v_end = wavelet_dsp.detect_boundaries_advanced(frequencies, power_spectrum_db)

# 3. Classify anomalies
detected_carriers = carrier_cuc_analysis.extract_and_analyze_carriers(frequencies, power_spectrum_db, v_start, v_end, clean_power)

# visualization
plt.figure(figsize=(12, 6))

# Plot the raw data and the wavelet-denoised data
plt.plot(frequencies, power_spectrum_db, color='lightgray', label='Raw Sensor Data', alpha=0.7)
plt.plot(frequencies, clean_power, color='dodgerblue', label='Wavelet Denoised Spectrum')

# Shade the carriers based on their classification
for c in detected_carriers:
    color = 'red' if c["spoofed_flag"] else 'mediumseagreen'
    label = 'Pirate Anomaly (CUC)' if c["spoofed_flag"] else 'Authorized Carrier'
    plt.axvspan(frequencies[c["start_idx"]], frequencies[c["end_idx"]],
                color=color, alpha=0.3, label=label)

# Clean up duplicate labels in the legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper right')

plt.title("Automated Wavelet-Entropy Carrier Segmentation and CUC Detection")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Power (dB)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
