#!/usr/bin/env python3
"""
FDMA Satellite Transponder Simulation - HARDENED / REALISTIC VERSION

Adds realistic adverse-channel impairments on top of the clean transponder:
  1. Unequal carrier power     (power_db per carrier; breaks global thresholding)
  2. Tilted noise floor / gain (transponder gain slope across the band)
  4. No-guard-band carrier pair (overlapping roll-offs; only edge detection resolves)
  6. Global frequency offset    (Doppler / LO residual; nothing on exact bin centers)

The transponder is now SCENARIO-DRIVEN: pass any list of carrier dicts and
impairment parameters, so the same 36 MHz model can be reused for many
experiments (see experiment_scenarios.py). A carrier dict marked
{"is_cuc": True} is the ground-truth anomaly (Carrier-Under-Carrier).

Backward compatible: run as a script and it regenerates the original anomaly
and clean baseline datasets exactly as before.
"""
import math
import numpy as np
from gnuradio import gr, blocks, digital, analog
from gnuradio import filter as gr_filter


# Original 13 legitimate carriers (the default scenario / baseline experiment).
DEFAULT_CARRIERS = [
    # -- LEFT FLANK (Small & Medium VSAT Links) --
    {"id": 1,  "fc": -16.5e6, "sym_rate": 1e6,   "power_db": -6.0, "seed": 1},
    {"id": 2,  "fc": -14.5e6, "sym_rate": 1e6,   "power_db": -3.0, "seed": 2},
    {"id": 3,  "fc": -12.0e6, "sym_rate": 2e6,   "power_db":  1.0, "seed": 3},
    # -- WIDEBAND BURST (e.g., Video Broadcast) - strong --
    {"id": 4,  "fc": -7.5e6,  "sym_rate": 4e6,   "power_db":  3.0, "seed": 4},
    # -- CENTER CLUSTER (Narrowband Data & SCADA) - includes weak ones --
    {"id": 5,  "fc": -3.5e6,  "sym_rate": 1e6,   "power_db": -2.0, "seed": 5},
    {"id": 6,  "fc": -2.0e6,  "sym_rate": 500e3, "power_db": -8.0, "seed": 6},  # weak SCADA
    {"id": 7,  "fc": -0.5e6,  "sym_rate": 500e3, "power_db": -7.0, "seed": 7},
    {"id": 8,  "fc":  1.5e6,  "sym_rate": 2e6,   "power_db":  0.0, "seed": 8},
    # -- RIGHT FLANK (Mixed Traffic) --
    {"id": 9,  "fc":  4.5e6,  "sym_rate": 2e6,   "power_db": -1.0, "seed": 9},
    # -- NO-GUARD-BAND PAIR (overlapping roll-offs, no floor between them) --
    {"id": 14, "fc":  6.0e6,  "sym_rate": 1e6,   "power_db": -2.0, "seed": 14},
    {"id": 15, "fc":  7.0e6,  "sym_rate": 1e6,   "power_db": -3.0, "seed": 15},
    {"id": 11, "fc":  9.5e6,  "sym_rate": 1e6,   "power_db": -4.0, "seed": 11},
    {"id": 12, "fc": 12.0e6,  "sym_rate": 2e6,   "power_db":  2.0, "seed": 12},
    # -- THE TARGET ZONE (High Speed Data Hub) --
    {"id": 13, "fc": 16.0e6,  "sym_rate": 3e6,   "power_db":  1.0, "seed": 13},
]

DEFAULT_CUCS = [
    {"id": 99, "fc": 16.0e6, "sym_rate": 300e3, "power_db": -13.0, "seed": 99, "is_cuc": True},
    {"id": 98, "fc": 6.5e6,  "sym_rate": 250e3, "power_db": -15.0, "seed": 98, "is_cuc": True},
]


class FDMASatelliteTransponder(gr.top_block):
    def __init__(self,
                 carriers,
                 duration_sec=1.0,
                 filename="./data/thesis_dataset_cuc_anomaly.dat",
                 global_offset_hz=12.5e3,
                 tilt_db=2.0,
                 noise_voltage=0.18,
                 excess_bw=0.35,
                 data_len=10000):
        super(FDMASatelliteTransponder, self).__init__(
            "FDMA Satellite Transponder Simulation (Hardened)")

        # ---- GLOBAL SIMULATION PARAMETERS ----
        self.samp_rate = 40e6   # 40 MSps (simulates a full 36 MHz transponder)
        self.excess_bw = excess_bw
        self.carriers_meta = carriers   # keep for printing / manifest

        # ---- BUILD THE DSP GRAPH DYNAMICALLY ----
        self.space_combiner = blocks.add_cc()
        for i, c in enumerate(carriers):
            sps = int(self.samp_rate / c["sym_rate"])

            # data_len bytes of payload. The default (10000) REPEATS within a
            # capture, making each carrier periodic with a fixed fine spectral
            # structure -- fine for a single snapshot, but it leaves a
            # DETERMINISTIC residual under different-data baseline differencing
            # (it does not average down). Set data_len large enough that the
            # sequence does not repeat over the capture so the host PSD
            # converges to its smooth shape (realistic non-repeating traffic).
            np.random.seed(c["seed"])
            random_bytes = np.random.randint(0, 256, data_len).tolist()
            src = blocks.vector_source_b(random_bytes, True)

            mod = digital.psk_mod(
                constellation_points=4, mod_code="gray", differential=True,
                samples_per_symbol=sps, excess_bw=self.excess_bw)

            phase_inc = 2 * math.pi * c["fc"] / self.samp_rate
            rotator = blocks.rotator_cc(phase_inc)

            # IMPAIRMENT 1: unequal power. dB -> voltage amplitude: V = 10^(dB/20)
            amplitude = 10.0 ** (c["power_db"] / 20.0)
            amp = blocks.multiply_const_cc(amplitude)

            self.connect(src, mod, rotator, amp)
            self.connect(amp, (self.space_combiner, i))

        # ---- THE CHANNEL (impairments applied to the combined signal) ----
        # IMPAIRMENT 2: tilted transponder gain across the band.
        tilt_taps = self._make_tilt_taps(tilt_db)
        self.tilt = gr_filter.fir_filter_ccf(1, tilt_taps)
        self.connect(self.space_combiner, self.tilt)

        # Background thermal / cosmic noise (flat, white)
        self.noise = analog.noise_source_c(analog.GR_GAUSSIAN, noise_voltage, 0)
        self.final_add = blocks.add_cc()
        self.connect(self.noise, (self.final_add, 0))
        self.connect(self.tilt,  (self.final_add, 1))

        # IMPAIRMENT 6: global frequency offset (Doppler / LO residual).
        global_phase_inc = 2 * math.pi * global_offset_hz / self.samp_rate
        self.global_shift = blocks.rotator_cc(global_phase_inc)
        self.connect(self.final_add, self.global_shift)

        # ---- LIMITER + OUTPUT ----
        total_samples = int(self.samp_rate * duration_sec)
        self.head = blocks.head(gr.sizeof_gr_complex, total_samples)
        self.sink = blocks.file_sink(gr.sizeof_gr_complex, filename, False)
        self.connect(self.global_shift, self.head, self.sink)

    @staticmethod
    def _make_tilt_taps(tilt_db):
        """Short FIR whose magnitude response slopes across the band by ~tilt_db."""
        if tilt_db <= 0.0:
            return [1.0]
        a = min(0.25, 0.04 * tilt_db)
        taps = [a * 0.5, 1.0, -a * 0.5]
        dc = sum(taps)
        return [t / dc for t in taps]

    def ground_truth(self):
        """Return a list of carrier records with computed bandwidth (MHz)."""
        out = []
        for c in self.carriers_meta:
            out.append({
                "id": c["id"], "fc_mhz": c["fc"] / 1e6,
                "sym_mhz": c["sym_rate"] / 1e6,
                "bw_mhz": c["sym_rate"] * (1 + self.excess_bw) / 1e6,
                "power_db": c["power_db"], "is_cuc": bool(c.get("is_cuc", False)),
            })
        return out

    def print_ground_truth(self):
        print("\n[GROUND TRUTH] Carriers in this dataset:")
        print("  ID    fc(MHz)   sym(MHz)  power(dB)   BW(MHz)   role")
        for c in self.ground_truth():
            role = "CUC (carrier-under-carrier)" if c["is_cuc"] else ""
            print(f"  {c['id']:<5} {c['fc_mhz']:>7.2f}  {c['sym_mhz']:>7.3f}  "
                  f"{c['power_db']:>7.1f}   {c['bw_mhz']:>6.3f}   {role}")
        print()


def run_scenario(carriers, filename, duration_sec=1.0, **impairments):
    """Convenience: build, print ground truth, run, return the top_block."""
    tb = FDMASatelliteTransponder(carriers, duration_sec=duration_sec,
                                  filename=filename, **impairments)
    tb.print_ground_truth()
    tb.start()
    tb.wait()
    return tb


if __name__ == '__main__':
    import os
    os.makedirs("./data", exist_ok=True)
    sim_time = 1.0

    print("[*] Initializing HARDENED FDMA Transponder Simulation...")
    print(f"[*] Master Sample Rate: 40 MSps")

    # ---- ANOMALY dataset (default carriers + both CUCs) ----
    anomaly_file = "./data/thesis_dataset_cuc_anomaly.dat"
    tb = run_scenario(DEFAULT_CARRIERS + DEFAULT_CUCS, anomaly_file, sim_time)
    size_mb = int((tb.samp_rate * sim_time * 8) / 1024 / 1024)
    print(f"[+] Anomaly dataset complete: {anomaly_file}  (~{size_mb} MB)")

    # ---- CLEAN baseline (no CUC) for false-positive testing ----
    clean_file = "./data/thesis_dataset_clean.dat"
    run_scenario(DEFAULT_CARRIERS, clean_file, sim_time)
    print(f"[+] Clean baseline complete: {clean_file}  (~{size_mb} MB)")

    print("\n[+] Done. anomaly file should flag CUC(s); clean file should flag NOTHING.")
