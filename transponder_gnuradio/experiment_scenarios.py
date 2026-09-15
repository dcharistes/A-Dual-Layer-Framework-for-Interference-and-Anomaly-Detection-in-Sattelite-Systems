#!/usr/bin/env python3
"""
Ten experiments on the 36 MHz FDMA transponder.

Each scenario varies the carrier layout, symbol rates, power levels and/or
channel impairments to stress a different aspect of the dual-layer detector.
Every scenario carries an explicit ground truth (which carriers exist and
which are Carrier-Under-Carrier anomalies), written next to the dataset as a
JSON manifest so detection can be scored automatically (see run_experiments.py).

Generate all datasets (run from the project root):
    python3 transponder_gnuradio/experiment_scenarios.py

Datasets + manifests are written to ./data/experiments/.
"""
import json
import os

from fdma_tsp import (FDMASatelliteTransponder, DEFAULT_CARRIERS, DEFAULT_CUCS)

# Long captures (~4880 frames of 8192 samples at 40 MSps) so the single-frame
# -> multi-frame integration study can synthesise ANY frame count N from one
# recording. ~320 MB per dataset.
DURATION_SEC = 1.0
# Non-repeating payload: must exceed the symbols transmitted over the capture
# by the widest carrier (5 Msym x 1 s = 5M sym ~ 1.25M bytes) so carriers are
# NOT periodic. A periodic payload leaves a deterministic comb residual that
# breaks baseline differencing; non-repeating data converges to the true
# continuous QPSK/RRC spectrum (realistic) and cancels on differencing.
DATA_LEN = 4_000_000
OUT_DIR = "./data/experiments"


def C(cid, fc_mhz, sym_khz, power_db, seed, is_cuc=False):
    """Concise carrier builder (fc in MHz, symbol rate in kHz)."""
    d = {"id": cid, "fc": fc_mhz * 1e6, "sym_rate": sym_khz * 1e3,
         "power_db": power_db, "seed": seed}
    if is_cuc:
        d["is_cuc"] = True
    return d


def _grid(start_mhz, step_mhz, n, sym_khz, powers, seed0):
    """A row of n equally-spaced carriers (powers cycles if shorter than n)."""
    return [C(seed0 + k, start_mhz + k * step_mhz, sym_khz,
             powers[k % len(powers)], seed0 + k) for k in range(n)]


# ---------------------------------------------------------------------------
# The ten experiments.
# ---------------------------------------------------------------------------
SCENARIOS = [
    {
        "name": "E01_baseline",
        "desc": "Reference layout: 13 heterogeneous carriers + 2 CUCs "
                "(@ +6.5 narrow-host, +16.0 wide-host).",
        "carriers": DEFAULT_CARRIERS + DEFAULT_CUCS,
        "impairments": {},
    },
    {
        "name": "E02_dense_narrowband",
        "desc": "16 tightly-packed narrowband carriers (600 ksym, 2 MHz pitch). "
                "Stresses segmentation resolution / merging. 1 CUC @ +1.0.",
        "carriers": _grid(-15.0, 2.0, 16, 600, [-4, -2, 0, -3, -1], 200) +
                    [C(299, 1.0, 200, -13.0, 299, is_cuc=True)],
        "impairments": {},
    },
    {
        "name": "E03_sparse_wideband",
        "desc": "4 wide carriers (5 Msym, ~6.75 MHz BW). Stresses wide-carrier "
                "edge handling. 1 CUC @ +13.5 inside the widest host.",
        "carriers": [C(301, -13.5, 5000, 2.0, 301), C(302, -4.5, 5000, 3.0, 302),
                     C(303, 4.5, 5000, 1.0, 303), C(304, 13.5, 5000, 0.0, 304),
                     C(399, 13.5, 400, -12.0, 399, is_cuc=True)],
        "impairments": {},
    },
    {
        "name": "E04_uniform_control",
        "desc": "8 identical carriers (1 Msym, 0 dB, 4 MHz pitch). Easy control "
                "for false-alarm behaviour. 1 CUC @ +2.0.",
        "carriers": _grid(-14.0, 4.0, 8, 1000, [0.0], 400) +
                    [C(499, 2.0, 300, -13.0, 499, is_cuc=True)],
        "impairments": {},
    },
    {
        "name": "E05_extreme_power_spread",
        "desc": "Power from +6 dB down to -12 dB with mixed widths. Stresses the "
                "noise-floor estimate and weak-carrier survival. 1 CUC in a "
                "strong host @ +13.",
        "carriers": [C(501, -16.0, 1000, 6.0, 501), C(502, -11.0, 2000, 3.0, 502),
                     C(503, -6.0, 1000, -4.0, 503), C(504, -2.0, 500, -10.0, 504),
                     C(505, 2.0, 1000, -12.0, 505), C(506, 7.0, 2000, 4.0, 506),
                     C(507, 13.0, 3000, 1.0, 507),
                     C(599, 13.0, 350, -14.0, 599, is_cuc=True)],
        "impairments": {},
    },
    {
        "name": "E06_deep_cuc",
        "desc": "CUC buried at -19 dB (near the detection floor) inside a strong "
                "wide host @ 0. Tests sensitivity limit.",
        "carriers": [C(601, -10.0, 2000, 2.0, 601), C(602, 0.0, 4000, 3.0, 602),
                     C(603, 10.0, 2000, 1.0, 603),
                     C(699, 0.0, 300, -19.0, 699, is_cuc=True)],
        "impairments": {},
    },
    {
        "name": "E07_wide_cuc",
        "desc": "CUC almost as wide as its host (700 ksym CUC in a 1 Msym host "
                "@ +5): the bump is broad and hard to separate from the host.",
        "carriers": [C(701, -12.0, 1000, -2.0, 701), C(702, -6.0, 2000, 1.0, 702),
                     C(703, 0.0, 1000, 0.0, 703), C(704, 5.0, 1000, -1.0, 704),
                     C(705, 12.0, 2000, 1.0, 705),
                     C(799, 5.0, 700, -12.0, 799, is_cuc=True)],
        "impairments": {},
    },
    {
        "name": "E08_multi_cuc",
        "desc": "Three simultaneous CUCs in a wide (-8), a medium (+4) and a "
                "narrow (+14) host. Tests multi-anomaly detection.",
        "carriers": [C(801, -14.0, 1000, -3.0, 801), C(802, -8.0, 4000, 3.0, 802),
                     C(803, -1.0, 1000, -2.0, 803), C(804, 4.0, 2000, 0.0, 804),
                     C(805, 10.0, 1000, -4.0, 805), C(806, 14.0, 2000, 1.0, 806),
                     C(897, -8.0, 300, -14.0, 897, is_cuc=True),
                     C(898, 4.0, 250, -13.0, 898, is_cuc=True),
                     C(899, 14.0, 300, -15.0, 899, is_cuc=True)],
        "impairments": {},
    },
    {
        "name": "E09_strong_tilt_offset",
        "desc": "Baseline carriers + 1 CUC under a steep gain tilt (6 dB) and a "
                "large Doppler offset (60 kHz). Tests tilt-floor tracking.",
        "carriers": DEFAULT_CARRIERS + [DEFAULT_CUCS[1]],   # CUC @ +6.5 only
        "impairments": {"tilt_db": 6.0, "global_offset_hz": 60e3,
                        "noise_voltage": 0.22},
    },
    {
        "name": "E10_clean_complex",
        "desc": "Busy 14-carrier band with NO CUC under moderate tilt. "
                "False-positive stress test: detector must flag nothing.",
        "carriers": _grid(-16.0, 2.5, 13, 800, [-5, -2, 1, -3, 0, -6, 2], 900) +
                    [C(950, 6.0, 3000, 2.0, 950)],
        "impairments": {"tilt_db": 3.0},
    },
]


def generate_all():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[*] Generating {len(SCENARIOS)} experiment datasets into {OUT_DIR}/ "
          f"({DURATION_SEC}s each)\n")
    index = []
    for sc in SCENARIOS:
        dat = os.path.join(OUT_DIR, sc["name"] + ".dat")
        print("=" * 72)
        print(f"{sc['name']}: {sc['desc']}")
        tb = FDMASatelliteTransponder(sc["carriers"], duration_sec=DURATION_SEC,
                                      filename=dat, data_len=DATA_LEN,
                                      **sc["impairments"])
        tb.print_ground_truth()
        tb.start()
        tb.wait()

        gt = tb.ground_truth()
        manifest = {
            "name": sc["name"], "desc": sc["desc"], "dataset": dat,
            "duration_sec": DURATION_SEC, "samp_rate_hz": tb.samp_rate,
            "impairments": sc["impairments"],
            "carriers": gt,
            "cucs": [c for c in gt if c["is_cuc"]],
        }
        with open(os.path.join(OUT_DIR, sc["name"] + ".json"), "w") as f:
            json.dump(manifest, f, indent=2)
        index.append({"name": sc["name"], "desc": sc["desc"],
                      "n_carriers": len(gt),
                      "n_cucs": len(manifest["cucs"])})

    with open(os.path.join(OUT_DIR, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print("=" * 72)
    print(f"[+] Done. {len(SCENARIOS)} datasets + manifests written to {OUT_DIR}/")


if __name__ == "__main__":
    generate_all()
