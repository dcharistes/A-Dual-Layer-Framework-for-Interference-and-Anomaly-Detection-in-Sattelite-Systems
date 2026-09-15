#!/usr/bin/env python3
"""
Telemetry data ingest for the Layer-2 anomaly detector.

Two sources, ONE common in-memory representation (`Channel`):

  * SMAP/MSL (NASA telemanom release) -- the real benchmark. Drop the unpacked
    archive into data/telemetry/ so that it contains:
        train/<chan_id>.npy   test/<chan_id>.npy   labeled_anomalies.csv
    Each .npy is (T, n_features); column 0 is the telemetry value, the rest are
    one-hot command flags. labeled_anomalies.csv lists anomaly_sequences per
    channel. Download (the bucket key is gated from CI; fetch interactively):
        https://github.com/khundman/telemanom  ->  data.zip
        ('!'-prefix a curl/wget in the Claude Code prompt to run it in-session)

  * SYNTHETIC periodic generator -- a controlled, ground-truth stand-in (mirrors
    how the RF side simulates the FDMA transponder). Lets the whole pipeline run
    and be validated before the real data is in place, and gives clean labels.
    Written as train/<id>.npy, test/<id>.npy, labels.json.

The loader AUTO-DETECTS the source from the directory contents.
"""
import ast
import csv
import json
import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Channel:
    """One telemetry channel: anomaly-free train + labelled test."""
    cid: str
    train: np.ndarray                 # (T_train, n_features)
    test: np.ndarray                  # (T_test,  n_features)
    anomaly_sequences: list = field(default_factory=list)   # [[start, end], ...]
    spacecraft: str = ""

    @property
    def train_values(self):
        """Primary telemetry value of the train split (column 0)."""
        return self.train[:, 0]

    @property
    def test_values(self):
        return self.test[:, 0]

    def label_mask(self):
        """Boolean ground-truth mask over the test split (True = anomalous)."""
        m = np.zeros(len(self.test), dtype=bool)
        for s, e in self.anomaly_sequences:
            m[max(0, s):min(len(m), e + 1)] = True
        return m


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load(data_dir):
    """Load all channels from data_dir, auto-detecting telemanom vs synthetic."""
    if os.path.exists(os.path.join(data_dir, "labeled_anomalies.csv")):
        return _load_telemanom(data_dir)
    if os.path.exists(os.path.join(data_dir, "labels.json")):
        return _load_synthetic(data_dir)
    raise FileNotFoundError(
        f"No telemetry found in {data_dir}. Expected either a telemanom layout "
        f"(labeled_anomalies.csv + train/ + test/) or a synthetic one "
        f"(labels.json + train/ + test/). Generate a synthetic set with "
        f"generate_synthetic('{data_dir}').")


def _load_telemanom(data_dir):
    rows = {}
    with open(os.path.join(data_dir, "labeled_anomalies.csv")) as fh:
        for r in csv.DictReader(fh):
            rows[r["chan_id"]] = r
    channels = []
    for cid, r in rows.items():
        tp = os.path.join(data_dir, "train", cid + ".npy")
        ep = os.path.join(data_dir, "test", cid + ".npy")
        if not (os.path.exists(tp) and os.path.exists(ep)):
            continue                                   # csv lists it but files absent
        seqs = ast.literal_eval(r.get("anomaly_sequences", "[]"))
        channels.append(Channel(cid, _as2d(np.load(tp)), _as2d(np.load(ep)), seqs,
                                 r.get("spacecraft", "")))
    return channels


def _load_synthetic(data_dir):
    labels = json.load(open(os.path.join(data_dir, "labels.json")))
    channels = []
    for cid, seqs in labels.items():
        tr = np.load(os.path.join(data_dir, "train", cid + ".npy"))
        te = np.load(os.path.join(data_dir, "test", cid + ".npy"))
        channels.append(Channel(cid, _as2d(tr), _as2d(te), seqs, "SYNTH"))
    return channels


def _as2d(a):
    return a[:, None] if a.ndim == 1 else a


# ---------------------------------------------------------------------------
# Synthetic periodic-telemetry generator
# ---------------------------------------------------------------------------
def _periodic_base(n, period, rng, harmonics=(1, 2, 3), phases=None):
    """A smooth periodic carrier (sum of a few harmonics) + light noise.

    `phases` is shared between the train and test realisations so they describe
    the SAME normal process (only the noise differs).
    """
    t = np.arange(n)
    sig = np.zeros(n)
    for i, h in enumerate(harmonics):
        ph = phases[i] if phases is not None else rng.uniform(0, 2 * np.pi)
        sig += (1.0 / h) * np.sin(2 * np.pi * h * t / period + ph)
    return sig + 0.03 * rng.standard_normal(n)


def _add_periodic_transient(sig, period, rng, phase=0.5, width=3,
                            amp=1.8, jitter=0.5):
    """
    Add a sharp bump once per period at a fixed phase, with per-period AMPLITUDE
    jitter. This is BENIGN normal structure, but because the amplitude varies the
    sparse model cannot cancel it exactly -> a recurring residual peak at this
    phase every period. A constant threshold over-flags these benign peaks; the
    period-referenced EWMA expects them. This reproduces the paper's premise that
    periodic telemetry yields periodic reconstruction-residual peaks.
    """
    n = len(sig)
    sigma = max(1.0, width / 2.0)
    for c in range(int(phase * period), n, period):
        a = amp * (1.0 + jitter * rng.standard_normal())
        lo, hi = max(0, c - 3 * width), min(n, c + 3 * width + 1)
        tt = np.arange(lo, hi) - c
        sig[lo:hi] += a * np.exp(-(tt ** 2) / (2 * sigma ** 2))
    return sig


def generate_synthetic(out_dir, n_channels=4, period=100, n_train_periods=20,
                       n_test_periods=20, seed=7, benign_transient=False):
    """
    Write a small synthetic telemetry benchmark in the loader's synthetic layout.

    Periodic, anomaly-free TRAIN; periodic TEST with injected COLLECTIVE anomalies
    of three flavours that exercise both threshold bounds:
      - 'spike'  : amplified / offset oscillation        -> high residual
      - 'frozen' : sensor stuck at a constant value       -> low residual
      - 'freq'   : local change of oscillation frequency  -> high residual

    With benign_transient=True a jittered once-per-period bump is added to BOTH
    splits (normal structure) so the reconstruction residual has recurring benign
    peaks -- the setting in which the period-EWMA threshold beats a fixed one.
    Returns the path written.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(out_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "test"), exist_ok=True)
    labels = {}

    flavours = ["spike", "frozen", "freq", "spike"]
    for c in range(n_channels):
        cid = f"S-{c+1}"
        n_tr = period * n_train_periods
        n_te = period * n_test_periods
        phases = rng.uniform(0, 2 * np.pi, size=3)        # shared normal process
        train = _periodic_base(n_tr, period, rng, phases=phases)
        test = _periodic_base(n_te, period, rng, phases=phases)
        if benign_transient:
            _add_periodic_transient(train, period, rng)
            _add_periodic_transient(test, period, rng)

        # inject 2 anomalies in the test split (away from the edges)
        seqs = []
        flavour = flavours[c % len(flavours)]
        for a in range(2):
            length = rng.integers(period // 2, period)          # ~half-to-full period
            start = int((a + 1) * n_te / 3) + rng.integers(-period // 4, period // 4)
            start = int(np.clip(start, period, n_te - period - length))
            end = start + int(length)
            seg = slice(start, end)
            if flavour == "spike":
                test[seg] = test[seg] * 3.0 + 2.0
            elif flavour == "frozen":
                test[seg] = test[start]                          # stuck value
            elif flavour == "freq":
                t = np.arange(end - start)
                test[seg] = np.sin(2 * np.pi * 5 * t / period)   # wrong frequency
            # slice(start, end) corrupts indices [start, end-1]; record an
            # INCLUSIVE [start, end-1] sequence so it matches the telemanom
            # convention that Channel.label_mask expects (m[s : e+1]).
            seqs.append([start, end - 1])

        np.save(os.path.join(out_dir, "train", cid + ".npy"), train.astype(np.float32))
        np.save(os.path.join(out_dir, "test", cid + ".npy"), test.astype(np.float32))
        labels[cid] = seqs

    json.dump(labels, open(os.path.join(out_dir, "labels.json"), "w"), indent=2)
    print(f"[+] synthetic telemetry -> {out_dir}/ "
          f"({n_channels} channels, period={period})")
    return out_dir


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    out = os.path.abspath(os.path.join(here, "..", "..", "data", "telemetry", "synthetic"))
    generate_synthetic(out)
    chans = load(out)
    for ch in chans:
        print(f"  {ch.cid}: train {ch.train.shape} test {ch.test.shape} "
              f"anomalies {ch.anomaly_sequences}")
