#!/usr/bin/env python3
"""
Axis-2 ablation: the threshold prior, with the model held fixed.

Chapter 3 names two ends of Axis 2 (structure-free CFAR, periodicity-referenced
EWMA) but nothing traversed it. This instruments it at THREE points, mirroring
the three points on Axis 1, and adds the deployable variant the thesis lacks:

  constant-oracle : a scalar threshold swept for best point-adjusted F1.
                    Reproduces the published Chapter 7 numbers. NOT deployable:
                    the sweep reads the labels.
  CFAR-swept      : med + k*robust_sigma, k swept for best PA-F1. The
                    structure-free end of Axis 2, oracle-tuned for comparability.
  CFAR-fixed      : med + 4*robust_sigma, no sweep at all. DEPLOYABLE -- no
                    labels are read. The same rule Layer 1 uses.
  EWMA-plain      : the band with period=1, i.e. adaptive but with NO periodicity
                    prior. Isolates adaptivity from periodicity.
  EWMA-period     : the band with the estimated period. The reference method's
                    rule and the far end of Axis 2.

The band changes two things at once relative to a constant threshold: it becomes
adaptive AND periodic. EWMA-plain separates the two.

Chapter 3 claims two design axes. Axis 1 (model richness) is instrumented by the
K-SVD vs PCA comparison in evaluate.py. Axis 2 (the prior built into the
threshold) had no experiment: evaluate.py scores every method with a swept
CONSTANT threshold, so the period-referenced EWMA band of Chapter 6 never enters
the reported numbers.

This runs the identical K-SVD-ADMM scores through both threshold rules:

  constant : tau = a scalar, swept for best point-adjusted F1   (structure-free)
  period-EWMA : U_t = lam*U_{t-p} + (1-lam)*a_t, two-sided band with the
                additive floor, eta/gamma swept for best point-adjusted F1

Same detector, same scores, same data, same best-F1 protocol. Only the prior
changes, so any difference is attributable to the threshold alone.

Run:
  PYTHONPATH=src python3 -m telemetry_anomaly.ablation_threshold \
      --data data/telemetry/telemanom
"""
import argparse, csv, os
import numpy as np

from telemetry_anomaly import data_loader as dl
from telemetry_anomaly import preprocessing as pp
from telemetry_anomaly import sparse_model as sm
from telemetry_anomaly import detector as det

CSV_DIR = os.path.join("results", "metrics", "telemetry")
# The reference method's own sweep (docs/ksvd_admm_notes.md S6): eta in [0,2]
# step 0.1, gamma in [0,1] step 0.1 -> 21 x 11 = 231 combinations. Anything
# coarser under-searches the band relative to the 40-point sweep the constant
# threshold gets, and biases the comparison toward the constant rule.
ETAS   = [round(0.1*i, 1) for i in range(21)]   # 0.0 .. 2.0
GAMMAS = [round(0.1*i, 1) for i in range(11)]   # 0.0 .. 1.0


def ksvd_point_scores(ch, a):
    mu, sd = pp.standardizer(ch.train_values)
    S_tr = pp.sliding_window(ch.train_values, a.window, mu, sd)
    S_te = pp.sliding_window(ch.test_values, a.window, mu, sd)
    D = sm.ksvd_dictionary(S_tr, n_atoms=a.atoms, n_nonzero=a.sparsity,
                           n_iter=a.ksvd_iter, seed=a.seed)
    w, _ = sm.admm_residual_scores(D, S_te, alpha=a.alpha, beta=a.beta)
    return det.windows_to_points(w, a.window, len(ch.test_values))


def best_cfar(scores, mask):
    """med + k*robust_sigma, k swept. Structure-free end of Axis 2."""
    med = np.median(scores)
    mad = 1.4826 * np.median(np.abs(scores - med))
    spread = mad if mad > 1e-6 else float(np.std(scores))
    best = {"f1": -1.0}
    for k in np.arange(0.5, 12.01, 0.25):
        pred = scores > med + k * spread
        pa = det.evaluate_pa(pred, mask)
        if pa["f1"] > best["f1"]:
            best = {"f1": pa["f1"], "pa": pa,
                    "pw": det.evaluate_mask(pred, mask), "k": float(k)}
    return best


def fixed_cfar(scores, mask, k=4.0):
    """The DEPLOYABLE rule: no sweep, no labels read to pick the threshold."""
    pred = scores > det.fixed_threshold(scores, k=k)
    return {"pa": det.evaluate_pa(pred, mask),
            "pw": det.evaluate_mask(pred, mask), "k": k}


def best_constant(scores, mask, n_grid=40):
    lo, hi = np.percentile(scores, 50), scores.max()
    best = {"f1": -1.0}
    for thr in np.linspace(lo, hi, n_grid):
        pred = scores > thr
        pa = det.evaluate_pa(pred, mask)
        if pa["f1"] > best["f1"]:
            best = {"f1": pa["f1"], "pa": pa, "pw": det.evaluate_mask(pred, mask)}
    return best


def best_ewma(scores, mask, period):
    best = {"f1": -1.0}
    for eta in ETAS:
        for gam in GAMMAS:
            pred, _, _, _ = det.ewma_detect(scores, period, lam=0.8,
                                            eta=eta, gamma=gam, floor=None)
            pa = det.evaluate_pa(pred, mask)
            if pa["f1"] > best["f1"]:
                best = {"f1": pa["f1"], "pa": pa,
                        "pw": det.evaluate_mask(pred, mask),
                        "eta": eta, "gamma": gam}
    return best


def micro(cs):
    tp=sum(c["tp"] for c in cs); fp=sum(c["fp"] for c in cs)
    fn=sum(c["fn"] for c in cs); tn=sum(c["tn"] for c in cs)
    P=tp/(tp+fp) if tp+fp else 0.0; R=tp/(tp+fn) if tp+fn else 0.0
    return P, R, (2*P*R/(P+R) if P+R else 0.0), (fp/(fp+tn) if fp+tn else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("data","telemetry","telemanom"))
    ap.add_argument("--max-channels", type=int, default=0, dest="max_channels")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--atoms", type=int, default=128)
    ap.add_argument("--sparsity", type=int, default=4)
    ap.add_argument("--ksvd-iter", type=int, default=10, dest="ksvd_iter")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=2.6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(CSV_DIR, exist_ok=True)
    chans = dl.load(a.data)
    if a.max_channels: chans = chans[:a.max_channels]

    rules = ["constant-oracle", "CFAR-swept", "CFAR-fixed",
             "EWMA-plain", "EWMA-period"]
    pw = {r: [] for r in rules}; pa = {r: [] for r in rules}
    rows = []
    for n, ch in enumerate(chans, 1):
        if len(ch.test_values) < a.window + 5 or not ch.anomaly_sequences:
            continue
        s = ksvd_point_scores(ch, a); m = ch.label_mask()
        p = det.estimate_period(ch.train_values)
        res = {"constant-oracle": best_constant(s, m),
               "CFAR-swept":      best_cfar(s, m),
               "CFAR-fixed":      fixed_cfar(s, m),
               "EWMA-plain":      best_ewma(s, m, 1),
               "EWMA-period":     best_ewma(s, m, p)}
        for r in rules:
            pw[r].append((ch.spacecraft, res[r]["pw"]))
            pa[r].append((ch.spacecraft, res[r]["pa"]))
        rows.append([ch.cid, ch.spacecraft, p]
                    + [round(res[r]["pa"]["f1"], 3) for r in rules]
                    + [round(res[r]["pw"]["f1"], 3) for r in rules])
        print(f"  [{n:>2}/{len(chans)}] {ch.cid:<6} p={p:<5} "
              + " ".join(f"{r.split('-')[0][:5]}={res[r]['pa']['f1']:.2f}" for r in rules),
              flush=True)

    print(f"\n  {'subset':<6}{'threshold':<14}{'pw_P':>7}{'pw_R':>7}{'pw_F1':>7}   "
          f"{'pa_P':>7}{'pa_R':>7}{'pa_F1':>7}{'pa_FPR':>8}")
    print("  " + "-"*78)
    summary=[]
    for sc in ("ALL","SMAP","MSL"):
        for r in rules:
            sub_pw=[c for s_,c in pw[r] if sc=="ALL" or s_==sc]
            sub_pa=[c for s_,c in pa[r] if sc=="ALL" or s_==sc]
            if not sub_pa: continue
            pP,pR,pF,_ = micro(sub_pw); aP,aR,aF,aFPR = micro(sub_pa)
            print(f"  {sc:<6}{r:<14}{pP:>7.3f}{pR:>7.3f}{pF:>7.3f}   "
                  f"{aP:>7.3f}{aR:>7.3f}{aF:>7.3f}{aFPR:>8.4f}")
            summary.append([sc,r,round(pP,3),round(pR,3),round(pF,3),
                            round(aP,3),round(aR,3),round(aF,3),round(aFPR,4)])
        print("  " + "-"*78)

    with open(os.path.join(CSV_DIR,"threshold_ablation.csv"),"w",newline="") as fh:
        w=csv.writer(fh); w.writerow(["subset","threshold","pw_P","pw_R","pw_F1",
                                      "pa_P","pa_R","pa_F1","pa_FPR"]); w.writerows(summary)
    with open(os.path.join(CSV_DIR,"threshold_ablation_per_channel.csv"),"w",newline="") as fh:
        w=csv.writer(fh)
        w.writerow(["chan","spacecraft","period"]
                   + [f"{r}_pa_F1" for r in rules] + [f"{r}_pw_F1" for r in rules])
        w.writerows(rows)
    print(f"\n[+] -> {CSV_DIR}/threshold_ablation.csv (+ _per_channel.csv)")


if __name__ == "__main__":
    main()
