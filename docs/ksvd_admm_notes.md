# K-SVD-ADMM — verified equations, reasoning, & implementation notes

Reference: He et al., "Application of sparse representation method based on
K-SVD-ADMM in anomaly detection of satellite telemetry," 2022 Global Reliability
and PHM (Yantai).

This file is the implementation contract for `src/telemetry_anomaly/`. The ADMM
updates were re-derived from the augmented Lagrangian and **verified correct**;
the paper has a few notation typos that are corrected here so the code matches
the math, not the printed symbols. Each section states *what* the equation does
and *why the method needs it*, so the chain of reasoning can be followed
end-to-end.

---

## 0. Why sparse representation for telemetry — the reasoning

Housekeeping telemetry defeats fixed alarm limits for two reasons: normal
behaviour is large and periodic (so a limit tight enough to catch faults fires
on every nominal swing), and the most dangerous faults are **collective
anomalies** — individually in-range samples whose *shape over time* is wrong
(drifts, stuck values, distorted cycles). Detecting a wrong *shape* requires a
model of what normal shapes look like.

The sparse-representation bet is: **normal telemetry is locally redundant** —
every normal window is approximately a combination of a *few* recurring local
patterns. So: learn a dictionary `D` of those patterns from clean telemetry;
then any window that is normal can be reconstructed from a few atoms of `D`,
and any window that *cannot* be sparsely reconstructed is, by definition of the
model, anomalous. The reconstruction residual is the anomaly evidence. This is
the same reconstruction-residual principle as the RF layer
(`docs/framework_chapter_outline.md`) with the model richness turned all the
way up: from one fixed baseline template to a *learned, overcomplete* pattern
library.

Why *overcomplete* (more atoms than the window length)? Because normal
telemetry has more distinct local shapes than any orthogonal basis has room
for; overcompleteness plus a sparsity constraint lets each window pick the few
atoms that fit it, instead of forcing all windows through one small subspace.
(That "one small subspace" alternative is exactly the PCA baseline we compare
against in Phase 3 — the comparison isolates what overcompleteness buys.)

## 1. Problem chain (eqs 1 -> 6) — consistent

- **Eq 1 (SR):** `min_x ||y - Dx||_2^2  s.t. ||x||_0 <= T0`.
  `y in R^N` (signal), `D in R^{N x M}` (M atoms), `x in R^M` (sparse code).
  *Meaning:* "explain window `y` using at most `T0` dictionary atoms."
- **Eq 3 (dictionary learning):** `min_{X,D} ||S - DX||_F^2  s.t. ||x_j||_0 <= T0`,
  solved by K-SVD on the training sample matrix `S_train in R^{w x L1}`.
  *Meaning:* choose the atoms themselves so that clean training windows are
  sparsely explainable. This is the offline phase; it requires anomaly-free data.
- **Eq 4 (online model):** `S_test = D*X_test + E`.
  *This is the crucial modelling move:* instead of computing the residual
  after the fact, a residual matrix `E` is put *into the model* and estimated
  jointly with the codes. `E` carries the part of the signal the *normal*
  dictionary cannot explain -> the anomaly evidence, separated out by the
  optimisation itself.
- **Eqs 5 -> 6 (convex relaxation):** the ideal sparsity counts (`l0` norms) are
  combinatorial, so both are relaxed to their tightest convex surrogates:
  - `||E||_{2,0} -> ||E||_{2,1} = sum_i ||E(:,i)||_2` (column-group sparsity).
    *Why groups:* an anomaly hits a whole window/time-step, not scattered
    entries — so whole *columns* of `E` should switch on or off together.
  - `||X||_0 -> ||X||_{1,1}` (plain elementwise sparsity on the codes).
  - Eq 6: `min_{X,E} (1/2)||S - DX - E||_F^2 + alpha*||X||_{1,1} + beta*||E||_{2,1}`.
    *Read it as a competition:* the data term wants everything explained; `alpha`
    taxes dictionary usage; `beta` taxes declaring anomalies. A window becomes
    an anomaly only when explaining it with atoms is *more expensive* than
    paying the `beta` toll — `beta` is literally the detection sensitivity knob.

## 2. K-SVD (Algorithm 1) — standard Aharon et al.

For each atom `k`: `E_k = Y - sum_{j!=k} d_j * x_T^j` (here `x_T^j` = the *j-th
row* of X), restrict columns to the samples that use atom `k` -> `E_k^R`,
take `SVD(E_k^R) = U S V^T`, then update
`d_k <- u_1` (first left singular vector) and the coefficient row
`x_T^k <- sigma_1 * v_1`. Textbook K-SVD.

*Why SVD:* with all other atoms frozen, "find the best replacement for atom `k`
and its coefficients" is exactly a rank-1 approximation of the residual matrix
`E_k^R`, and Eckart–Young says the leading singular pair *is* the optimal
rank-1 approximation. Restricting to the columns that already use atom `k` is
what preserves sparsity — otherwise the update would recruit every sample.

## 3. ADMM (Algorithm 2) — re-derived, correct

*Why ADMM at all:* eq 6 is convex but has two coupled non-smooth penalties.
ADMM splits it into sub-problems that each have a **closed-form** solution, by
duplicating `X` into a copy `Z` (constraint `Z = X`) and enforcing the copy
through the augmented Lagrangian (eq 8):

```
L = (1/2)||S - DX - E||_F^2 + alpha*||Z||_{1,1} + beta*||E||_{2,1}
      + <M, Z - X> + (mu/2)||Z - X||_F^2
```

Each variable then gets the sub-problem it is easiest to solve:

**X-update** — with the non-smooth terms shifted onto `Z` and `E`, this is pure
least squares (set grad_X L = 0):
```
(D^T D + mu*I) X = D^T (S - E) + M + mu*Z
X = (D^T D + mu*I)^{-1} ( D^T (S - E) + M + mu*Z )
```
**Z-update** — the `l1` proximal step, i.e. element-wise soft-threshold `S_t(.)`:
```
P = X - M/mu ;  gamma = alpha/mu ;  Z = S_gamma(P)
```
**E-update** — the `l2,1` proximal step: *block* soft-threshold, column-wise,
with `Q = S - D X` (the current unexplained part):
```
E(:,i) = max(0, 1 - beta/||Q(:,i)||_2) * Q(:,i)     # = 0 if ||Q(:,i)||_2 <= beta
```
*This line is the detector.* A column whose unexplained energy is below `beta`
is zeroed — declared normal; above `beta` it survives into `E` — declared
anomalous evidence. The group norm acts on whole columns, which is exactly the
collective-anomaly modelling choice of eq 5.

**Dual + penalty:**
```
M <- M + mu*(Z - X) ;  mu <- rho*mu     (rho > 1)
```
**Anomaly score (eq 9):** `score(j) = ||E(:,j)||_2` — the per-window norm of
the evidence the optimisation itself refused to explain as normal.

## 4. Adaptive threshold (Algorithm 3, eqs 10-12)

*Why adaptive:* periodic telemetry produces residual peaks at every period even
when healthy; a fixed threshold either fires on them or is blind between them.
The paper's answer is to reference the running statistic **one period back**:

- **Eq 10 (period-referenced EWMA):** `U_t = lambda*U_{t-p} + (1-lambda)*a_t`
  (uses the previous *period* `t-p`, not `t-1`), so the threshold "remembers"
  what this phase of the cycle normally scores.
- **Eqs 11-12 (two-sided band):** `tau_up = U_t*(1+eta)`, `tau_down = U_t*(1-gamma)`.
- **Reset rule:** when a point is flagged, set `U_t = U_{t-p}` so the anomaly does
  not poison the running average for later points. Keep it.
- **Why two-sided:** a *high* score = pattern not representable by the normal
  dictionary (novelty); a *low* score = signal reconstructed too easily
  (e.g. a frozen / stuck sensor — their anomaly 3, which trips the LOWER bound).
  Do not drop the lower bound assuming residuals are one-sided.

**What we found in practice (Phase 2, framework-relevant):** the band of
eqs 11-12 is *multiplicative* — its width scales with `U_t`. When the K-SVD
dictionary reconstructs normal data near-perfectly and ADMM's `beta` step zeroes
normal columns, the score is near-binary (~0 baseline, sparse peaks), `U_t -> 0`,
the band collapses, and everything over-flags. Our fix: an **additive floor**
on the band (`detector.ewma_detect(floor=...)`). That additive margin is
*exactly* the RF layer's CFAR term `max(k*MAD, min_excess)` — i.e. the
telemetry method's own failure pushed us to the threshold the RF layer already
used. This is the strongest single piece of evidence for the unified-framework
claim (CFAR = the non-periodic generalisation of the periodic EWMA).

---

## 5. PAPER TYPOS to correct in code (these will bite you)

1. **Eq 7 drops the `1/2` and the square** shown in eq 6. Keep the data term as
   `(1/2)||S - DX - E||_F^2` or the X-update is not the least-squares solution.
2. **Algorithm 2 "Output X_test in R^{N x L2}" is a dimension typo.** With
   `D in R^{N x M}` the codes are `X_test in R^{M x L2}`; residual
   `E in R^{N x L2}` (same shape as `S_test`).
3. **`D_i`, `E_i` subscripts in step 1 are spurious** — it is just `D` and `E^k`.
4. **`N` vs `w` conflation:** the `N` of eq 1 (signal dimension) *is* the window
   length `w` of the telemetry section. Same number, two symbols.
5. **3-block ADMM (X, Z, E):** classic 2-block convergence guarantees do not
   strictly apply. It works empirically with `rho > 1`. Note as a caveat; do not
   claim guaranteed convergence.

## 6. Reference hyperparameters (their case study)

- Sparse coding: `T0 = 4`, OMP for the K-SVD coding step.
- Dictionary: `K = 1500` atoms, K-SVD <= 50 iterations.
- ADMM: `alpha = 1`, `beta = 2.6`, `mu0 = 1`, `rho = 1.25`, <= 100 iterations.
- EWMA: `lambda = 0.8`; sweep `eta in [0,2] step 0.1`, `gamma in [0,1] step 0.1`
  to maximise F1.
- Window: `w = 60` (their anomalies 1,2,4) or `w = 100` (anomaly 3).
- Our Phase-2/3 runs use `atoms = 128`, K-SVD 10-12 iterations for speed;
  `K = 1500` is oversized for SMAP/MSL channel lengths.

## 7. Notes for SMAP/MSL (our data, not theirs)

- *Why this dataset:* the paper validates on a single private dataset. SMAP/MSL
  (NASA telemanom) is public, its train split is anomaly-free (-> dictionary
  learning) and its test split carries labelled anomaly *sequences* (collective
  anomalies) — a perfect structural match. Crucially, the LSTM-NDT method the
  paper compares against was *published on this very dataset*, giving us a
  head-to-head with published numbers instead of re-implementations.
- SMAP/MSL channels are command-driven and **not all periodic** -> the
  period-referenced EWMA may not transfer. Fallback: the RF layer's CFAR
  (`med + k*MAD`) threshold, the non-periodic generalization. This is
  the bridge to the unified-framework chapter (`framework_chapter_outline.md`).
- Scoring: the community often reports *point-adjusted* metrics on SMAP/MSL,
  which credit an entire anomaly segment for a single detected point and are
  known to inflate (Kim et al. 2022). We report **both** point-wise and
  point-adjusted, because — as Phase 3 confirmed — the metric choice alone can
  flip the winner.

## Phased build — plan and what each phase actually taught us

- **Phase 1 (done):** loader + sliding window + naive dictionary (random train
  windows) + **OMP** reconstruction + residual score + **fixed** threshold,
  validated on a synthetic periodic generator.
  *Finding:* spike and frequency-shift anomalies detected (recall 1.0), but the
  frozen/stuck-sensor channel scored **zero recall** — a frozen segment gives a
  *low* residual and the one-sided threshold never sees it. This independently
  reproduced the paper's motivation for the two-sided band (section 4) before we
  had implemented it: the failure predicted by the theory appeared on schedule.
- **Phase 2 (done):** true **K-SVD + ADMM** (sections 2-3) + **EWMA** threshold
  (section 4). On the clean synthetic: all 4 channels detected, including the
  frozen one (recall 0 -> 0.57) — the learned oscillatory atoms make a flat
  segment unexplainable, which is what fixes Phase 1's miss.
  *Finding:* the multiplicative EWMA band degenerates on well-reconstructed
  data (see section 4) -> additive CFAR floor. We did **not** keep tuning the
  synthetic until the paper's adaptive-vs-fixed improvement reappeared — that
  would have been circular; the degeneration and its fix are the honest result.
- **Phase 3 (done):** full SMAP/MSL run (81 channels) against a controlled
  **PCA reconstruction baseline** — same pipeline, same thresholding, the only
  difference being dense-linear subspace vs sparse-overcomplete dictionary —
  plus the published LSTM-NDT numbers for context.
  *Result (micro-avg):* K-SVD-ADMM wins **point-wise** (F1 0.522 vs 0.437,
  higher precision, lower FPR -> tighter localisation); PCA wins
  **point-adjusted** (F1 0.841 vs 0.786) by firing broadly and collecting
  whole-segment credit. Per spacecraft, both methods fall below LSTM-NDT on SMAP
  and above it on MSL (LSTM-NDT: 0.89 SMAP / 0.564 MSL,
  point-adjusted, protocol differs — context only).
  *Takeaway:* the sparse model buys localisation precision, not raw coverage,
  and the **scoring metric decides the apparent winner** — which is why both
  are reported. Artifacts: `results/metrics/telemetry/*.csv`,
  `results/figures/telemetry/comparison.png`.
