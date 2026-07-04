# PREREGISTRATION — Process-Pullback Geometry: shape forecasting (2026-07-04)

## Hypothesis under test
The data process (its mixed-state / belief-state geometry, computable from the
process definition alone) determines the topology of the activation manifold a
transformer trained on that process exhibits — not model-internal factors.

## Domains (5, geometry computable BEFORE touching any network)
Sequences start from a KNOWN initial hidden state (BOS), so the belief state is
an exact function of the token history. Emission distributions are injective in
the hidden state, so a loss-optimal predictor must distinguish all states.

D1 clock12: hidden r in Z12, tokens {0,1,2} = increments {1,2,3};
   P(k|r) ~ exp(1.2*cos(2*pi*r/12 - 2*pi*k/3)). Ground-truth config: 12 points
   on a closed curve. PREDICTED Betti signature: (b0=1, b1=1) CIRCLE.
D2 clock8: same with modulus 8. PREDICTED: (1,1) CIRCLE.
D3 flag: states {pre,post}; pre: P=(.45,.45,.10) over {0,1,TRIG}; TRIG -> post;
   post: P=(.8,.2,0). PREDICTED: (2,0) TWO CLUSTERS.
D4 counter: c in {0..8} start 4, reflecting walk c+=(tok?+1:-1) clamped;
   P(1|c)=0.15+0.7*c/8. Ground truth: 9 evenly spaced collinear points.
   PREDICTED: (1,0) LINE (contractible, connected).
D5 twoflag: flags f1,f2 (triggers tok4/tok5, p=.06 each while unset);
   within {0,1}: P(0)=.8 if f1 else .2 (mass .5); within {2,3}: P(2)=.8 if f2
   else .2 (mass .5, both after renormalizing trigger mass). Ground truth:
   4 points forming a square. PREDICTED: (4,0) FOUR CLUSTERS.

Distinct predicted labels used: {(1,1),(1,0),(2,0),(4,0)} -> 4 classes.

## Model & training (fixed in advance)
From-scratch GPT: 2 layers, d_model=64, 4 heads, MLP x4, learned pos emb,
seq len 64 (BOS + 63 tokens), AdamW lr 3e-3, batch 64, 3000 steps, CPU,
seed 0. One preregistered extension to 8000 steps if the loss gate fails.

## Measurement
Activation = residual stream after final block (pre-final-LN), positions
t in [20,62], grouped by ground-truth hidden state s_t; centroid per state
(>=3000 eval sequences). Geometry classified on the centroid config.

CLASSIFY(points P): diam = max pairwise distance.
  b1_hat = count of Rips H1 bars with persistence > T1*diam   (T1 = 0.25)
  b0_hat = # single-linkage components at cut T0*diam          (T0 = 0.45)
Betti signature = (b0_hat, b1_hat). Match = equals the predicted signature.

## Calibration lock (BEFORE any model activations are computed)
- ORACLE (positive control): CLASSIFY(ground-truth emission-prob configs) must
  return the predicted signature for all 5 domains. If not, the measurement is
  broken; thresholds T0/T1 may be adjusted ONLY on control data, then locked.
- NEGATIVE control: 200 draws of n=12 (and n=8) iid Gaussian points in R^64;
  false-circle rate (b1_hat>=1) must be < 5%.

## Gates
- Loss gate per domain: model CE - ground-truth CE < 0.05 nats on a held-out
  eval set. Domains failing after the 8000-step extension are excluded
  ("not learned"); geometry claims do not apply to them.

## Decision rule (locked)
Let k = # learned domains, m = # of those whose measured signature matches the
prediction. Null: uniform guess over the 4 used classes, p0 = 1/4.
- SUPPORTED  iff k >= 4 and m = k  (p = (1/4)^k <= 0.0039 < 0.01).
- REFUTED    iff any learned circle domain (D1/D2) passes the loss gate with
  b1_hat = 0 (provably-circular process represented without a loop at matched
  loss — the preregistered falsifier), OR m <= expected-by-chance (m <= 1).
- INCONCLUSIVE otherwise (incl. k < 4, or broken oracle -> BROKEN_MEASUREMENT).

## CALIBRATION AMENDMENT (2026-07-04, before any model was trained)
Oracle v1 failed on control data: clock8 (eccentric emission ellipse, sparse
8-pt sampling -> b0/b1 broke) and twoflag (4-pt square gives a spurious Rips
H1 bar, pers/diam=0.293, overlapping true-circle clock12 at 0.266 -> not
threshold-separable). Classifier v2, fixed on CONTROL DATA ONLY, locked before
any model run:
  1. Center; project onto PCA axes with variance > 0.05 * top; whiten kept axes.
  2. b0: single-linkage components at cut T0*diam, T0=0.45 (unchanged).
  3. b1: only defined for n >= 6 points (a 4-cycle is not 1-manifold evidence);
     count H1 bars with persistence > T1*diam, T1=0.25 (unchanged).
Stress controls added: noisy ellipses (aspect 4), noisy collinear 9-pt, noisy
square, 2 blobs. All must classify to their true signature; Gaussian FPR < 5%.
Second control finding: clock emission softness 1.2 makes the TRUE mixed-state
curve a rounded triangle (3-fold symmetric, whitening is a no-op), so 8-point
sampling under-samples it -> ground truth itself reads (2,0). Domain amendment
(pre-model): alpha 1.2 -> 0.7 in D1/D2 so the first Fourier harmonic dominates
and the true curve is near-round. Predictions unchanged.
Third control finding: whitened clock8 single-linkage MST bottleneck = 0.484
of diam (8 pts on a still slightly triangular curve) vs min inter-cluster
separation 0.707 in every cluster-type control. T0 0.45 -> 0.55 (between, with
margin both sides). All thresholds now locked; models trained only after this.

## Known limits (stated in advance)
- Finite-state processes: topology of finite point configs is scale-relative;
  T0/T1 are fixed fractions of diameter, calibrated on controls only.
- Single small architecture, final-layer readout only, one seed per domain.
- "Survived" here != proven; it is one preregistered shot at falsification.
