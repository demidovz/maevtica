# Preregistration — Causal-effect conservation under SAE feature splitting

Date: 2026-07-04. Model: gpt2, resid_post layer 6. Activations: existing
acts_L6.npy (19968 x 768, wikitext) reused from width_persistent_latent run.

## Concept under test
"Causally-Conserved Concept": when a parent SAE latent at width W splits into
daughters at width 2W, the **vector sum of the daughters' KL-steering signatures
reconstructs the parent's steering signature** within tolerance — EVEN THOUGH
each daughter's per-latent reconstruction share (variance / activation mass)
varies wildly. Falsified if the daughter causal signatures fail to sum
specifically to the parent (causal effect as width-dependent / non-conserved as
reconstruction share).

## Operationalization
- Train two ReLU-L1 SAEs on the SAME centered/scaled activations:
  parent W=128, daughters 2W=256 (same train_sae as width_persistent_latent).
- Decoder columns unit-normalized -> each latent = a direction d.
- Split assignment: each daughter -> nearest parent by decoder cosine (argmax).
  daughters(p) = daughters assigned to p. A "split" = parent with |daughters|>=2.
- Steering signature s(d): add v = alpha * d to resid_post L6 at the last token
  (alpha = 0.5 * mean resid norm, the harness steer_c). Signature =
  mean over P=12 fixed probes of (logits_steered - logits_clean)[last] in R^vocab.
  KL magnitude = mean_probes KL(softmax(steered) || softmax(clean)).
- Signatures use alpha*(UNIT dir), i.e. every latent steered at EQUAL magnitude
  -> the sum ignores reconstruction share entirely (the strong form of the claim).

## Conservation metric (primary)
For each split parent p:
  recon_p = sum_{j in daughters(p)} s(d_{c_j})      (unweighted vector sum)
  cos_true = cos(recon_p, s(d_p))
Specificity control: cos_rand = cos(recon_p, s(d_{p'})) for random OTHER parents
p' (mean of several draws). This removes the confound that ALL steering
signatures share a common component (e.g. a generic frequent-token boost);
only specificity beyond that common component counts.
  PRIMARY = Delta = median(cos_true) - median(cos_rand).

## Premise contrast (reported, not gating)
Within each split parent, per-daughter reconstruction share = activation
frequency * mean active value. Report median coefficient of variation across
splits — expect HIGH (shares vary wildly) to confirm the premise the claim rests
on. Also report geometric-only control: cos(sum d_{c_j}, d_p).

## ORACLE / positive control (must pass or verdict = BROKEN_MEASUREMENT)
(a) Steering actually perturbs the distribution: median parent-latent steering
    KL >= 0.02 nats. Random-direction steering KL reported as scale reference.
(b) Signatures are signal, not noise: split the 12 probes into two halves,
    compute each parent-latent signature on each half; median cross-half cosine
    >= 0.30. If signatures are noise, all cosine matching is meaningless.
Either failing => BROKEN_MEASUREMENT (do NOT read a verdict past it).

## Base-rate gate
If #split parents (|daughters|>=2) < 8 => INCONCLUSIVE (too few splits to judge).

## DECISION RULE (frozen before running)
- oracle (a) fails OR (b) fails ................... BROKEN_MEASUREMENT
- else #splits < 8 ............................... INCONCLUSIVE
- else median(cos_true) >= 0.50 AND Delta >= 0.15 . SUPPORTED
- else Delta <= 0.05 ............................. REFUTED (causal effect not conserved)
- else ........................................... INCONCLUSIVE
