# External test of "Causal Role Carrier" — preregistration

Written BEFORE running, so the threshold can't be moved to fit the result.
This is the "domain research" step the Studio's own `benchmark_011/final_report.md`
demanded: does the concept beat existing practice, or is it a renamed bundle?

## Concept under test

**Causal Role Carrier (CRC):** an equivalence class of internal states /
features / directions / circuits that preserves the same *intervention-stable
input→output role* across prompts, bases, model instances, and scale.

**Core prediction (#1, the one we test):**
> Cross-model mechanisms matched by high causal-role overlap transfer
> steering/editing effects **better** than mechanisms matched only by feature
> labels or activation similarity.

## Minimal falsifiable experiment

- **Models:** Pythia-160m and Pythia-410m (same family, different scale — the
  "across scale / model instance" case CRC targets). Shared tokenizer ⇒ output
  (vocabulary) space is common to both models; residual spaces differ
  (d_model 768 vs 1024).
- **Behaviors (K):** K distinct *token-promotion* steering directions, one per
  semantic category (e.g. animals, colors, numbers, countries, body-parts …).
  For model M and behavior k we compute a steering vector `v[M,k]` at a fixed
  layer from contrastive prompt pairs. Ground truth: behavior k in A corresponds
  to behavior k in B.
- **The task:** given behavior k's mechanism in A, *infer* which of B's K
  mechanisms is the match. Three inference methods:
  1. **random** — floor.
  2. **activation-similarity (geometry)** — the standard baseline (RSA/cosine).
     Given a fair chance: align B's residual space to A's via a linear map
     learned from paired token embeddings (both models embed the same vocab),
     then cosine-match directions.
  3. **CRC (causal role)** — match by *causal effect signature*: the vector of
     logit changes over the shared vocabulary produced when the mechanism is
     applied. Because effects live in the common output space, they compare
     across models with **no alignment**. This is the concept's claim.
- **Metrics** (over K behaviors × S seeds):
  - **top-1 matching accuracy** — did the method pick B's true behavior-k mechanism?
  - **transfer effect** — apply the matched B mechanism as steering in B, measure
    logit mass moved toward category k's tokens (how well the intended effect
    actually transfers).

## Preregistered decision rule

Let `acc_crc`, `acc_geom`, `acc_rand` be mean top-1 accuracies.

- **CRC SUPPORTED (worth real follow-up)** iff `acc_crc − acc_geom ≥ 0.15`
  AND `acc_crc` is clearly above random, on held-out behaviors, stable across seeds.
- **CRC REJECTED as renamed bundle** iff `acc_crc ≤ acc_geom + 0.05`
  (causal-role matching buys ~nothing over geometry).
- **Inconclusive** in between — report honestly, note what a bigger test needs.

Same rule applies to the transfer-effect metric as a secondary check.

## Honest limits (stated up front)

- Two small same-family models, one intervention type (additive steering),
  token-promotion behaviors only. A positive result is a *tracer bullet*, not
  proof; it would justify scaling to cross-family models, SAE-dictionary
  baselines, and edit/ablation transfer.
- We are testing prediction #1 only. #2 (SAE features collapsing) and #3
  (scaling survival) are out of scope for this first run.
