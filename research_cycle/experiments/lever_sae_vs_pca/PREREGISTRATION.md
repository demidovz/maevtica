# Preregistration — Behavioral lever: trained SAE vs PCA vs random dictionary

Date: 2026-07-03. Preregistered BEFORE running (decision rule fixed here, cannot move).

## Concept under test
"Behavioral lever (dose-response feature)". A *lever* for a fixed behavior is a
single direction in the residual stream that, when added at increasing coefficient
(dose), monotonically increases that behavior (here: the last-token logit of a
target concept token) with a substantial effect.

## Prediction (the thing to falsify)
For a fixed behavior, the number of directions passing the lever test is small
(O(1-10)) and does NOT increase when a trained SAE dictionary is swapped for a PCA
or random dictionary of **equal size**. The best trained-SAE lever's effect lies
within the CI of the best PCA lever, and PCA occasionally wins.
**Falsified if** trained SAEs yield *strictly more*, OR *strictly stronger*, levers
than PCA at matched dictionary count.

## Setup
- Model: gpt2-small (transformer_lens), CPU. Layer L=7, hook `resid_pre`.
- Real trained SAE: `jbloom/GPT2-Small-SAEs-Reformatted` blocks.7.hook_resid_pre
  (d_sae=24576, d_in=768). This is a well-trained public SAE (strong, fair competitor).
- Matched dictionary count **N=512** for all three dictionaries:
  - **SAE**: 512 decoder rows (`W_dec`) of the 512 features with highest mean
    encoder activation over the corpus (the "live"/used learned features), unit-normalized.
  - **PCA**: top-512 principal components of centered `resid_pre` L7 activations over
    the same corpus, unit-normalized.
  - **RANDOM**: 512 Gaussian directions, unit-normalized. (negative control)
- Corpus for ranking/PCA: ~120 diverse short prompts, all token positions.
- Behaviors (fixed target tokens, run independently): ` dog`, ` war`, ` love`.
- Probes: M=6 neutral continuation prompts; behavior score = mean over probes of
  the target token's last-position logit.

## Lever test (per direction, per behavior)
Steer `resid_pre` L7 last position: `r += c * d̂`, doses c ∈ {0, D/4, D/2, D} with
D = 0.5 × mean resid_pre L7 norm. Each direction tested at ±sign; keep the sign that
maximizes Δlogit_T at max dose. A direction is a **lever** iff:
1. dose-response monotonic: Δlogit_T strictly increasing across the 4 doses
   (each increment > 0.05), AND
2. magnitude: Δlogit_T(max dose) ≥ τ = **3.0** logits.
- `count` = # levers in the dictionary. `best` = max Δlogit_T(max dose) over its directions.

## Oracle / positive control (MANDATORY)
Oracle direction = diff-in-means(concept prompts about T − neutral prompts),
unit-normalized, run through the *same* lever test. It MUST pass with
Δlogit_T(max dose) ≥ min_oracle = **3.0**. If oracle < min_oracle → `BROKEN_MEASUREMENT`
(pipeline broken, report nothing else). Negative control: RANDOM dict count expected ≈ 0.

## Decision rule (fixed)
Per behavior, with 95% bootstrap CI over probes on `best_SAE` and `best_PCA`:
- `strictly_more`   := count_SAE ≥ 2·count_PCA AND (count_SAE − count_PCA) ≥ 5
- `strictly_stronger` := best_SAE > upper95(best_PCA CI)   (non-overlapping, SAE above)
Aggregate over the 3 behaviors (majority):
- **REFUTED** if `strictly_more` OR `strictly_stronger` holds in ≥2 of 3 behaviors.
- **SUPPORTED** if NEITHER holds in ≥2 of 3 behaviors (oracle valid). "Survived", not "proven".
- **INCONCLUSIVE** otherwise, or if oracle broken → **BROKEN_MEASUREMENT**.
