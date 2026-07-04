# Preregistration — Behavioral Quotient Feature (intervention-equivalence class)

Date: 2026-07-04. Written BEFORE running. Decision rule fixed here.

## Concept under test
Behavior quotients the dictionary: many SAE latents collapse into few
behavioral coordinates that belong to the MODEL, not the dictionary.
Two computable clauses:
(i) effective rank k90 (singular directions covering 90% of behavioral
    variance of latent-steering effects) stays within 10% when SAE
    dictionary width grows 4x;
(ii) two SAE latents whose decoder vectors project mainly onto the same
    quotient coordinate produce steering effects with logit-diff-vector
    cosine > 0.8 even when their decoder cosine is < 0.1.
Falsified if same-coordinate latents steer incoherently or if effective
rank tracks dictionary size.

## Setup
- Model: gpt2-small (TransformerLens, CPU), layer 6, hook
  `blocks.6.hook_resid_post` (= OpenAI "resid_post_mlp" location).
- SAEs: OpenAI v5 TopK SAEs for gpt2-small, layer 6, widths 32768 and
  131072 — exactly 4x, same training recipe, same location. (This is the
  only ready-made same-recipe 4x-width pair on a CPU-sized model.)
- Probes: 8 fixed diverse prompts (in exp.py). Steering: add
  alpha * unit(decoder_i) to resid_post L6 at ALL positions,
  alpha = 0.5 * mean clean last-token resid norm over probes
  (harness steer_c=0.5 convention).
- Effect vector e_i = mean over 8 probes of (steered − clean) last-token
  logits (R^50257), one padded batch pass per latent.
- Latent sample: alive latents = appear ≥1 time in per-token top-k on a
  ~60-prompt diverse corpus; sample N=256 uniformly per SAE, seed=0.

## Clause (i) — effective rank
Per SAE: E = row matrix of the 256 effect vectors, mean-centered across
rows; k90 = min k with cumsum(s^2)/sum(s^2) ≥ 0.90 (singular values of
centered E). Equal N for both SAEs makes the comparison fair.
- supported-i if |k90_128k − k90_32k| / min(k90) ≤ 0.10
- refuted-i  if max(k90)/min(k90) ≥ 1.5 (rank tracks dictionary;
  literal 4x tracking would give ~4)
- inconclusive-i otherwise; ALSO inconclusive-i (sample-limited) if both
  k90 ≥ 0.8·N.
Diagnostic (reported, not decisive): k90 of the decoder matrices D
themselves — if k90(E) ≈ k90(D), rank is dictionary-geometry-driven and
we say so.

## Clause (ii) — same-coordinate coherence
Quotient coordinates estimated WITHOUT the 128k data (decoupling): ridge
fit on the 32k sample, W^T = (D^T D + λI)^{-1} D^T E with
λ = 1e-2 · trace(D^T D)/768; SVD of W^T (768×V) → top K = k90_32k left
singular vectors U_K = quotient coordinates in residual space.
For 128k latents: p_i = U_K^T d_i; coordinate c_i = argmax|p_i|;
dominance σ_i = p_i[c_i]^2 / ||p_i||^2. Qualifying pairs (i,j): both
σ ≥ 0.5, same c, decoder |cos(d_i,d_j)| < 0.1. If < 5 pairs at σ≥0.5,
fall back to σ≥0.4 (preregistered ladder, one step only). Cap 200 pairs.
Sign correction (preregistered): the prediction is about the same
UNSIGNED coordinate; metric = median over pairs of
sign(p_i[c]·p_j[c]) · cos(e_i, e_j).
Fair null (steering ANY direction shares a generic logit shift, which
inflates cosines): median |cos(e_i,e_j)| over 200 random 128k pairs with
decoder |cos| < 0.1 and NO same-coordinate requirement.
- supported-ii if median signed cos > 0.8 AND (median − null) ≥ 0.2
- refuted-ii  if median signed cos < 0.4
- inconclusive-ii otherwise; also inconclusive-ii if null itself > 0.8
  (measurement cannot discriminate) or < 5 qualifying pairs after ladder.

## Oracle / positive control (MANDATORY — verdict refused if broken)
- O1 (split-half reliability): per steered latent, cos(mean effect over
  probes 0–3, mean over probes 4–7); median over all 512 latents must be
  ≥ 0.6, else BROKEN_MEASUREMENT (effect vectors are noise).
- O2 (pair-machinery positive control): cross-SAE near-duplicate pairs
  (decoder cos > 0.9, up to 50): median effect cos must be > 0.8. If
  < 10 such pairs exist, fallback control: 20 latents steered at alpha
  and 1.1·alpha → median effect cos must be > 0.9.
- O3 (reported): mean ||e_i|| vs repeat-clean-run noise floor.

## Decision rule (fixed)
BROKEN_MEASUREMENT if O1 or O2 fails. Else:
SUPPORTED iff supported-i AND supported-ii;
REFUTED iff refuted-i OR refuted-ii;
INCONCLUSIVE otherwise.

## Honest limits (stated in advance)
One model (gpt2-small), one layer, one steering scale, one intervention
type (add), logit-diff at last token only, N=256 latent sample. Survival
here ≠ proven; it means "survived a minimal decisive test in this regime".

## Budget
~535 batched forward passes on CPU (~5–10 min), 1 GB SAE download.

## Result (2026-07-04, written AFTER the run; rule above unchanged)
Oracles PASSED: O1 split-half median 0.828; O2 rescale control 0.999
(only 2 cross-SAE dup pairs existed, preregistered fallback used);
mean ||e|| ≈ 192–195 vs noise floor 0.0.
- Clause (i): SUPPORTED — k90_32k=96 vs k90_128k=93 (3.2% diff, bound 10%)
  at equal N=256; not sample-limited. Diagnostic: decoder-geometry k90 was
  148/143 — behavioral rank (≈95) is measurably BELOW dictionary geometry
  rank, so behavior genuinely compresses the dictionary.
- Clause (ii): INCONCLUSIVE_NO_PAIRS — the premise population is EMPTY:
  0 of 256 sampled 128k latents put ≥40% of their quotient projection
  energy on any single coordinate (K=96). "Projects mainly onto one
  quotient coordinate" essentially never happens; the clause is vacuous
  as stated. Exploratory (post-hoc, NOT decisive): over 500 low-decoder-cos
  pairs, spearman(quotient-profile sim, effect cos)=0.52 — a graded
  relation exists; but the 3 pairs with profile-sim>0.8 had median effect
  cos 0.32, well below the predicted 0.8.
Overall preregistered verdict: INCONCLUSIVE (i supported, ii vacuous;
exploratory evidence leans against the strong >0.8 coherence claim).
Regime: gpt2-small L6, add-steering at 0.5·resid-norm, last-token logits,
N=256/SAE, one seed. Survived-in-part ≠ proven.
