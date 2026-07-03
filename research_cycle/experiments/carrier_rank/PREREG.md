# Preregistration — Carrier rank ρ(P) vs SAE feature-splitting

Date: 2026-07-04. Written BEFORE running. Decision rule fixed here.

## Concept under test
"Carrier rank" ρ(P): the representational dimensionality of a property P.
Prediction: the count of SAE atoms significantly aligned with P (|cos|>0.3)
grows with ρ(P), while single-atom max-cosine recovery falls as ρ grows.
Concrete claims: ρ=1 properties have exactly one dominant matching atom
(cos>0.8); ρ≥3 properties have no atom above 0.5 and spread across ≥ρ atoms.

## Construction of ρ (ground truth by design, validated by measurement)
Base concepts C = {dog, war, love, music, snow}, 6 prompts each + 6 neutral
prompts. Concept direction u_c = unit(mean last-token resid_pre L7 act over
concept prompts − neutral mean). A property of rank ρ is the UNION of ρ base
concepts; its pooled direction v_P = unit(mean over ALL P prompts − neutral mean).

Fixed property list (10):
- ρ=1: {dog}, {war}, {love}, {music}, {snow}
- ρ=2: {dog,war}, {love,music}, {war,snow}
- ρ=3: {dog,war,love}, {music,snow,war}

Validity gates (if violated → design invalid, INCONCLUSIVE, not a verdict on
the concept): (V1) pairwise |cos(u_c, u_c')| < 0.5 for all concept pairs;
(V2) participation ratio of singular values of each ρ=3 property's component
mean-diff matrix ≥ 2.0 (constructed rank is real in activation space).

## Operationalization ("atom aligned with P")
Primary: |cos(decoder atom, v_P)| against the pooled direction — this makes
"single-atom max-cosine recovery falls with ρ" a real empirical claim (an atom
matching one component of a rank-3 union has cos ≈ 0.85/√3 ≈ 0.49 with v_P).
Secondary (reported, not decisive): projection norm of atom onto the
orthonormalized component subspace.

Atoms: unit-normalized W_dec rows of the trained SAE
jbloom/GPT2-Small-SAEs-Reformatted, blocks.7.hook_resid_pre (d_sae=24576),
restricted to LIVE atoms (mean encoder activation > 1e-6 over the same
60-prompt diverse corpus used in the lever experiment).

Metrics per property: maxcos = max_a |cos(a, v_P)|;
count03 = #{a : |cos| > 0.3}; count05, count08 likewise.

## Oracle / positive control (MANDATORY)
- O1 (pipeline oracle): feed a known live atom's decoder row through the same
  pipeline as v → maxcos must be > 0.99. Near-zero/low ⇒ BROKEN_MEASUREMENT.
- O2 (relevance oracle): each ρ=1 concept direction's maxcos must exceed the
  99.5th pct of the null (max |cos| over live atoms for 200 random unit dirs)
  — else concept dirs are SAE-irrelevant noise ⇒ BROKEN_MEASUREMENT.

## Decision rule (fixed; falsification conditions from the prediction)
Compute Spearman rank correlation r_count = spearman(ρ, count03) and
r_max = spearman(ρ, maxcos) over the 10 properties.
- BROKEN_MEASUREMENT if O1 or O2 fails.
- INCONCLUSIVE(design) if V1 or V2 fails.
- REFUTED if (F1) r_count ≤ 0.3 OR mean count03(ρ=3) ≤ mean count03(ρ=1)
  ["split-count independent of ρ"], OR (F2) any ρ=3 property has
  maxcos ≥ 0.8 ["high-ρ property still captured by a single atom"].
- SUPPORTED if r_count ≥ 0.6 AND r_max ≤ −0.3 AND no F2. The concrete
  thresholds (ρ=1: exactly one atom >0.8; ρ=3: maxcos<0.5, count03≥3) are
  reported as pass/fail per property — misses are caveats, honestly stated,
  and if ≥4/5 of the ρ=1 properties lack ANY atom >0.5, the "ρ=1 ⇒ one
  dominant atom" half of the concrete claim is declared refuted-in-detail
  even if the trend supports the concept.
- Else INCONCLUSIVE.

## Budget
gpt2-small CPU, ~40 forward passes + one 24576×768 cosine matmul. Minutes.

## Amendment (run 2) — preregistered 2026-07-04 AFTER run 1, BEFORE run 2
Run 1 (result.json): oracle PASSED (O1=1.0, concept maxcos 0.33–0.42 > null
0.17) but validity gates V1+V2 FAILED — raw diff-in-means directions share a
large common "topic vs neutral" component (pairwise cos 0.47–0.63, ρ=3
participation ratio ~1.9). Verdict: INCONCLUSIVE_DESIGN, as preregistered.
Fix (construction only, decision rule UNCHANGED): center concept diff vectors
by their mean across the 5 concepts before normalizing; property pooled
direction = unit(mean of its components' centered diffs). Same gates, same
oracles (O2 on centered dirs), same thresholds, same decision rule.
Note: centering 5 vectors induces expected pairwise cos ≈ −0.25; V1 remains
|cos| < 0.5. Output: result2.json.
