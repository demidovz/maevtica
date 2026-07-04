# PREREG — Causal hierarchy containment (ablate the parent, spare the child)

**Date:** 2026-07-04  **Model:** gpt2 (cpu)  **ONE intervention:** project the
`fruit` direction OUT of the residual stream at every position of one layer L
(parameter-free ablation — no magnitude knob).

## Concept
If `apple` is represented compositionally as **fruit-membership component +
independent identity**, then ablating the *parent category* direction (`fruit`)
should knock out MEMBERSHIP ("apple is a type of fruit") while largely SPARING
IDENTITY (apple is red / round / sweet / juicy / grows on a tree). If instead
the concept is monolithic, identity collapses together with membership.

## Prediction (author)
Fruit-ablation drops MEMBERSHIP by a large margin while IDENTITY drops far less:
specifically **membership_drop >= 3 * identity_drop**, AND **identity stays above
0.6 of clean** (identity_retained >= 0.6). This is causal evidence apple =
fruit-component + independent identity.
FALSIFIED if identity falls as much as membership (monolithic, not separable),
or if membership barely moves. If membership barely moves AND the oracle is dead
-> BROKEN_MEASUREMENT.

## Operationalisation
- **Direction** v = mean_resid(fruit frames) - mean_resid(veg frames) at layer L,
  frame "I bought a fresh {word}" (last token = the food word). Exemplars are
  DECOUPLED from the target apple: fruits {banana,orange,grape,peach,mango,pear},
  vegs {carrot,potato,broccoli,onion,celery,spinach}. Unit-normalise -> u.
- **Ablation** = project u out of resid_post at ALL positions of layer L:
  r <- r - (r.u) u. One intervention, no alpha.
- **MEMBERSHIP(apple)** over 6 category probes ("An apple is a type of", ...):
  fruit-vs-veg preference as a probability p = P(' fruit')/(P(' fruit')+P(' vegetable')),
  averaged. M0 clean, Ma ablated. membership_drop = (M0-Ma)/M0.
- **IDENTITY(apple)** over 5 property probes, each with a fixed candidate set
  (color->{red,green,yellow,blue,purple}; shape->{round,...}; taste->{sweet,...};
  texture->{juicy,...}; grows-on->{tree,...}): candidate-normalised probability of
  the CLEAN-correct token, averaged. I0 clean, Ia ablated.
  identity_drop = (I0-Ia)/I0 ; identity_retained = Ia/I0.
- **COHERENCE** (honesty guard, not decisive): mean next-token entropy on 6
  neutral non-food prompts under the same ablation; report ratio to clean.

## Controls
- **ORACLE / positive control (decoupled target):** the SAME ablation must reduce
  fruit-classification of KNOWN fruits that are NOT the apple (banana/orange/grape
  "is a type of"). oracle = mean(clean logit(' fruit') - ablated logit(' fruit'))
  over those probes. Near-zero => the ablation does nothing => BROKEN_MEASUREMENT.
  min_oracle = 1.0 logit.
- **Layer choice** is frozen by rule (not by result): among L in {4,6,8,10}, pick
  the layer with the LARGEST oracle effect (most effective ablation), then read
  the apple membership/identity numbers there. This prevents cherry-picking the
  layer that most flatters the concept.

## DECISION RULE (frozen before running)
Let L* = argmax_L oracle(L). Read md=membership_drop, idd=identity_drop,
ret=identity_retained, orc=oracle, all at L*.
- **BROKEN_MEASUREMENT** if orc < 1.0 logit -> refuse to conclude.
- **SUPPORTED** if orc >= 1.0 AND md >= 0.10 AND md >= 3*max(idd,0) AND ret >= 0.6.
- **REFUTED (monolithic)** if orc >= 1.0 AND md >= 0.10 AND (md < 3*max(idd,0) OR ret < 0.6).
- **REFUTED (target-inert)** if orc >= 1.0 AND md < 0.10 (parent ablation works in
  general but does not touch apple's membership -> containment claim about apple fails).
- **INCONCLUSIVE** otherwise.
