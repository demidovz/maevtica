# Map of ideas — is the concept hierarchy a navigable TREE? (2026-07-06)

**Verdict: SUPPORTED (both halves).** Inside gpt2, the concept hierarchy is a real —
if fuzzy — tree, and you can WALK it by category-arithmetic, cleanly and directionally.

## Measurement (calibrated, controlled)
- Model gpt2, layer 9. Oracle (steer a member by +its own category direction →
  that category's logit up) passed: +3.12. ✅ canary green.
- 6 categories (fruit, vegetable, bird, fish, tree, flower), single-token members.

## Q1 — TREE: does each thing sit under its right branch?
- Leave-one-out: a held-out word lands on its correct branch **67%** (chance 16.7% → **4× chance**).
- Category directions are **distinct axes**: mean pairwise |cos| = 0.34 (< 0.50).
- → **The tree is real but FUZZY** — two-thirds land right, a third wander. Not a crisp taxonomy.

## Q2 — WALK: can we navigate by arithmetic (subtract A, add B → land on B)?
Swept steering strength (this is what v2 got wrong — it used the oracle-MAX strength,
which over-drives and a random push flips everything too):

| strength | directed swap A→B lands on B | random push of same size derails |
|---|---|---|
| 0.25 | 95% | **0%** |
| **0.5** | **100%** | **12%** |
| 0.75 | 100% | 35% |
| 1.5 | 100% | 65% |
| 2.0 | 100% | 59% |

- **Clean directional window at gentle strength (0.25–0.5):** directed "fruit→bird" lands
  on bird ~100%, a random push of the same size does essentially nothing.
- At high strength it degrades into brute disruption (random also flips) — that over-drive
  is exactly why v2 falsely read BROKEN.
- → **WALK SUPPORTED**: the same arithmetic move (−A +B) works across the whole 6×6
  category matrix, directionally and cleanly.

## Honest caveats
- One small model (gpt2), 6 common categories, single-token words.
- The tree is fuzzy (67%), not a crisp taxonomy; concepts overlap.
- "Walk" = flipping the model's stated category in "A {w} is a type of ___" — a logit-level
  navigation, not a full behavioral rewrite.
- Prior art: Park 2024 (geometry of hierarchical concepts). We replicate the tree structure
  and add the CAUSAL navigability + the clean-window-vs-brute-disruption distinction.

## Process (honest — 3 tries, all caught by controls)
v1: direction built at the wrong token position → oracle failed (BROKEN, correctly).
v2: strength picked at oracle-MAX → over-driven, random flipped 82% (BROKEN, correctly).
v3: build direction where we measure + SWEEP strength for the clean directional window → clean SUPPORTED.
**Recurring lesson (3rd time): calibrate steering strength to the CLEAN DIRECTIONAL WINDOW
(directed works AND random doesn't), never to max-oracle intensification.**

**Second confirmed brick of the reflection-intelligence vision: the map of ideas inside
the model is a navigable tree you can steer by arithmetic.**
