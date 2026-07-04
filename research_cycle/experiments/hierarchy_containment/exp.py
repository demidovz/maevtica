#!/usr/bin/env python3
"""Causal hierarchy containment — ablate the parent (fruit), spare the child.
See PREREG.md for the frozen decision rule."""
import os, numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(max(1, os.cpu_count() or 4))
from transformer_lens import HookedTransformer

m = HookedTransformer.from_pretrained("gpt2", device="cpu"); m.eval()

def tid(w):
    return int(m.to_tokens(" " + w, prepend_bos=False)[0, 0])

FRUIT_T, VEG_T = tid("fruit"), tid("vegetable")

def resid_last(prompts, L):
    h = f"blocks.{L}.hook_resid_post"; out = []
    for p in prompts:
        _, c = m.run_with_cache(m.to_tokens(p), names_filter=h)
        out.append(c[h][0, -1].float())
    return torch.stack(out)

def logits_ablate(prompt, L=None, u=None):
    """Last-token logits, optionally projecting unit dir u out of resid_post at
    ALL positions of layer L."""
    if u is None:
        return m(m.to_tokens(prompt))[0, -1].float()
    h = f"blocks.{L}.hook_resid_post"
    def fn(r, hook, u=u):
        uu = u.to(r.dtype)
        r[:] = r - (r @ uu).unsqueeze(-1) * uu
        return r
    return m.run_with_hooks(m.to_tokens(prompt), fwd_hooks=[(h, fn)])[0, -1].float()

# exemplars decoupled from the target apple
FRUITS = ["banana", "orange", "grape", "peach", "mango", "pear"]
VEGS   = ["carrot", "potato", "broccoli", "onion", "celery", "spinach"]
frame  = "I bought a fresh {}"
NEUTRAL = ["The weather today is", "She opened the door and", "In the morning I like to",
           "The meeting will start at", "He walked down the", "They decided to"]

CAT_PROBES = [
    "An apple is a type of", "The apple is a kind of", "Botanically an apple is a",
    "At the store, apples are sold as a", "An apple belongs to the category of",
    "Apples are classified as a",
]
ID_PROBES = [
    ("The color of a ripe apple is usually", ["red", "green", "yellow", "blue", "purple"]),
    ("The shape of an apple is",             ["round", "square", "flat", "long", "thin"]),
    ("An apple tastes",                      ["sweet", "sour", "bitter", "salty", "spicy"]),
    ("When you bite into an apple it is",    ["juicy", "dry", "hard", "soft", "hollow"]),
    ("An apple grows on a",                  ["tree", "vine", "bush", "root", "stem"]),
]
# ORACLE: known NON-apple fruits (decoupled target for the positive control)
ORACLE_PROBES = ["A banana is a type of", "An orange is a type of", "A grape is a type of",
                 "A peach is a type of"]

def softmax(lg): return torch.softmax(lg, -1)

def membership_prob(lg):
    p = softmax(lg); f, v = float(p[FRUIT_T]), float(p[VEG_T])
    return f / (f + v + 1e-12)

def entropy(lg):
    p = softmax(lg); return float(-(p * torch.log(p + 1e-12)).sum())

# ---- clean baselines (once) ----
clean_cat_lg = [logits_ablate(p) for p in CAT_PROBES]
M0 = float(np.mean([membership_prob(lg) for lg in clean_cat_lg]))
clean_id_arg = []
I0_terms = []
for (p, cand) in ID_PROBES:
    lg = logits_ablate(p); ids = [tid(w) for w in cand]
    probs = softmax(lg); sub = np.array([float(probs[i]) for i in ids])
    arg = int(sub.argmax()); clean_id_arg.append(arg)
    I0_terms.append(sub[arg] / (sub.sum() + 1e-12))
I0 = float(np.mean(I0_terms))
clean_neut_H = float(np.mean([entropy(logits_ablate(p)) for p in NEUTRAL]))
clean_oracle_lf = [float(logits_ablate(p)[FRUIT_T]) for p in ORACLE_PROBES]

print(f"CLEAN: apple membership fruit-pref M0={M0:.3f} (>.5 => model knows apple=fruit)")
print(f"CLEAN: apple identity score I0={I0:.3f}; identity argmax = "
      + ", ".join(f"{ID_PROBES[i][1][clean_id_arg[i]]}" for i in range(len(ID_PROBES))))
print(f"CLEAN: neutral entropy={clean_neut_H:.3f}")
print(f"CLEAN: oracle logit(fruit) on known fruits = {[round(x,2) for x in clean_oracle_lf]}\n")

def run_layer(L):
    fr = resid_last([frame.format(w) for w in FRUITS], L).mean(0)
    vg = resid_last([frame.format(w) for w in VEGS], L).mean(0)
    d = fr - vg; u = d / d.norm()
    # oracle: fruit-logit drop on KNOWN fruits
    orc = float(np.mean([c - float(logits_ablate(p, L, u)[FRUIT_T])
                         for p, c in zip(ORACLE_PROBES, clean_oracle_lf)]))
    # apple membership ablated
    Ma = float(np.mean([membership_prob(logits_ablate(p, L, u)) for p in CAT_PROBES]))
    # apple identity ablated (candidate-normalised prob of clean-correct token)
    Ia_terms = []
    for (p, cand), arg in zip(ID_PROBES, clean_id_arg):
        lg = logits_ablate(p, L, u); ids = [tid(w) for w in cand]
        probs = softmax(lg); sub = np.array([float(probs[i]) for i in ids])
        Ia_terms.append(sub[arg] / (sub.sum() + 1e-12))
    Ia = float(np.mean(Ia_terms))
    Ha = float(np.mean([entropy(logits_ablate(p, L, u)) for p in NEUTRAL]))
    return dict(L=L, orc=orc, Ma=Ma, Ia=Ia, cohR=Ha / clean_neut_H)

rows = [run_layer(L) for L in [4, 6, 8, 10]]
print(f"{'L':>2} {'oracle':>7} {'M(abl)':>7} {'I(abl)':>7} {'md':>6} {'idd':>6} {'ret':>6} {'cohR':>6}")
for r in rows:
    md = (M0 - r['Ma']) / M0; idd = (I0 - r['Ia']) / I0; ret = r['Ia'] / I0
    r.update(md=md, idd=idd, ret=ret)
    print(f"{r['L']:>2} {r['orc']:>+7.2f} {r['Ma']:>7.3f} {r['Ia']:>7.3f} "
          f"{md:>+6.2f} {idd:>+6.2f} {ret:>6.2f} {r['cohR']:>6.2f}")

# frozen layer choice: max oracle effect
Lstar = max(rows, key=lambda r: r['orc'])
md, idd, ret, orc = Lstar['md'], Lstar['idd'], Lstar['ret'], Lstar['orc']
print(f"\nL* (max-oracle) = {Lstar['L']}: oracle={orc:+.2f} membership_drop={md:+.2f} "
      f"identity_drop={idd:+.2f} identity_retained={ret:.2f} cohR={Lstar['cohR']:.2f}")

print("\n=== VERDICT ===")
if orc < 1.0:
    print(f"BROKEN_MEASUREMENT (oracle {orc:+.2f} < 1.0) — ablation ineffective, refuse to conclude")
elif md < 0.10:
    print(f"REFUTED (target-inert): parent ablation works (oracle {orc:+.2f}) but apple "
          f"membership barely moves (md={md:+.2f}) — no containment on apple")
elif md >= 3 * max(idd, 0.0) and ret >= 0.6:
    print(f"SUPPORTED: membership_drop {md:+.2f} >= 3x identity_drop {idd:+.2f}, "
          f"identity_retained {ret:.2f} >= 0.6; oracle {orc:+.2f}")
elif md < 3 * max(idd, 0.0) or ret < 0.6:
    print(f"REFUTED (monolithic): identity falls with membership "
          f"(md={md:+.2f}, idd={idd:+.2f}, ret={ret:.2f}); category not separable")
else:
    print("INCONCLUSIVE")
