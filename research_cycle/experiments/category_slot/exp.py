#!/usr/bin/env python3
"""Category-slot detachability — minimal falsification. See PREREG.md."""
import os, numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(max(1, os.cpu_count() or 4))
from transformer_lens import HookedTransformer

m = HookedTransformer.from_pretrained("gpt2", device="cpu"); m.eval()
rng = np.random.default_rng(0)

def tid(w):
    return int(m.to_tokens(" " + w, prepend_bos=False)[0, 0])

FRUIT_T, VEG_T = tid("fruit"), tid("vegetable")

def resid_last(prompts, L):
    h = f"blocks.{L}.hook_resid_post"; out = []
    for p in prompts:
        _, c = m.run_with_cache(m.to_tokens(p), names_filter=h)
        out.append(c[h][0, -1].float())
    return torch.stack(out)

def logits_last(prompt, L=None, vec=None):
    if vec is None:
        return m(m.to_tokens(prompt))[0, -1].float()
    h = f"blocks.{L}.hook_resid_post"
    def fn(r, hook, v=vec): r[:, -1, :] = r[:, -1, :] + v.to(r.dtype); return r
    return m.run_with_hooks(m.to_tokens(prompt), fwd_hooks=[(h, fn)])[0, -1].float()

# exemplars (decoupled from target apple)
FRUITS = ["banana", "orange", "grape", "peach", "mango", "pear"]
VEGS   = ["carrot", "potato", "broccoli", "onion", "celery", "spinach"]
frame  = "I bought a fresh {}"
NEUTRAL = ["The weather today is", "She opened the door and", "In the morning I like to",
           "The meeting will start at", "He walked down the", "They decided to"]

# apple category probes
CAT_PROBES = [
    "An apple is a type of", "The apple is a kind of", "Botanically an apple is a",
    "At the store, apples are sold as a", "An apple belongs to the category of",
    "Apples are classified as a",
]
# apple identity probes: (prompt, candidate property set)
ID_PROBES = [
    ("The color of a ripe apple is usually", ["red", "green", "yellow", "blue", "purple"]),
    ("The shape of an apple is",             ["round", "square", "flat", "long", "thin"]),
    ("An apple tastes",                      ["sweet", "sour", "bitter", "salty", "spicy"]),
    ("When you bite into an apple it is",    ["juicy", "dry", "hard", "soft", "hollow"]),
    ("An apple grows on a",                  ["tree", "vine", "bush", "root", "stem"]),
]
# genuine vegetables for the ORACLE
ORACLE_PROBES = ["A carrot is a type of", "A potato is a type of", "A broccoli is a type of"]

def logitdiff(lg):  # +ve => prefers vegetable
    return float(lg[VEG_T] - lg[FRUIT_T])

def entropy(lg):
    p = torch.softmax(lg, -1); return float(-(p * torch.log(p + 1e-12)).sum())

# clean baselines (probe-level, reused)
clean_cat = [logits_last(p) for p in CAT_PROBES]
clean_cat_ld = [logitdiff(l) for l in clean_cat]
clean_id_arg = []
for p, cand in ID_PROBES:
    lg = logits_last(p); ids = [tid(w) for w in cand]
    clean_id_arg.append(int(np.argmax([float(lg[i]) for i in ids])))
clean_neut_H = np.mean([entropy(logits_last(p)) for p in NEUTRAL])
clean_oracle_ld = [logitdiff(logits_last(p)) for p in ORACLE_PROBES]

print(f"clean apple category logitdiff (veg-fruit, want <0): "
      f"mean={np.mean(clean_cat_ld):+.2f}  per={[round(x,1) for x in clean_cat_ld]}")
print(f"clean neutral entropy = {clean_neut_H:.3f}")
print(f"clean identity argmax = {clean_id_arg} (0=first candidate=correct prop)")
print()

def eval_vec(vec, L):
    # FLIP
    flips = 0
    for base_ld, p in zip(clean_cat_ld, CAT_PROBES):
        s_ld = logitdiff(logits_last(p, L, vec))
        if base_ld < 0 and s_ld > 0: flips += 1
    FLIP = flips / len(CAT_PROBES)
    # PRESERVE
    pres = 0
    for (p, cand), base_arg in zip(ID_PROBES, clean_id_arg):
        lg = logits_last(p, L, vec); ids = [tid(w) for w in cand]
        if int(np.argmax([float(lg[i]) for i in ids])) == base_arg: pres += 1
    PRES = pres / len(ID_PROBES)
    # COHERENCE
    H = np.mean([entropy(logits_last(p, L, vec)) for p in NEUTRAL])
    return FLIP, PRES, H / clean_neut_H

def oracle(vec, L):
    d = [logitdiff(logits_last(p, L, vec)) - c for p, c in zip(ORACLE_PROBES, clean_oracle_ld)]
    return float(np.mean(d))

neut_norm = resid_last(NEUTRAL, 8).norm(dim=-1).mean().item()  # reference scale
rows, best = [], None
print(f"{'L':>2} {'alpha':>5} {'FLIP':>5} {'PRES':>5} {'cohR':>5} {'oracle':>7}")
for L in [6, 8, 10]:
    fr = resid_last([frame.format(w) for w in FRUITS], L).mean(0)
    vg = resid_last([frame.format(w) for w in VEGS], L).mean(0)
    d = vg - fr; u = d / d.norm()
    nn = resid_last(NEUTRAL, L).norm(dim=-1).mean().item()
    for a in [2, 4, 6, 8]:
        vec = u * (a * nn / 10.0)   # alpha in units of 0.1*meanNorm
        FLIP, PRES, cohR = eval_vec(vec, L)
        orc = oracle(vec, L)
        rows.append((L, a, FLIP, PRES, cohR, orc))
        print(f"{L:>2} {a:>5} {FLIP:>5.2f} {PRES:>5.2f} {cohR:>5.2f} {orc:>+7.2f}")
        ok = FLIP >= 0.7 and PRES >= 0.7 and 0.8 <= cohR <= 1.2
        if ok and (best is None):
            best = (L, a, FLIP, PRES, cohR, orc, u, nn)

# overall oracle (best correct-direction magnitude across grid, at strong alpha)
max_oracle = max(r[5] for r in rows)
print(f"\nmax oracle across grid = {max_oracle:+.2f} (min_oracle=1.0)")

# random control at winning config (or strongest-FLIP config if none win)
ref = best[:2] if best else max(rows, key=lambda r: (r[2], r[3]))[:2]
Lr, ar = ref
frr = resid_last([frame.format(w) for w in FRUITS], Lr).mean(0)
vgr = resid_last([frame.format(w) for w in VEGS], Lr).mean(0)
dnorm = (vgr - frr).norm().item()
nnr = resid_last(NEUTRAL, Lr).norm(dim=-1).mean().item()
rnd = torch.tensor(rng.standard_normal(vgr.shape[0]), dtype=torch.float32)
rnd = rnd / rnd.norm() * (ar * nnr / 10.0)
rFLIP, rPRES, rcohR = eval_vec(rnd, Lr)
print(f"random control @ L={Lr} alpha={ar}: FLIP={rFLIP:.2f} PRES={rPRES:.2f} cohR={rcohR:.2f}")

print("\n=== VERDICT ===")
if max_oracle < 1.0:
    print("BROKEN_MEASUREMENT (oracle < 1.0) — refuse to conclude")
elif best and rFLIP < 0.5:
    print(f"SUPPORTED: window L={best[0]} alpha={best[1]} "
          f"FLIP={best[2]:.2f} PRES={best[3]:.2f} cohR={best[4]:.2f}; random FLIP={rFLIP:.2f}")
elif best and rFLIP >= 0.5:
    print(f"INCONCLUSIVE: window exists but random noise also flips (FLIP={rFLIP:.2f})")
else:
    print("REFUTED: oracle passes but NO config met FLIP>=0.7 & PRESERVE>=0.7 within coherence band")
