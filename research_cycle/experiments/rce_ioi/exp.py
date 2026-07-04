#!/usr/bin/env python3
"""RCE on GPT-2-small IOI — see PREREG.md. venv: ~/.local/state/mst/crc-venv311"""
import json, os, time
import numpy as np, torch

torch.set_grad_enabled(False)
torch.set_num_threads(max(1, os.cpu_count() or 4))
torch.manual_seed(0); np.random.seed(0)

from transformer_lens import HookedTransformer

m = HookedTransformer.from_pretrained("gpt2", device="cpu"); m.eval()
m.set_use_attn_result(True)
nL, nH = m.cfg.n_layers, m.cfg.n_heads

pairs = [("John","Mary"),("Tom","Sarah"),("James","Anna"),("Paul","Kate"),
         ("Mark","Alice"),("Dan","Emma"),("Peter","Rose"),("Jack","Mary")]
prompts, io_ids, s_ids = [], [], []
for A, B in pairs:
    for a, b in ((A, B), (B, A)):  # ABBA + BABA
        prompts.append(f"When {a} and {b} went to the store, {b} gave a drink to")
        io_ids.append(int(m.to_tokens(" "+a, prepend_bos=False)[0,0]))
        s_ids.append(int(m.to_tokens(" "+b, prepend_bos=False)[0,0]))
toks = m.to_tokens(prompts)
assert toks.shape[0] == 16
io_ids = torch.tensor(io_ids); s_ids = torch.tensor(s_ids)
NB = toks.shape[0]
ar = torch.arange(NB)

def ld_of(logits):
    last = logits[:, -1].float()
    return (last[ar, io_ids] - last[ar, s_ids])

# ---- clean run: LD, z cache, DLA ----
zfilt = lambda n: n.endswith("hook_z")
rfilt = lambda n: n.endswith("hook_result") or n == "ln_final.hook_scale"
logits, zc = m.run_with_cache(toks, names_filter=zfilt)
clean_ld_vec = ld_of(logits)
clean_ld = float(clean_ld_vec.mean())
acc = float((clean_ld_vec > 0).float().mean())
z_clean = {L: zc[f"blocks.{L}.attn.hook_z"] for L in range(nL)}   # [b,pos,head,dh]
z_mean  = {L: z_clean[L].mean(0, keepdim=True) for L in range(nL)}
perm = torch.roll(ar, 1)

U = m.W_U  # [d_model, vocab]
udiff = (U[:, io_ids] - U[:, s_ids]).T.float()   # [b, d_model]
w_ln = m.ln_final.w.float()

def dla_from_cache(c):
    """[nL,nH] mean DLA to logit diff."""
    scale = c["ln_final.hook_scale"][:, -1].float()          # [b,1]
    out = np.zeros((nL, nH), np.float32)
    for L in range(nL):
        r = c[f"blocks.{L}.attn.hook_result"][:, -1].float() # [b,head,d_model]
        r = r - r.mean(-1, keepdim=True)
        r = r / scale.unsqueeze(1) * w_ln
        out[L] = (r * udiff.unsqueeze(1)).sum(-1).mean(0).numpy()
    return out

_, cc = m.run_with_cache(toks, names_filter=rfilt)
dla_clean = dla_from_cache(cc)
del cc

def hooks_for(S, proto):
    """S: set of (L,H). One hook per layer touched."""
    byL = {}
    for L, H in S: byL.setdefault(L, []).append(H)
    hs = []
    for L, Hs in byL.items():
        def fn(z, hook, L=L, Hs=Hs):
            for H in Hs:
                if proto == "zero":     z[:, :, H] = 0.0
                elif proto == "mean":   z[:, :, H] = z_mean[L][:, :, H]
                else:                   z[:, :, H] = z_clean[L][perm][:, :, H]
            return z
        hs.append((f"blocks.{L}.attn.hook_z", fn))
    return hs

def run_abl(S, proto, cache=False):
    if cache:
        with m.hooks(fwd_hooks=hooks_for(S, proto)):
            lg, c = m.run_with_cache(toks, names_filter=rfilt)
        return float(ld_of(lg).mean()), c
    lg = m.run_with_hooks(toks, fwd_hooks=hooks_for(S, proto))
    return float(ld_of(lg).mean()), None

protos = ["zero", "mean", "resample"]

# ---- raw total effects, all 144 heads x 3 protocols ----
t0 = time.time()
raw = {p: np.zeros((nL, nH), np.float32) for p in protos}
for p in protos:
    for L in range(nL):
        for H in range(nH):
            ldA, _ = run_abl({(L, H)}, p)
            raw[p][L, H] = clean_ld - ldA
print("raw effects done", round(time.time()-t0, 1), "s")

# ---- oracle ----
oracle_99 = float(abs(raw["zero"][9, 9]))
oracle_ok = clean_ld > 1.0 and acc > 0.9 and oracle_99 >= 0.5
print(f"ORACLE clean_ld={clean_ld:.3f} acc={acc:.2f} zero-abl 9.9 effect={oracle_99:.3f} ok={oracle_ok}")

# ---- eval subset (fixed rule) ----
mean_abs = np.mean([np.abs(raw[p]) for p in protos], 0)
flat = [(-mean_abs[L, H], L, H) for L in range(nL) for H in range(nH)]
flat.sort()
subset = [(L, H) for _, L, H in flat[:12]]
for nm in [(9, 9), (9, 6), (10, 0)]:
    if nm not in subset: subset.append(nm)
print("subset:", [f"{L}.{H}" for L, H in subset])

# ---- repair closure ----
def closure(seed, proto, thresh=0.1, cap=6):
    S = [seed]
    sgn = 1.0 if dla_clean[seed] >= 0 else -1.0
    ldA, c = run_abl(set(S), proto, cache=True)
    while len(S) - 1 < cap:
        d = dla_from_cache(c)
        score = sgn * (d - dla_clean)
        for L, H in S: score[L, H] = -np.inf
        L2, H2 = np.unravel_index(np.argmax(score), score.shape)
        if score[L2, H2] < thresh: break
        S.append((int(L2), int(H2)))
        ldA, c = run_abl(set(S), proto, cache=True)
    return S, clean_ld - ldA

t0 = time.time()
rce = {p: {} for p in protos}
clos = {p: {} for p in protos}
for p in protos:
    for seed in subset:
        S, eff = closure(seed, p)
        rce[p][seed] = eff
        clos[p][seed] = S
        print(f"{p:8s} seed {seed[0]}.{seed[1]}: k={len(S)-1} "
              f"closure={[f'{L}.{H}' for L,H in S[1:]]} rce={eff:+.3f} raw={raw[p][seed]:+.3f}")
print("closures done", round(time.time()-t0, 1), "s")

# ---- Spearman ----
def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return float((ra*rb).sum() / np.sqrt((ra*ra).sum()*(rb*rb).sum()))

def min_pairwise(vals):
    out = {}
    for i in range(3):
        for j in range(i+1, 3):
            out[f"{protos[i]}-{protos[j]}"] = spearman(vals[protos[i]], vals[protos[j]])
    return out, min(out.values())

raw_sub  = {p: np.array([raw[p][s] for s in subset]) for p in protos}
rce_sub  = {p: np.array([rce[p][s] for s in subset]) for p in protos}
raw_all  = {p: raw[p].flatten() for p in protos}
sp_raw_sub, min_raw_sub = min_pairwise(raw_sub)
sp_rce_sub, min_rce_sub = min_pairwise(rce_sub)
sp_raw_all, min_raw_all = min_pairwise(raw_all)

print("raw subset spearman:", {k: round(v,3) for k,v in sp_raw_sub.items()}, "min", round(min_raw_sub,3))
print("rce subset spearman:", {k: round(v,3) for k,v in sp_rce_sub.items()}, "min", round(min_rce_sub,3))
print("raw all-144 spearman:", {k: round(v,3) for k,v in sp_raw_all.items()}, "min", round(min_raw_all,3))

# ---- P2 grading ----
B = {(9,0),(9,7),(10,1),(10,2),(10,6),(10,10),(11,2),(11,9)}
N = {(10,7),(11,10)}
p2 = {}
for seed in [(9,9),(9,6),(10,0)]:
    per = {}
    for p in protos:
        S = clos[p][seed]; rep = S[1:]
        per[p] = {"k": len(rep),
                  "repairers": [f"{L}.{H}" for L, H in rep],
                  "all_in_BuN": all(r in B | N for r in rep),
                  "any_in_B": any(r in B for r in rep)}
    p2[f"{seed[0]}.{seed[1]}"] = per

def p2_ok(p):
    return all(per[p]["k"] <= 3 and per[p]["all_in_BuN"] and per[p]["any_in_B"]
               for per in p2.values())
p2_refuted = any(per["mean"]["k"] > 3 or not per["mean"]["all_in_BuN"] for per in p2.values())

# ---- verdicts per prereg ----
if not oracle_ok:
    v1 = v2 = overall = "BROKEN_MEASUREMENT"
else:
    if min_rce_sub >= 0.9 and min_raw_sub <= 0.6: v1 = "SUPPORTED"
    elif min_rce_sub <= min_raw_sub or min_rce_sub < 0.7: v1 = "REFUTED"
    else: v1 = "INCONCLUSIVE"
    if p2_ok("mean"): v2 = "SUPPORTED"
    elif p2_refuted: v2 = "REFUTED"
    else: v2 = "INCONCLUSIVE"
    overall = ("SUPPORTED" if v1 == v2 == "SUPPORTED"
               else "REFUTED" if "REFUTED" in (v1, v2) else "INCONCLUSIVE")

res = {"clean_ld": clean_ld, "acc": acc, "oracle_zero_9.9": oracle_99,
       "oracle_ok": bool(oracle_ok),
       "subset": [f"{L}.{H}" for L, H in subset],
       "spearman_raw_subset": sp_raw_sub, "min_raw_subset": min_raw_sub,
       "spearman_rce_subset": sp_rce_sub, "min_rce_subset": min_rce_sub,
       "spearman_raw_all144": sp_raw_all, "min_raw_all144": min_raw_all,
       "raw_subset": {p: [float(x) for x in raw_sub[p]] for p in protos},
       "rce_subset": {p: [float(x) for x in rce_sub[p]] for p in protos},
       "closures": {p: {f"{s[0]}.{s[1]}": [f"{L}.{H}" for L, H in clos[p][s]]
                        for s in subset} for p in protos},
       "p2": p2, "verdict_p1": v1, "verdict_p2": v2, "verdict": overall}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result.json")
json.dump(res, open(out, "w"), indent=1)
print("VERDICT p1:", v1, "| p2:", v2, "| overall:", overall)
print("wrote", out)
