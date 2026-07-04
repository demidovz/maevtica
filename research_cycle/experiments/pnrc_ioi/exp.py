#!/usr/bin/env python3
"""PNRC on GPT-2-small IOI — see PREREG.md. venv: ~/.local/state/mst/crc-venv311"""
import json, os, time
import numpy as np, torch

torch.set_grad_enabled(False)
torch.set_num_threads(max(1, os.cpu_count() or 4))
torch.manual_seed(0); np.random.seed(0)

from transformer_lens import HookedTransformer

m = HookedTransformer.from_pretrained("gpt2", device="cpu"); m.eval()
nL, nH = m.cfg.n_layers, m.cfg.n_heads

pairs = [("John","Mary"),("Tom","Sarah"),("James","Anna"),("Paul","Kate"),
         ("Mark","Alice"),("Dan","Emma"),("Peter","Rose"),("Jack","Mary")]
prompts, io_ids, s_ids = [], [], []
for A, B in pairs:
    for a, b in ((A, B), (B, A)):
        prompts.append(f"When {a} and {b} went to the store, {b} gave a drink to")
        io_ids.append(int(m.to_tokens(" "+a, prepend_bos=False)[0,0]))
        s_ids.append(int(m.to_tokens(" "+b, prepend_bos=False)[0,0]))
toks = m.to_tokens(prompts)
assert toks.shape[0] == 16
io_ids = torch.tensor(io_ids); s_ids = torch.tensor(s_ids)
NB = toks.shape[0]; ar = torch.arange(NB)

def ld_of(logits):
    last = logits[:, -1].float()
    return (last[ar, io_ids] - last[ar, s_ids])

zfilt = lambda n: n.endswith("hook_z")
logits, zc = m.run_with_cache(toks, names_filter=zfilt)
clean_ld_vec = ld_of(logits)
clean_ld = float(clean_ld_vec.mean())
acc = float((clean_ld_vec > 0).float().mean())
z_clean = {L: zc[f"blocks.{L}.attn.hook_z"] for L in range(nL)}   # [b,pos,head,dh]
z_mean  = {L: z_clean[L].mean(0, keepdim=True).expand_as(z_clean[L]) for L in range(nL)}
perm = torch.roll(ar, 1)
z_res   = {L: z_clean[L][perm] for L in range(nL)}

protos = ["zero", "mean", "resample"]
alphas = [1/3, 2/3, 1.0]

def target(L, H, p):
    if p == "zero":  return torch.zeros_like(z_clean[L][:, :, H])
    if p == "mean":  return z_mean[L][:, :, H]
    return z_res[L][:, :, H]

# ε from clean cache: mean ||z'-z_clean|| / mean ||z_clean||, per head/proto/alpha
def eps_of(L, H, p, a):
    d = a * (target(L, H, p) - z_clean[L][:, :, H])          # [b,pos,dh]
    num = d.norm(dim=-1).mean().item()
    den = z_clean[L][:, :, H].norm(dim=-1).mean().item()
    return num / (den + 1e-9)

def run_point(L, H, p, a):
    tgt = target(L, H, p)
    def fn(z, hook):
        z[:, :, H] = z[:, :, H] + a * (tgt - z[:, :, H].float()).to(z.dtype)
        return z
    lg = m.run_with_hooks(toks, fwd_hooks=[(f"blocks.{L}.attn.hook_z", fn)])
    return float(ld_of(lg).mean()) - clean_ld

# NOTE: inside the hook z is the CLEAN z (single-head hook, nothing upstream
# altered at that site), so z + a*(tgt - z) == z_clean + a*(tgt - z_clean).

# ---- pooled isotonic (PAV), both directions, best R² ----
def _pav(y, w):
    y = y.astype(np.float64).copy(); w = w.astype(np.float64).copy()
    n = len(y); lvl = list(range(n))
    i = 0
    vals = y.tolist(); wts = w.tolist(); idx = [[k] for k in range(n)]
    changed = True
    while changed:
        changed = False
        k = 0
        while k < len(vals) - 1:
            if vals[k] > vals[k+1] + 1e-12:
                tw = wts[k] + wts[k+1]
                vals[k] = (vals[k]*wts[k] + vals[k+1]*wts[k+1]) / tw
                wts[k] = tw; idx[k] += idx[k+1]
                del vals[k+1], wts[k+1], idx[k+1]
                changed = True
                if k: k -= 1
            else:
                k += 1
    out = np.empty(n)
    for v, ii in zip(vals, idx):
        for j in ii: out[j] = v
    return out

def iso_r2(eps, dl):
    o = np.argsort(eps)
    e, d = np.asarray(eps)[o], np.asarray(dl)[o]
    # merge exact-tie eps by averaging (PAV handles order; ties fine as-is)
    w = np.ones_like(d)
    sst = ((d - d.mean())**2).sum()
    if sst < 1e-12: return 1.0, "flat"
    fit_inc = _pav(d, w)
    fit_dec = -_pav(-d, w)
    r2i = 1 - ((d - fit_inc)**2).sum() / sst
    r2d = 1 - ((d - fit_dec)**2).sum() / sst
    return (float(r2i), "inc") if r2i >= r2d else (float(r2d), "dec")

# ---- main loop ----
t0 = time.time()
EPS = np.zeros((nL, nH, 3, 3), np.float32)   # [L,H,proto,alpha]
DL  = np.zeros((nL, nH, 3, 3), np.float32)
for L in range(nL):
    for H in range(nH):
        for pi, p in enumerate(protos):
            for ai, a in enumerate(alphas):
                EPS[L, H, pi, ai] = eps_of(L, H, p, a)
                DL[L, H, pi, ai]  = run_point(L, H, p, a)
    print(f"layer {L} done {round(time.time()-t0,1)}s", flush=True)

# ---- oracle (amended prereg: pipeline checks, not the repair-confounded total effect) ----
tgt99 = target(9, 9, "zero")
def _fn99(z, hook):
    z[:, :, 9] = z[:, :, 9] + 1.0 * (tgt99 - z[:, :, 9].float()).to(z.dtype)
    return z
with m.hooks(fwd_hooks=[("blocks.9.attn.hook_z", _fn99)]):
    _, c99 = m.run_with_cache(toks, names_filter=lambda n: n == "blocks.9.attn.hook_z")
mech_norm = float(c99["blocks.9.attn.hook_z"][:, :, 9].norm())
nm_full = max(float(abs(DL[L, H, pi, 2])) for L, H in [(9, 9), (9, 6), (10, 0)] for pi in range(3))
oracle_ok = clean_ld > 1.0 and acc > 0.9 and mech_norm < 1e-4 and nm_full >= 0.5
print(f"ORACLE clean_ld={clean_ld:.3f} acc={acc:.2f} mech_norm={mech_norm:.2e} "
      f"nm_max_full_effect={nm_full:.3f} ok={oracle_ok}")

# ---- grading ----
heads = [(L, H) for L in range(nL) for H in range(nH)]
maxabs = np.abs(DL).reshape(nL, nH, 9).max(-1)
effect = [(L, H) for L, H in heads if maxabs[L, H] >= 0.05]

r2 = {}; direction = {}
for L, H in effect:
    r2[(L, H)], direction[(L, H)] = iso_r2(EPS[L, H].flatten(), DL[L, H].flatten())

C  = {h for h in effect if r2[h] > 0.9}
NC = set(effect) - C

F = set()
for L, H in effect:
    full = DL[L, H, :, 2]
    if full.max() > 0.05 and full.min() < -0.05:
        F.add((L, H))

# direct conditional-disagreement at matched eps
gaps = []
for L, H in effect:
    e = EPS[L, H].flatten(); d = DL[L, H].flatten()
    pid = np.repeat(np.arange(3), 3)
    rng_e = e.max() - e.min(); rng_d = d.max() - d.min()
    if rng_e < 1e-9 or rng_d < 1e-9: continue
    best = None
    for i in range(9):
        for j in range(i+1, 9):
            if pid[i] != pid[j] and abs(e[i]-e[j]) <= 0.1*rng_e:
                g = abs(d[i]-d[j]) / rng_d
                best = g if best is None else max(best, g)
    if best is not None: gaps.append(best)
med_gap = float(np.median(gaps)) if gaps else None

frac_collapse = len(C) / len(effect) if effect else float("nan")
uni = F | NC
jac = 1.0 if not uni else len(F & NC) / len(uni)

# ---- verdict per prereg ----
if not oracle_ok:
    v1 = v2 = overall = "BROKEN_MEASUREMENT"
else:
    v1 = "SUPPORTED" if frac_collapse >= 0.5 else "REFUTED"
    if not F and not NC: v2 = "SUPPORTED"
    elif jac >= 0.75: v2 = "SUPPORTED"
    elif jac <= 0.25: v2 = "REFUTED"
    else: v2 = "INCONCLUSIVE"
    overall = ("SUPPORTED" if v1 == v2 == "SUPPORTED"
               else "REFUTED" if "REFUTED" in (v1, v2) else "INCONCLUSIVE")

fmt = lambda s: sorted(f"{L}.{H}" for L, H in s)
res = {"clean_ld": clean_ld, "acc": acc,
       "oracle_mech_norm": mech_norm, "oracle_nm_max_full_effect": nm_full,
       "oracle_ok": bool(oracle_ok),
       "n_effect": len(effect), "n_flat": 144 - len(effect),
       "frac_collapse_r2_gt_0.9": frac_collapse,
       "median_r2_effect": float(np.median([r2[h] for h in effect])) if effect else None,
       "sign_flip_heads": fmt(F), "non_collapse_heads": fmt(NC),
       "jaccard_F_NC": jac,
       "median_matched_eps_gap": med_gap, "n_heads_with_matched_pairs": len(gaps),
       "r2_by_head": {f"{L}.{H}": round(r2[(L,H)], 4) for L, H in effect},
       "worst10_r2": sorted(((round(r2[h],3), f"{h[0]}.{h[1]}") for h in effect))[:10],
       "verdict_p1": v1, "verdict_p2": v2, "verdict": overall}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result.json")
json.dump(res, open(out, "w"), indent=1)
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "curves.npz"), EPS=EPS, DL=DL)
print("effect heads:", len(effect), "| collapse frac:", round(frac_collapse,3),
      "| median R2:", res["median_r2_effect"])
print("sign-flip:", fmt(F))
print("non-collapse:", fmt(NC))
print("jaccard(F,NC):", round(jac,3), "| median matched-eps gap:", med_gap)
print("VERDICT p1:", v1, "| p2:", v2, "| overall:", overall)
print("wrote", out)
