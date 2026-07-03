import glob, numpy as np, torch, os
from safetensors.torch import load_file
from scipy.stats import pearsonr, spearmanr, ttest_1samp
torch.set_grad_enabled(False); torch.set_num_threads(max(1, os.cpu_count() or 4))
rng = np.random.default_rng(0)

L = 8
STEER_C = 4.0
SNAP = glob.glob("/home/friemann/.cache/huggingface/hub/models--jbloom--GPT2-Small-SAEs-Reformatted/snapshots/*/")[0]
w = load_file(f"{SNAP}blocks.{L}.hook_resid_pre/sae_weights.safetensors")
W_dec = w["W_dec"].float().numpy()               # [d_sae, d_in]
dec_unit = W_dec / (np.linalg.norm(W_dec, axis=1, keepdims=True) + 1e-9)
d_sae, d_in = W_dec.shape

from transformer_lens import HookedTransformer
m = HookedTransformer.from_pretrained("gpt2", device="cpu"); m.eval()
HOOK = f"blocks.{L}.hook_resid_pre"

# ---- reader Gram G over layers >= L (from reader_direction/exp.py) ----
G = np.zeros((d_in, d_in), dtype=np.float64)
for layer in range(L, m.cfg.n_layers):
    b = m.blocks[layer]; mats = []
    for W in [b.attn.W_Q, b.attn.W_K, b.attn.W_V]:
        Wn = W.detach().numpy(); mats.append(Wn.transpose(1, 0, 2).reshape(d_in, -1))
    mats.append(b.mlp.W_in.detach().numpy())
    R_ = np.concatenate(mats, axis=1); G += R_ @ R_.T
evals, V = np.linalg.eigh(G)
lam_max = evals[-1]
top = V[:, -1]; bot = V[:, 0]
rho_top = np.sqrt(max(top @ G @ top, 0)); rho_bot = np.sqrt(max(bot @ G @ bot, 0))
rr = rng.standard_normal((2000, d_in)); rr /= np.linalg.norm(rr, axis=1, keepdims=True)
rho_rand = np.sqrt(np.clip(np.einsum('ij,ij->i', rr @ G, rr), 0, None))
OB_ratio = rho_top / (rho_bot + 1e-12)
print(f"OB lam_max {lam_max:.3f} rho_top {rho_top:.3f} rho_bot {rho_bot:.4g} ratio {OB_ratio:.2f} "
      f"rho_rand {rho_rand.mean():.3f} (expect {np.sqrt(np.trace(G)/d_in):.3f})")

def readfrac(u):
    return float((u @ G @ u) / lam_max)

# ---- categories ----
CATS = {
 "colors": "red blue green black white yellow brown pink gray gold",
 "animals": "dog cat horse cow pig sheep bird fish lion bear wolf fox",
 "fruits": "apple banana orange lemon grape peach pear",
 "metals": "gold silver iron steel copper bronze",
 "vehicles": "car truck bus train plane boat ship bike",
 "clothing": "shirt hat coat dress shoe sock glove",
 "furniture": "chair table bed desk sofa lamp",
 "weather": "rain snow wind sun cloud storm",
 "emotions": "joy fear anger love hope grief",
 "body": "head hand foot arm leg eye nose ear",
 "jobs": "doctor teacher farmer nurse driver cook",
 "sports": "soccer tennis golf boxing hockey",
 "instruments": "guitar piano drum violin flute",
 "drinks": "water wine beer juice milk coffee tea",
 "tools": "hammer knife saw drill wrench",
 "weapons": "gun sword bomb spear dagger",
 "countries": "France China Japan Spain India Italy Egypt",
 "buildings": "house church school store tower castle",
}

def stid(word):
    t = m.to_tokens(" " + word, prepend_bos=False)[0]
    return int(t[0]) if t.shape[0] == 1 else None

cat_words = {}
for c, s in CATS.items():
    ids = {w: stid(w) for w in s.split()}
    valid = [w for w, i in ids.items() if i is not None]
    if len(valid) >= 3:
        cat_words[c] = valid
print("categories kept:", len(cat_words))

def templates(words):
    ws = words[:6]
    tmpl = []
    for i in range(len(ws)):
        a = ws[i].capitalize(); b = ws[(i + 1) % len(ws)]
        tmpl.append(f"{a}, {b} and")
    return tmpl[:4]

def resid_last(prompts):
    out = []
    for p in prompts:
        _, cache = m.run_with_cache(m.to_tokens(p), names_filter=HOOK)
        out.append(cache[HOOK][0, -1].float())
    return torch.stack(out)

# probe (diff-in-means) directions
per = {c: resid_last(templates(ws)) for c, ws in cat_words.items()}
grand = torch.cat(list(per.values()), 0).mean(0)
probe = {c: (per[c].mean(0) - grand) for c in cat_words}
probe_u = {c: (probe[c] / probe[c].norm()).numpy() for c in cat_words}

# matched SAE atom (max cosine)
atom_u = {}; match_cos = {}
for c in cat_words:
    cs = dec_unit @ probe_u[c]
    j = int(np.argmax(cs)); atom_u[c] = dec_unit[j].copy(); match_cos[c] = float(cs[j])

# steering usefulness
NEUTRAL = ["The", "It was", "I think that", "We saw a", "There is a", "Yesterday I found a"]
n_scale = resid_last(NEUTRAL).norm(dim=-1).mean().item()
print(f"n_scale (mean resid norm L{L}) {n_scale:.2f}")
neutral_tok = [m.to_tokens(p) for p in NEUTRAL]
clean = [m(t)[0, -1].float().numpy() for t in neutral_tok]

def usefulness(unit_dir, answer_ids):
    v = torch.tensor(unit_dir, dtype=torch.float32) * (STEER_C * n_scale)
    def fn(r, hook, v=v):
        r[:, -1, :] = r[:, -1, :] + v.to(r.dtype); return r
    lifts = []
    for pi, t in enumerate(neutral_tok):
        out = m.run_with_hooks(t, fwd_hooks=[(HOOK, fn)])[0, -1].float().numpy()
        lifts.append((out[answer_ids] - clean[pi][answer_ids]).mean())
    return float(np.mean(lifts))

rand_u = rng.standard_normal(d_in); rand_u /= np.linalg.norm(rand_u)

cats = list(cat_words)
R = np.array([readfrac(probe_u[c]) for c in cats])
u_probe = np.array([usefulness(probe_u[c], [stid(w) for w in cat_words[c]]) for c in cats])
u_atom  = np.array([usefulness(atom_u[c],  [stid(w) for w in cat_words[c]]) for c in cats])
u_rand  = np.array([usefulness(rand_u,     [stid(w) for w in cat_words[c]]) for c in cats])
gap = u_atom - u_probe
mcos = np.array([match_cos[c] for c in cats])

# ---- ORACLE OA ----
oracle_probe = float(u_probe.mean()); oracle_rand = float(u_rand.mean())
print(f"OA oracle_probe {oracle_probe:.3f} oracle_rand {oracle_rand:.3f} "
      f"ratio {oracle_probe/(abs(oracle_rand)+1e-9):.2f}")

# ---- PRIMARY ----
pr, pp = pearsonr(gap, R)
pa = pearsonr(u_atom, R); ppb = pearsonr(u_probe, R)
med = np.median(R); below = R < med
ratio = np.where(u_probe > 1e-6, u_atom / u_probe, np.nan)
sharp = np.nanmean((ratio[below] < 0.10).astype(float))

print("\n== per-category ==")
print("cat            R      u_probe  u_atom   gap     matchcos")
for i, c in enumerate(cats):
    print(f"{c:13s} {R[i]:.3f}  {u_probe[i]:7.3f} {u_atom[i]:7.3f} {gap[i]:7.3f} {mcos[i]:.3f}")

print(f"\nn={len(cats)}")
print(f"PRIMARY Pearson(gap,R) = {pr:.3f} (p={pp:.3f})")
print(f"  Pearson(u_atom,R)  = {pa[0]:.3f} (p={pa[1]:.3f})")
print(f"  Pearson(u_probe,R) = {ppb[0]:.3f} (p={ppb[1]:.3f})")
print(f"  Spearman(gap,R)    = {spearmanr(gap,R).correlation:.3f}")
print(f"  sharp-form pass (below-median-R with atom/probe<0.10): {sharp:.2f}")
print(f"  mean matchcos {mcos.mean():.3f}  mean u_probe {u_probe.mean():.3f} mean u_atom {u_atom.mean():.3f}")

# ---- SUPPLEMENT: guard against restriction-of-range false-REFUTED ----
R_atom = np.array([readfrac(atom_u[c]) for c in cats])
# random-direction baseline read-fraction
R_randbase = float(np.mean(rho_rand**2) / lam_max)
print(f"\n[supp] R(probe) range {R.min():.3f}-{R.max():.3f} | R(atom) range "
      f"{R_atom.min():.3f}-{R_atom.max():.3f} | random-dir R baseline {R_randbase:.3f}")
# pooled 36 directions: each dir's usefulness on its own category, vs its R
Rpool = np.concatenate([R, R_atom]); Upool = np.concatenate([u_probe, u_atom])
prp = pearsonr(Upool, Rpool)
print(f"[supp] pooled Pearson(usefulness, R) over {len(Rpool)} dirs = {prp[0]:.3f} (p={prp[1]:.3f}); "
      f"pooled R range {Rpool.min():.3f}-{Rpool.max():.3f}")

# ---- VERDICT ----
broken = (OB_ratio < 5) or (oracle_probe <= 0) or (oracle_probe < 3 * abs(oracle_rand))
if broken:
    V_ = "BROKEN_MEASUREMENT"
elif pr >= 0.40 and pp < 0.05:
    V_ = "SUPPORTED"
elif pr < 0.20:
    V_ = "REFUTED"
else:
    V_ = "INCONCLUSIVE"
print("\nVERDICT", V_)
