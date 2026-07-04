import os, numpy as np, torch
torch.set_num_threads(max(1, os.cpu_count() or 4))
rng = np.random.default_rng(0)

from transformer_lens import HookedTransformer
m = HookedTransformer.from_pretrained("gpt2", device="cpu"); m.eval()
d_in = m.cfg.d_model
L = 8
K = 4
NV = 80
HOOK = f"blocks.{L}.hook_resid_pre"

CATS = {
 "colors": "red blue green black white yellow brown pink gray gold",
 "animals": "dog cat horse cow pig sheep bird fish lion bear wolf fox",
 "fruits": "apple banana orange lemon grape peach pear",
 "countries": "France China Japan Spain India Italy Egypt",
 "drinks": "water wine beer juice milk coffee tea",
 "body": "head hand foot arm leg eye nose ear",
}
BASE = ["The", "It was", "I think that", "We saw a", "There is a", "Yesterday I found a"]

def stid(w):
    t = m.to_tokens(" " + w, prepend_bos=False)[0]
    return int(t[0]) if t.shape[0] == 1 else None

cat_ids = {c: np.array([i for i in (stid(w) for w in s.split()) if i is not None])
           for c, s in CATS.items()}
union_ids = np.array(sorted({i for v in cat_ids.values() for i in v}))

def src_prompts(c):
    ws = CATS[c].split()[:5]
    return [f"{ws[i].capitalize()}, {ws[(i+1)%5]} and" for i in range(4)]

def B_of_logits(lg, c):
    return lg[..., cat_ids[c]].mean(-1) - lg[..., union_ids].mean(-1)

def resid_full(prompt):
    with torch.no_grad():
        _, cache = m.run_with_cache(m.to_tokens(prompt), names_filter=HOOK)
    return cache[HOOK][0].float()          # [seq, d]

def grad_read_dir(prompt, c):
    store = {}
    def fn(r, hook):
        r.requires_grad_(True); r.retain_grad(); store["r"] = r; return r
    lg = m.run_with_hooks(m.to_tokens(prompt), fwd_hooks=[(HOOK, fn)])[0, -1]
    B = lg[torch.tensor(cat_ids[c])].mean() - lg[torch.tensor(union_ids)].mean()
    B.backward()
    g = store["r"].grad[0, -1].detach().float().numpy()
    m.zero_grad(set_to_none=True)
    return g

def unit(v):
    v = np.asarray(v, dtype=np.float64); return v / (np.linalg.norm(v) + 1e-12)

def topk_basis(rows, k):
    A = np.asarray(rows, dtype=np.float64)          # [n, d]
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return Vt[:k].T                                  # [d, k] orthonormal

def r2_ols(x, y):
    X = np.column_stack([np.ones(len(y)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    ss = (y - y.mean()) @ (y - y.mean())
    return float(1 - resid @ resid / ss) if ss > 0 else np.nan

# ---- precompute per-concept sources, grand mean ----
src_mean = {}
for c in CATS:
    src_mean[c] = torch.stack([resid_full(p)[-1] for p in src_prompts(c)])  # [4, d]
grand = torch.stack([src_mean[c].mean(0) for c in CATS]).mean(0)

base_resid = [resid_full(p) for p in BASE]                                   # each [seq, d]
with torch.no_grad():
    clean_full = {c: [float(B_of_logits(m(m.to_tokens(p))[0, -1].float(), c))
                      for p in BASE] for c in CATS}

groups = []
o0_max = 0.0
for c in CATS:
    xs = src_mean[c].mean(0)
    alpha = float(np.mean([float((xs - br[-1]).norm()) for br in base_resid]))
    # subspaces
    G = np.stack([grad_read_dir(p, c) for p in BASE])                        # [6, d]
    Rb = topk_basis(G, K)                                                    # [d, K]
    Wrows = (src_mean[c] - grand).numpy()                                    # [4, d]
    Wb = topk_basis(Wrows, K)
    # principal angle
    s = np.linalg.svd(Rb.T @ Wb, compute_uv=False)
    theta1 = float(np.degrees(np.arccos(np.clip(s.max(), -1, 1))))
    # O2 projection sanity
    inR = (Rb @ np.array([unit(rng.standard_normal(K)) for _ in range(20)]).T).T
    amb = np.array([unit(rng.standard_normal(d_in)) for _ in range(20)])
    pR_in = np.linalg.norm(inR @ Rb, axis=1).min()
    pR_amb = np.linalg.norm(amb @ Rb, axis=1).mean()
    inW = (Wb @ np.array([unit(rng.standard_normal(K)) for _ in range(20)]).T).T
    pW_in = np.linalg.norm(inW @ Wb, axis=1).min()
    pW_amb = np.linalg.norm(amb @ Wb, axis=1).mean()
    # sample v (prereg scheme)
    Vs = []
    for _ in range(NV):
        cR, cW, cA = rng.dirichlet([1.0, 1.0, 1.0])
        uR = Rb @ unit(rng.standard_normal(K))
        uW = Wb @ unit(rng.standard_normal(K))
        uA = unit(rng.standard_normal(d_in))
        Vs.append(unit(np.sqrt(cR) * uR + np.sqrt(cW) * uW + np.sqrt(cA) * uA))
    Vs = np.array(Vs)                                                        # [NV, d]
    g_mean = G.mean(0)
    r1 = Rb[:, 0]
    dm = unit((xs - grand).numpy())
    extras = np.stack([np.zeros(d_in), r1, -r1, dm])                         # clean,+r1,-r1,dm
    allv = np.concatenate([extras, Vs])                                      # [NV+4, d]
    dv = torch.tensor(allv, dtype=torch.float32) * alpha                     # scaled

    Bmat = np.zeros((len(BASE), len(allv)))
    for i, br in enumerate(base_resid):
        batch = br.unsqueeze(0).repeat(len(allv), 1, 1).clone()
        batch[:, -1, :] += dv
        with torch.no_grad():
            lg = m(batch, start_at_layer=L)[:, -1].float()
        Bmat[i] = B_of_logits(lg, c).numpy()
        o0_max = max(o0_max, abs(Bmat[i, 0] - clean_full[c][i]))
    E = (Bmat - Bmat[:, :1]).mean(0)                                         # vs start_at_layer clean
    e_p, e_n, e_dm = E[1], E[2], E[3]
    e_star = max(abs(e_p), abs(e_n))
    Ev = np.abs(E[4:])
    pR = np.linalg.norm(Vs @ Rb, axis=1)
    pW = np.linalg.norm(Vs @ Wb, axis=1)
    r2R = r2_ols(pR, Ev)
    r2W = r2_ols(pW, Ev)
    lin = np.abs(Vs @ g_mean) * alpha
    r2lin = r2_ols(lin, Ev)
    groups.append(dict(c=c, alpha=alpha, theta1=theta1, r2R=r2R, r2W=r2W, D=r2R - r2W,
                       r2lin=r2lin, e_star=e_star, e_p=e_p, e_n=e_n, e_dm=e_dm,
                       pR_in=pR_in, pR_amb=pR_amb, pW_in=pW_in, pW_amb=pW_amb))
    print(f"{c:10s} alpha {alpha:6.1f} theta1 {theta1:5.1f} r2R {r2R:.3f} r2W {r2W:.3f} "
          f"D {r2R-r2W:+.3f} r2lin {r2lin:.3f} |E(r1)| {e_star:6.2f} E(+r1) {e_p:+6.2f} "
          f"E(-r1) {e_n:+6.2f} E(dm) {e_dm:+6.2f}")

r2Rs = np.array([g["r2R"] for g in groups])
Ds = np.array([g["D"] for g in groups])
th = np.array([g["theta1"] for g in groups])
elig = th > 20.0
medR = float(np.median(r2Rs))
medD_el = float(np.median(Ds[elig])) if elig.any() else np.nan

O1 = float(np.mean([g["e_star"] for g in groups]))
O2ok = all(g["pR_in"] >= 0.99 and g["pW_in"] >= 0.99 and
           g["pR_amb"] < 0.3 and g["pW_amb"] < 0.3 for g in groups)
signflip = sum(1 for g in groups if g["e_p"] * g["e_n"] < 0)

print(f"\nO0 max|B_full - B_start_at_layer| = {o0_max:.5f}")
print(f"O1 mean |E(top reader dir)| = {O1:.3f} logits")
print(f"O2 projection sanity ok = {O2ok} "
      f"(min in-subspace {min(min(g['pR_in'], g['pW_in']) for g in groups):.4f}, "
      f"max ambient {max(max(g['pR_amb'], g['pW_amb']) for g in groups):.4f})")
print(f"O3 sign-flip E(+r1) vs E(-r1): {signflip}/6 | median r2lin {np.median([g['r2lin'] for g in groups]):.3f}")
print(f"median r2R {medR:.3f} | median r2W {np.median([g['r2W'] for g in groups]):.3f} | "
      f"eligible(theta1>20) {int(elig.sum())}/6 | median D eligible {medD_el:.3f}")
print(f"theta1 per group: {np.round(th,1).tolist()}")

broken = (o0_max >= 1e-3) or (O1 < 0.5) or (not O2ok)
if broken:
    V = "BROKEN_MEASUREMENT"
elif (medR < 0.5) or (elig.any() and medD_el <= 0.05):
    V = "REFUTED"
elif (medR >= 0.8) and elig.any() and (medD_el >= 0.2):
    V = "SUPPORTED"
else:
    V = "INCONCLUSIVE"
print("\nVERDICT", V)
