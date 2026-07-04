#!/usr/bin/env python3
"""Causal-effect conservation under SAE feature splitting. See PREREG.md.
venv: ~/.local/state/mst/crc-venv311. Usage: run.py <scratch_dir_with_acts_L6.npy>"""
import os, sys, time, json
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(os.cpu_count() or 4)

SCR = sys.argv[1]
LAYER = 6
HERE = os.path.dirname(__file__)

# ---------- data ----------
A = np.load(os.path.join(SCR, "acts_L6.npy")).astype(np.float32)
mu = A.mean(0, keepdims=True); Xc = torch.tensor(A - mu)
N, d = Xc.shape
scale = float((Xc**2).sum(1).mean()**0.5) / d**0.5; X = Xc / scale
print(f"data {X.shape} scale={scale:.2f}", flush=True)

# ---------- SAE trainer (same as width_persistent_latent) ----------
def train_sae(m, l1=1.0, epochs=25, bs=2048, lr=1e-3, seed=0):
    g = torch.Generator().manual_seed(seed)
    We = torch.randn(m, d, generator=g) * (1/d**0.5); We.requires_grad_(True)
    Wd = We.detach().clone().t().contiguous(); Wd = Wd / Wd.norm(dim=0, keepdim=True); Wd.requires_grad_(True)
    be = torch.zeros(m, requires_grad=True); bd = X.mean(0).clone().requires_grad_(True)
    opt = torch.optim.Adam([We, Wd, be, bd], lr=lr)
    idx = torch.arange(N)
    with torch.enable_grad():
        for ep in range(epochs):
            perm = idx[torch.randperm(N, generator=g)]
            for i in range(0, N, bs):
                xb = X[perm[i:i+bs]]
                z = torch.relu((xb - bd) @ We.t() + be)
                xr = z @ Wd.t() + bd
                loss = ((xr - xb)**2).sum(1).mean() + l1 * z.abs().sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    Wd /= Wd.norm(dim=0, keepdim=True).clamp_min(1e-8)
    with torch.no_grad():
        z = torch.relu((X - bd) @ We.t() + be)
        l0 = (z > 0).float().sum(1).mean().item()
        var = ((X - (z @ Wd.t() + bd))**2).sum(1).mean().item() / (X**2).sum(1).mean().item()
        freq = (z > 0).float().mean(0)                       # activation frequency
        actval = (z.sum(0) / (z > 0).float().sum(0).clamp_min(1))  # mean active value
        D = (Wd / Wd.norm(dim=0, keepdim=True)).t().contiguous()   # [m,d] unit dirs (normed model space)
    return D, l0, 1 - var, freq, actval

t0 = time.time()
Dp, l0p, fvep, freqp, avp = train_sae(128, seed=0)
Dc, l0c, fvec, freqc, avc = train_sae(256, seed=0)
print(f"parent W=128 L0={l0p:.1f} FVE={fvep:.3f} | daughters 2W=256 L0={l0c:.1f} FVE={fvec:.3f} t={time.time()-t0:.1f}s", flush=True)

# directions live in centered+scaled space; a delta transforms isotropically, so the
# raw-space UNIT direction == the X-space unit direction (Dp rows are already unit).
# Steer every latent at the SAME magnitude alpha (set below) so parents/daughters/random
# control are all comparable. (v1 bug: used *scale ~= 7 while random ctrl used alpha ~= 45.)
Dp_u = Dp / Dp.norm(dim=1, keepdim=True)   # unit dirs
Dc_u = Dc / Dc.norm(dim=1, keepdim=True)

# ---------- split assignment: each daughter -> nearest parent ----------
Cs = (Dc / Dc.norm(dim=1, keepdim=True)) @ (Dp / Dp.norm(dim=1, keepdim=True)).t()  # [256,128] cosine
assign = Cs.argmax(1).numpy()           # daughter -> parent idx
from collections import defaultdict
daughters = defaultdict(list)
for j, p in enumerate(assign):
    daughters[int(p)].append(j)
splits = {p: js for p, js in daughters.items() if len(js) >= 2}
print(f"#split parents (>=2 daughters) = {len(splits)}; daughter-count hist: "
      f"{np.bincount([len(v) for v in daughters.values()]).tolist()}", flush=True)

# ---------- model + probes ----------
from transformer_lens import HookedTransformer
M = HookedTransformer.from_pretrained("gpt2", device="cpu"); M.eval()
mean_resid_norm = float(torch.tensor(A).norm(dim=1).mean())
alpha = 0.5 * mean_resid_norm
Dp_raw = Dp_u * alpha       # [128,768] steering vectors, norm=alpha
Dc_raw = Dc_u * alpha
print(f"mean_resid_norm={mean_resid_norm:.2f} alpha={alpha:.2f} steer_norm={float(Dp_raw[0].norm()):.2f}", flush=True)

PROBES = [
    "The weather today is quite", "She opened the door and saw",
    "In the beginning there was", "He picked up the phone to",
    "The scientist looked at the", "They walked into the room and",
    "My favorite thing about summer is", "The old man sat down by the",
    "After the meeting ended everyone", "The book on the table was about",
    "We drove all night to reach the", "The children played in the",
]
L = 6
tok = [M.to_tokens(p)[0][:L] for p in PROBES]
tok = [t for t in tok if t.shape[0] == L]
T = torch.stack(tok)                    # [P, L]
P = T.shape[0]
HN = f"blocks.{LAYER}.hook_resid_post"
clean_logits = M(T)[:, -1, :].float()   # [P, vocab]
clean_lp = torch.log_softmax(clean_logits, -1)
print(f"probes P={P} L={L} vocab={clean_logits.shape[1]}", flush=True)

def signature(v):
    """steer add v at last pos; return (delta_logits[P,vocab], mean KL nats)."""
    def fn(r, hook, v=v):
        r[:, -1, :] = r[:, -1, :] + v.to(r.dtype); return r
    steered = M.run_with_hooks(T, fwd_hooks=[(HN, fn)])[:, -1, :].float()
    dl = (steered - clean_logits)
    slp = torch.log_softmax(steered, -1); sp = slp.exp()
    kl = (sp * (slp - clean_lp)).sum(-1).mean().item()   # KL(steered||clean)
    return dl, kl

# ---------- compute signatures for all latents involved ----------
need_p = sorted(splits.keys())
need_c = sorted({j for js in splits.values() for j in js})
t1 = time.time()
sig_p, kl_p = {}, {}
for p in need_p:
    dl, kl = signature(Dp_raw[p]); sig_p[p] = dl.mean(0); kl_p[p] = kl   # mean over probes -> [vocab]
    # split-half for oracle (b)
sig_c = {}
for j in need_c:
    dl, kl = signature(Dc_raw[j]); sig_c[j] = dl.mean(0)
print(f"signatures: {len(need_p)} parents + {len(need_c)} daughters in {time.time()-t1:.1f}s", flush=True)

# ---------- oracle (a): steering KL, real vs random ----------
med_kl = float(np.median(list(kl_p.values())))
rng = np.random.default_rng(0)
rand_kls = []
for _ in range(20):
    rv = torch.tensor(rng.standard_normal(d), dtype=torch.float32); rv = rv / rv.norm() * alpha
    _, kl = signature(rv); rand_kls.append(kl)
med_rand_kl = float(np.median(rand_kls))
print(f"ORACLE(a) median parent-steer KL={med_kl:.4f} nats | random-dir KL={med_rand_kl:.4f}", flush=True)

# ---------- oracle (b): split-half signature reproducibility ----------
half1, half2 = list(range(0, P, 2)), list(range(1, P, 2))
def sig_half(v, rows):
    def fn(r, hook, v=v):
        r[:, -1, :] = r[:, -1, :] + v.to(r.dtype); return r
    steered = M.run_with_hooks(T[rows], fwd_hooks=[(HN, fn)])[:, -1, :].float()
    return (steered - clean_logits[rows]).mean(0)
def cos(a, b):
    a = a / (a.norm() + 1e-9); b = b / (b.norm() + 1e-9); return float((a * b).sum())
shcos = []
for p in need_p[:40]:
    shcos.append(cos(sig_half(Dp_raw[p], half1), sig_half(Dp_raw[p], half2)))
med_shcos = float(np.median(shcos))
print(f"ORACLE(b) median split-half signature cosine={med_shcos:.3f}", flush=True)

# ---------- primary: conservation ----------
parents = need_p
cos_true, cos_rand, geo_cos, share_cv = [], [], [], []
for p in parents:
    js = splits[p]
    recon = torch.stack([sig_c[j] for j in js]).sum(0)      # vector sum of daughter signatures
    cos_true.append(cos(recon, sig_p[p]))
    # specificity control: random other parents
    others = [q for q in parents if q != p]
    picks = rng.choice(len(others), size=min(5, len(others)), replace=False)
    cos_rand.append(float(np.mean([cos(recon, sig_p[others[k]]) for k in picks])))
    # geometric-only conservation
    geo = (Dc[js]).sum(0)
    geo_cos.append(cos(geo, Dp[p]))
    # reconstruction-share CV across daughters (freq*meanactval)
    shares = np.array([float(freqc[j] * avc[j]) for j in js])
    share_cv.append(float(shares.std() / (shares.mean() + 1e-12)))

cos_true = np.array(cos_true); cos_rand = np.array(cos_rand)
med_true = float(np.median(cos_true)); med_rand = float(np.median(cos_rand))
Delta = med_true - med_rand
med_geo = float(np.median(geo_cos)); med_cv = float(np.median(share_cv))
print(f"\n#splits={len(parents)} median cos_true={med_true:.3f} median cos_rand={med_rand:.3f} "
      f"Delta={Delta:.3f}", flush=True)
print(f"[premise] median reconstruction-share CV across daughters={med_cv:.2f} (high => shares vary wildly)")
print(f"[geom ctrl] median cos(sum d_daughters, d_parent)={med_geo:.3f}", flush=True)

# ---------- verdict (frozen rule) ----------
if med_kl < 0.02 or med_shcos < 0.30:
    verdict = "BROKEN_MEASUREMENT"
elif len(parents) < 8:
    verdict = "INCONCLUSIVE (too few splits)"
elif med_true >= 0.50 and Delta >= 0.15:
    verdict = "SUPPORTED"
elif Delta <= 0.05:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"
print(f"\nVERDICT (frozen rule): {verdict}", flush=True)

out = dict(model="gpt2", layer=LAYER, W=128, twoW=256, alpha=alpha,
           l0_parent=l0p, fve_parent=fvep, l0_daughter=l0c, fve_daughter=fvec,
           n_splits=len(parents), oracle_median_kl=med_kl, oracle_random_kl=med_rand_kl,
           oracle_splithalf_cos=med_shcos, median_cos_true=med_true, median_cos_rand=med_rand,
           delta=Delta, median_geom_cos=med_geo, median_share_cv=med_cv,
           cos_true_list=cos_true.tolist(), cos_rand_list=cos_rand.tolist(), verdict=verdict)
json.dump(out, open(os.path.join(HERE, "result.json"), "w"), indent=2)
print("saved result.json", flush=True)
