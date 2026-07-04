#!/usr/bin/env python3
"""Width-Persistent Latent test. See PREREG.md. venv: ~/.local/state/mst/crc-venv311."""
import os, sys, time, json
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np, torch
torch.set_num_threads(os.cpu_count() or 4)
SCR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/wpl"
A = np.load(os.path.join(SCR, "acts_L6.npy")).astype(np.float32)   # [N,768]
mu = A.mean(0, keepdims=True); X = torch.tensor(A - mu)             # center
N, d = X.shape
scale = float((X**2).sum(1).mean()**0.5) / d**0.5; X = X / scale     # normalize mean-norm -> sqrt(d)
print(f"data {X.shape} scale={scale:.2f}", flush=True)

def train_sae(m, l1=1.0, epochs=25, bs=2048, lr=1e-3, seed=0):
    g = torch.Generator().manual_seed(seed)
    We = torch.randn(m, d, generator=g) * (1/d**0.5); We.requires_grad_(True)
    Wd = (We.detach().clone().t().contiguous()); Wd = Wd / Wd.norm(dim=0, keepdim=True); Wd.requires_grad_(True)
    be = torch.zeros(m, requires_grad=True); bd = X.mean(0).clone().requires_grad_(True)
    opt = torch.optim.Adam([We, Wd, be, bd], lr=lr)
    idx = torch.arange(N)
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
        D = (Wd / Wd.norm(dim=0, keepdim=True)).t().contiguous().numpy()  # [m,d] unit dirs
    return D, l0, 1 - var  # dirs, avg L0, fraction variance explained

def maxcos(Da, Db):  # for each row of Da, max cosine to any row of Db
    return (Da @ Db.T).max(1)

WIDTHS = [128, 192, 288, 432, 648]
t0 = time.time(); dicts = {}; stats = {}
for w in WIDTHS:
    D, l0, fve = train_sae(w, seed=0); dicts[w] = D; stats[w] = (l0, fve)
    print(f"W={w:4d} L0={l0:5.1f} FVE={fve:.3f}  t={time.time()-t0:5.1f}s", flush=True)

# ORACLE: two seeds same width 192
Da2, _, _ = train_sae(192, seed=1)
oracle = float(maxcos(dicts[192], Da2).mean())
rng = np.random.default_rng(0); R = rng.standard_normal((192, d)); R /= np.linalg.norm(R, axis=1, keepdims=True)
rand_ctrl = float(maxcos(dicts[192], R).mean())
print(f"ORACLE two-seed@192 mean maxcos = {oracle:.3f} | random-dir ctrl = {rand_ctrl:.3f}", flush=True)

# Persistence + survival, anchor = W1
anchor = dicts[WIDTHS[0]]
c2 = maxcos(anchor, dicts[WIDTHS[1]]); c3 = maxcos(anchor, dicts[WIDTHS[2]]); c4 = maxcos(anchor, dicts[WIDTHS[3]])
persistence = (c2 + c3 + c4) / 3.0
c5 = maxcos(anchor, dicts[WIDTHS[4]])
TAU = 0.5
survives = (c5 >= TAU).astype(int)
event_rate = float((c5 < TAU).mean())

def auc(score, label):
    pos = score[label == 1]; neg = score[label == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    # rank-based AUC
    allv = np.concatenate([pos, neg]); order = allv.argsort()
    ranks = np.empty_like(order, float); ranks[order] = np.arange(1, len(allv)+1)
    # average ties
    return (ranks[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))

AUC = auc(persistence, survives)
# negative control: shuffle label
rng2 = np.random.default_rng(7); AUC_shuf = np.mean([auc(persistence, rng2.permutation(survives)) for _ in range(200)])
# quartiles
q1, q3 = np.quantile(persistence, [0.25, 0.75])
botmask = persistence <= q1; topmask = persistence >= q3
bot_split = float((c5[botmask] < TAU).mean()); top_split = float((c5[topmask] < TAU).mean())
ratio = bot_split / top_split if top_split > 0 else float("inf")
print(f"\nn_anchor={len(persistence)} event_rate(split/absorb)={event_rate:.3f}")
print(f"c5 quartiles of persistence: bottomQ split={bot_split:.3f} topQ split={top_split:.3f} ratio={ratio:.2f}x")
print(f"AUC(persistence->survives)={AUC:.3f} | shuffle-ctrl AUC={AUC_shuf:.3f}")

# verdict
if oracle < 0.30:
    verdict = "BROKEN_MEASUREMENT"
elif event_rate < 0.08 or event_rate > 0.92:
    verdict = "INCONCLUSIVE (degenerate base-rate)"
elif AUC >= 0.60 and ratio >= 2.0:
    verdict = "SUPPORTED"
elif AUC <= 0.55:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"
print(f"\nVERDICT (frozen rule, tau=0.5): {verdict}", flush=True)

# --- ROBUSTNESS (labeled, does not move the frozen verdict) ---
med = float(np.median(c5)); AUC_med = auc(persistence, (c5 >= med).astype(int))
print(f"\n[robust] tau-free median-split AUC(persistence->c5>={med:.3f}) = {AUC_med:.3f} (shuffle {AUC_shuf:.3f})")
print("[robust] tau  base_rate  botQ_split  topQ_split  ratio")
sweep = {}
for t in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
    br = float((c5 < t).mean()); bs_ = float((c5[botmask] < t).mean()); ts_ = float((c5[topmask] < t).mean())
    rt = bs_/ts_ if ts_ > 0 else float("inf")
    sweep[str(t)] = dict(base_rate=br, botQ=bs_, topQ=ts_, ratio=rt)
    print(f"[robust] {t:.2f}  {br:.3f}      {bs_:.3f}       {ts_:.3f}      {rt:.2f}")
out = dict(widths=WIDTHS, stats={str(k): v for k, v in stats.items()}, oracle=oracle, rand_ctrl=rand_ctrl,
          n_anchor=int(len(persistence)), event_rate=event_rate, bot_split=bot_split, top_split=top_split,
          ratio=ratio, AUC=AUC, AUC_shuffle=float(AUC_shuf), tau=TAU, verdict=verdict,
          robust_median_auc=AUC_med, robust_tau_sweep=sweep)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "result.json"), "w"), indent=2)
print("saved result.json")
