#!/usr/bin/env python3
"""repair_rhat — weight-native repair predictor vs measured self-repair.

Preregistered in PREREG.md (same dir). Model: pythia-160m, CPU, layers 6-11,
zero-ablation of one head's z at the LAST position, 16 prompts.
venv: ~/.local/state/mst/crc-venv311/bin/python
"""
import numpy as np, torch, json, os, sys
torch.set_grad_enabled(False)
torch.set_num_threads(max(1, os.cpu_count() or 4))

from transformer_lens import HookedTransformer

MODEL = "pythia-160m"
LAYERS = list(range(6, 12))
DE_MIN = 0.05

PROMPTS = [
    "The Eiffel Tower is located in the city of",
    "When the sun rises in the morning, the sky turns",
    "She opened the fridge and took out a carton of",
    "The capital of France is",
    "After the long hike, they were extremely",
    "He plugged the charger into his",
    "The chef added salt and pepper to the",
    "On Monday morning she drove her car to",
    "The dog wagged its tail because it was",
    "Two plus two equals",
    "The scientist looked through the microscope at the",
    "In winter, the lake freezes and children skate on the",
    "The orchestra was led by a famous",
    "He signed the letter with his full",
    "The library was quiet except for the sound of",
    "They boarded the plane and fastened their seat",
]

m = HookedTransformer.from_pretrained(MODEL, device="cpu", fold_ln=False,
                                      center_writing_weights=False, center_unembed=False,
                                      dtype=torch.float64); m.eval()
d_model = m.cfg.d_model; n_heads = m.cfg.n_heads; n_layers = m.cfg.n_layers
gamma = m.ln_final.w.float(); beta = m.ln_final.b.float()
W_U = m.W_U.float(); b_U = m.b_U.float()
eps = m.cfg.eps

def ln(r):  # r: [d]
    mu = r.mean(); s = ((r - mu).pow(2).mean() + eps).sqrt()
    return (r - mu) / s * gamma + beta, s

# ---------- clean runs ----------
names = [f"blocks.{l}.attn.hook_z" for l in LAYERS] + ["ln_final.hook_normalized"]
caches, toks, resid_fin, clean_logit, tok_ids, sigmas = [], [], [], [], [], []
oracle1 = 0.0  # analytic-LN reconstruction error
for p in PROMPTS:
    t = m.to_tokens(p)
    logits, c = m.run_with_cache(t, names_filter=lambda n: n.startswith("blocks.") and n.endswith("hook_z") or n in ("ln_final.hook_scale", "ln_final.hook_normalized") or n.endswith("hook_resid_post"))
    r = c[f"blocks.{n_layers-1}.hook_resid_post"][0, -1].float()
    ln_r, s = ln(r)
    recon = ln_r @ W_U + b_U
    oracle1 = max(oracle1, (recon - logits[0, -1].float()).abs().max().item())
    tid = int(logits[0, -1].argmax())
    caches.append(c); toks.append(t); resid_fin.append(r)
    clean_logit.append(logits[0, -1, tid].float().item()); tok_ids.append(tid); sigmas.append(s)

print(f"[oracle1] analytic-LN max recon error: {oracle1:.2e}")

# ---------- per-head measurement ----------
HEADS = [(l, h) for l in LAYERS for h in range(n_heads)]
DE = np.zeros((len(HEADS), len(PROMPTS)))
DEr = np.zeros_like(DE)   # recomputed-LN direct effect
TE = np.zeros_like(DE)

for hi, (l, h) in enumerate(HEADS):
    W_O_h = m.W_O[l, h].float()  # [d_head, d_model]
    for pi in range(len(PROMPTS)):
        z = caches[pi][f"blocks.{l}.attn.hook_z"][0, -1, h].float()  # [d_head]
        o = z @ W_O_h                                                # [d_model]
        tid = tok_ids[pi]
        # frozen-LN direct effect
        DE[hi, pi] = (((o - o.mean()) / sigmas[pi]) * gamma) @ W_U[:, tid]
        # recomputed-LN direct effect
        ln_mod, _ = ln(resid_fin[pi] - o)
        DEr[hi, pi] = clean_logit[pi] - float(ln_mod @ W_U[:, tid] + b_U[tid])
        # total effect: zero-ablate z at last pos, rerun downstream
        def zero(v, hook, h=h):
            v[:, -1, h, :] = 0.0; return v
        out = m.run_with_hooks(toks[pi], fwd_hooks=[(f"blocks.{l}.attn.hook_z", zero)])
        TE[hi, pi] = clean_logit[pi] - float(out[0, -1, tid])

mDE, mTE, mDEr = DE.mean(1), TE.mean(1), DEr.mean(1)
SR = mDE - mTE
LN_rep = mDE - mDEr

# oracle 2: layer-11 heads — TE must equal DEr exactly (no downstream)
i11 = [i for i, (l, _) in enumerate(HEADS) if l == n_layers - 1]
oracle2 = float(np.abs(TE[i11] - DEr[i11]).max())
# oracle 3: effect size
oracle3 = float(mDE.max())
print(f"[oracle2] layer-11 |TE-DEr| max: {oracle2:.2e}")
print(f"[oracle3] max mean DE: {oracle3:.3f}")

# ---------- weight-native R-hat (weights only) ----------
readers = {}  # per layer: list of (W, ||W||_F)
for l in range(n_layers):
    rs = [m.W_V[l, hh].float().T for hh in range(n_heads)]  # [d_model, d_head] each -> W_V is [n_heads,d_model,d_head]
    rs = [m.W_V[l, hh].float() for hh in range(n_heads)]     # [d_model, d_head]
    rs.append(m.W_in[l].float())                             # [d_model, d_mlp]
    readers[l] = [(W, float(W.norm())) for W in rs]

sqrt_d = d_model ** 0.5
R_LN = np.zeros(len(HEADS)); R_down = np.zeros(len(HEADS))
for hi, (l, h) in enumerate(HEADS):
    Vh = m.W_V[l, h].float(); Oh_ = m.W_O[l, h].float()      # [d,64],[64,d]
    nO = float(torch.linalg.matrix_norm(Vh @ Oh_))
    R_LN[hi] = nO
    ovs = []
    for l2 in range(l + 1, n_layers):
        for W, nW in readers[l2]:
            num = float(torch.linalg.matrix_norm(Vh @ (Oh_ @ W)))
            ovs.append(num * sqrt_d / (nO * nW + 1e-12))
    R_down[hi] = float(np.mean(ovs)) if ovs else 0.0

def z(x): return (x - x.mean()) / (x.std() + 1e-12)
R_hat = z(R_LN) + z(R_down)

# ---------- decision rule (frozen in PREREG.md) ----------
def pearson(a, b): return float(np.corrcoef(a, b)[0, 1])
def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    return pearson(ra, rb)

elig = mDE > DE_MIN
RF = 1.0 - mTE[elig] / mDE[elig]
r_p = pearson(R_hat[elig], RF); r_s = spearman(R_hat[elig], RF)

# falsification clause: top-quartile SR heads vs R_down/R_LN percentiles
q75 = np.quantile(SR[elig], 0.75)
top = elig & (SR >= q75)
falsif = (np.median(R_down[top]) < np.quantile(R_down, 0.25)) and \
         (np.median(R_LN[top]) < np.quantile(R_LN, 0.25))

# secondary: LN share
pos = elig & (SR > 0)
ln_share = float(LN_rep[pos].sum() / SR[pos].sum()) if pos.any() else float("nan")
r_ln = pearson(R_LN[elig], LN_rep[elig])

broken = (oracle1 > 1e-3) or (oracle2 > 0.01) or (oracle3 < 0.5)
if broken:
    verdict = "BROKEN_MEASUREMENT"
elif falsif or r_p < 0.2:
    verdict = "REFUTED"
elif r_p > 0.5 and r_s > 0.4:
    verdict = "SUPPORTED"
else:
    verdict = "INCONCLUSIVE"

res = dict(model=MODEL, n_heads_tested=len(HEADS), n_eligible=int(elig.sum()),
           oracle_ln_recon=oracle1, oracle_l11_identity=oracle2, oracle_max_DE=oracle3,
           pearson_Rhat_RF=r_p, spearman_Rhat_RF=r_s,
           pearson_RLN_only=pearson(R_LN[elig], RF), pearson_Rdown_only=pearson(R_down[elig], RF),
           falsification_clause_fired=bool(falsif),
           agg_LN_share=ln_share, pearson_RLN_vs_LNrepair=r_ln,
           mean_SR_eligible=float(SR[elig].mean()), mean_RF=float(RF.mean()),
           verdict=verdict)
print(json.dumps(res, indent=2))
out_dir = os.path.dirname(os.path.abspath(__file__))
json.dump(res, open(os.path.join(out_dir, "results.json"), "w"), indent=2)

# small per-head dump for inspection
np.savez(os.path.join(out_dir, "perhead.npz"), heads=np.array(HEADS), mDE=mDE, mTE=mTE,
         SR=SR, LN_rep=LN_rep, R_LN=R_LN, R_down=R_down, R_hat=R_hat, elig=elig)
