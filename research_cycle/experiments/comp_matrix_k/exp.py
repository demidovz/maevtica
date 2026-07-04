#!/usr/bin/env python3
"""Compensation Matrix K vs self-repair. See PREREG.md (commit 2d6f535) — run AFTER prereg."""
import os, json, time
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(max(1, os.cpu_count() or 4))

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ, NP_, SMOKE = 32, 24, os.environ.get("SMOKE") == "1"

from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained("pythia-160m", device="cpu"); model.eval()
nL, nH = model.cfg.n_layers, model.cfg.n_heads
W_O = model.W_O  # (nL, nH, d_head, d_model)
W_U, b_U = model.W_U, model.b_U

# ---- prompts: wikitext-103 test, per PREREG ----
import pyarrow.parquet as pq
PARQ = os.path.expanduser("~/.cache/huggingface/hub/datasets--Salesforce--wikitext/"
    "snapshots/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1/test-00000-of-00001.parquet")
texts = pq.read_table(PARQ, columns=["text"]).column("text").to_pylist()
toks_list, tgts = [], []
for t in texts:
    if len(t) < 400: continue
    ids = model.to_tokens(t)[0]  # includes BOS
    if ids.shape[0] < 34: continue
    toks_list.append(ids[:33]); tgts.append(int(ids[33]))
    if len(toks_list) == NP_: break
assert len(toks_list) == NP_, f"only {len(toks_list)} prompts"
toks = torch.stack(toks_list)          # (NP, 33): feed first 32... wait, ids[:33] is 33 tokens incl BOS
# feed all 33? PREREG: input = first 32 tokens, target = token 33. With BOS prepended,
# use first 32 positions as input and ids[32] as target.
tgts = toks[:, 32].tolist()
toks = toks[:, :32].contiguous()       # (NP, 32)
tgt = torch.tensor(tgts)               # (NP,)
U_t = W_U[:, tgt].T.contiguous()       # (NP, d_model) per-prompt unembed column

FILT = lambda n: n.endswith("attn.hook_z") or n.endswith("hook_mlp_out") \
    or n == f"blocks.{nL-1}.hook_resid_post" or n == "ln_final.hook_scale"

def contrib(v, s):
    """frozen-LN contribution of write v (NP, d_model) with clean scale s (NP,1)."""
    return (((v - v.mean(-1, keepdim=True)) / s) * U_t).sum(-1)  # (NP,)

def per_component_contribs(cache, s):
    """returns heads (nL, nH, NP) and mlps (nL, NP) frozen-LN contributions at final pos."""
    hc = torch.empty(nL, nH, NP_); mc = torch.empty(nL, NP_)
    for l in range(nL):
        z = cache[f"blocks.{l}.attn.hook_z"][:, -1]          # (NP, nH, d_head)
        w = torch.einsum("bhd,hdm->bhm", z, W_O[l])          # (NP, nH, d_model)
        hc[l] = torch.stack([contrib(w[:, h], s) for h in range(nH)])
        mc[l] = contrib(cache[f"blocks.{l}.hook_mlp_out"][:, -1], s)
    return hc, mc

def frozen_logit(cache, s):
    r = cache[f"blocks.{nL-1}.hook_resid_post"][:, -1]
    return (((r - r.mean(-1, keepdim=True)) / s) * U_t).sum(-1) + b_U[tgt]

# ---- clean run ----
t0 = time.time()
logits, ccache = model.run_with_cache(toks, names_filter=FILT)
L_clean = logits[torch.arange(NP_), -1, tgt].float()                    # (NP,)
logprob_clean = torch.log_softmax(logits[:, -1].float(), -1)[torch.arange(NP_), tgt]
s_clean = ccache["ln_final.hook_scale"][:, -1].float()                  # (NP, 1)
hc_clean, mc_clean = per_component_contribs(ccache, s_clean)
Lf_clean = frozen_logit(ccache, s_clean)
oracle2 = (Lf_clean - L_clean).abs().median().item()
mean_z = {l: ccache[f"blocks.{l}.attn.hook_z"][:, -1].mean(0) for l in range(nL)}  # (nH, d_head)
print(f"clean done {time.time()-t0:.1f}s  mean logprob {logprob_clean.mean():.3f}  frozen-check {oracle2:.2e}", flush=True)

# ---- ablated runs ----
heads = [(l, h) for l in range(nL - 1) for h in range(nH)]
if SMOKE: heads = [(4, 3)]
DE = np.zeros((len(heads), NP_)); TE = np.zeros_like(DE)
ROWNORM = np.zeros_like(DE); KSUM = np.zeros_like(DE); DLN = np.zeros_like(DE)
CLOSURE = np.zeros_like(DE)
for i, (l, h) in enumerate(heads):
    def abl(z, hook):
        z[:, -1, h] = mean_z[l][h]; return z
    with model.hooks(fwd_hooks=[(f"blocks.{l}.attn.hook_z", abl)]):
        alogits, acache = model.run_with_cache(toks, names_filter=FILT)
    L_abl = alogits[torch.arange(NP_), -1, tgt].float()
    hc_a, mc_a = per_component_contribs(acache, s_clean)
    te = (L_clean - L_abl)
    de = hc_clean[l, h] - hc_a[l, h]
    dK_h = (hc_a[l+1:] - hc_clean[l+1:])            # (layers>l, nH, NP)
    dK_m = (mc_a[l+1:] - mc_clean[l+1:])            # (layers>l, NP)
    ksum = dK_h.sum((0, 1)) + dK_m.sum(0)
    rown = dK_h.abs().sum((0, 1)) + dK_m.abs().sum(0)
    dln = frozen_logit(acache, s_clean) - L_abl
    TE[i], DE[i] = te.numpy(), de.numpy()
    KSUM[i], ROWNORM[i], DLN[i] = ksum.numpy(), rown.numpy(), dln.numpy()
    CLOSURE[i] = (de - ksum + dln - te).abs().numpy()
    if i % 24 == 0 or SMOKE:
        print(f"[{i+1}/{len(heads)}] L{l}H{h} TE={te.mean():+.4f} DE={de.mean():+.4f} "
              f"rep={(de-te).mean():+.4f} rown={rown.mean():.4f} dln={dln.mean():+.4f} "
              f"closure_max={CLOSURE[i].max():.2e}  {time.time()-t0:.0f}s", flush=True)

# ---- aggregates & preregistered rules ----
repair = (DE - TE).mean(1); rown = ROWNORM.mean(1)
sem = (DE - TE).std(1, ddof=1) / np.sqrt(NP_)
X = np.stack([rown, np.ones_like(rown)], 1)
beta, *_ = np.linalg.lstsq(X, repair, rcond=None)
pred = X @ beta
R2 = 1 - ((repair - pred) ** 2).sum() / ((repair - repair.mean()) ** 2).sum()
r_pearson = float(np.corrcoef(rown, repair)[0, 1])
# secondary: |repair|
Xa = X; ba, *_ = np.linalg.lstsq(Xa, np.abs(repair), rcond=None)
R2_abs = 1 - ((np.abs(repair) - Xa @ ba) ** 2).sum() / ((np.abs(repair) - np.abs(repair).mean()) ** 2).sum()
# zero-rownorm clause
k = max(1, len(heads) // 10)
idx = np.argsort(rown)[:k]
clause_frac = float(np.mean(np.abs(repair[idx]) <= 2 * sem[idx]))
# LN share
ln_share = float(np.mean(np.abs(DLN.mean(1))) / max(np.mean(np.abs(repair)), 1e-9))
# oracles
o1_med, o1_p99 = float(np.median(CLOSURE)), float(np.percentile(CLOSURE, 99))
o3 = float(np.max(np.abs(TE.mean(1))))
strong = np.sum((np.abs(repair) > 0.02) & (np.abs(repair) > 2 * sem))
o5 = float(logprob_clean.mean())
oracle_ok = (o1_med < 1e-3 and o1_p99 < 0.02 and oracle2 < 1e-3 and o3 >= 0.05 and o5 > -7)
phenomenon = int(strong) >= 10
if not oracle_ok: verdict = "BROKEN_MEASUREMENT"
elif not phenomenon: verdict = "NO_PHENOMENON->inconclusive"
elif R2 >= 0.7 and clause_frac >= 0.75: verdict = "SUPPORTED"
elif R2 < 0.3 or (R2 < 0.7 and clause_frac < 0.75): verdict = "REFUTED"
else: verdict = "INCONCLUSIVE"

res = dict(model="pythia-160m", n_prompts=NP_, n_heads=len(heads),
           clean_mean_logprob=o5, oracle_closure_median=o1_med, oracle_closure_p99=o1_p99,
           oracle_frozen_clean=oracle2, oracle_max_abs_TE=o3, n_strong_repair_heads=int(strong),
           oracle_ok=bool(oracle_ok), phenomenon=bool(phenomenon),
           R2_primary=float(R2), pearson_r=r_pearson, slope=float(beta[0]),
           R2_abs_secondary=float(R2_abs), zero_rownorm_clause_frac=clause_frac,
           ln_share_of_repair=ln_share,
           repair_stats=dict(mean=float(repair.mean()), std=float(repair.std()),
                             min=float(repair.min()), max=float(repair.max())),
           verdict=verdict)
print(json.dumps(res, indent=1))
if not SMOKE:
    json.dump(res, open(os.path.join(HERE, "result.json"), "w"), indent=1)
    np.savez(os.path.join(HERE, "arrays.npz"), DE=DE, TE=TE, ROWNORM=ROWNORM,
             KSUM=KSUM, DLN=DLN, CLOSURE=CLOSURE,
             heads=np.array([f"{l}.{h}" for l, h in heads]))
