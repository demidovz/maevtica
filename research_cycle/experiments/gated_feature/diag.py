#!/usr/bin/env python3
"""Post-hoc diagnostic for run 2 (NOT part of the decision rule): reproduce the
effect matrix and count, per dead latent, contexts with effect >= 3*layer_median.
Distinguishes 'gates unlearnable' from 'no high-effect contexts exist'."""
from __future__ import annotations
import glob, json, time
import numpy as np, torch
import torch.nn.functional as Fn
torch.set_grad_enabled(False); torch.set_num_threads(12)
from transformer_lens import HookedTransformer
from safetensors.torch import load_file
import importlib.util
spec = importlib.util.spec_from_file_location("exp2", __file__.rsplit("/",1)[0] + "/exp2.py")
# don't exec exp2 (it runs everything); just copy corpus by parsing constants
L = 7; HOOK = f"blocks.{L}.hook_resid_pre"
M_TARGET = 256; MIN_FIRE = 40; MIN_FIRE_FB = 25; CHUNK = 128
OUT = __file__.rsplit("/", 1)[0]
src = open(OUT + "/exp2.py").read()
ns = {}
exec(src.split("CORPUS = ",1)[1].split("assert",1)[0].join(["CORPUS = ",""]), ns)
CORPUS = ns["CORPUS"]; assert len(CORPUS) == 120

m = HookedTransformer.from_pretrained("gpt2", device="cpu"); m.eval()
d_model = m.cfg.d_model
base = glob.glob("/home/friemann/.cache/huggingface/hub/models--jbloom--GPT2-Small-SAEs-Reformatted/snapshots/*/" + HOOK)[0]
sd = load_file(base + "/sae_weights.safetensors")
W_enc = sd["W_enc"].float(); b_enc = sd["b_enc"].float()
W_dec = sd["W_dec"].float(); b_dec = sd["b_dec"].float()

toks_list, feats = [], []
for p in CORPUS:
    t = m.to_tokens(p)
    _, c = m.run_with_cache(t, names_filter=HOOK)
    r = c[HOOK][0].float()
    f = torch.relu((r - b_dec) @ W_enc + b_enc)
    f[0] = 0.0
    toks_list.append(t); feats.append(f)
P = len(CORPUS)
fires = torch.stack([(f > 0).any(0) for f in feats])
fire_count = fires.sum(0)
qual = (fire_count >= MIN_FIRE).nonzero().flatten()
if qual.numel() < M_TARGET:
    qual = (fire_count >= MIN_FIRE_FB).nonzero().flatten()
rng = np.random.default_rng(0)
qn = qual.numpy()
sample = np.sort(rng.choice(qn, size=min(M_TARGET, len(qn)), replace=False))
M = len(sample)

clean_lp_all, clean_p_all = [], []
for t in toks_list:
    lp = Fn.log_softmax(m(t)[0].float(), -1)
    clean_lp_all.append(lp); clean_p_all.append(lp.exp())
def kl_pos(pi, abl_lp):
    return (clean_p_all[pi].unsqueeze(0) * (clean_lp_all[pi].unsqueeze(0) - abl_lp)).sum(-1)

sample_t = torch.tensor(sample)
eff = np.full((M, P), np.nan, np.float32)
t0 = time.time()
for pi, (t, f) in enumerate(zip(toks_list, feats)):
    live = (f[:, sample_t] > 0).any(0).nonzero().flatten()
    for s in range(0, live.numel(), CHUNK):
        sl = live[s:s + CHUNK]; bs = sl.numel()
        lat = sample_t[sl]
        acts = f[:, lat]
        delta = -acts.permute(1, 0).unsqueeze(-1) * W_dec[lat].unsqueeze(1)
        def fn(r, hook, d=delta): return r + d.to(r.dtype)
        lg = m.run_with_hooks(t.repeat(bs, 1), fwd_hooks=[(HOOK, fn)]).float()
        kls = kl_pos(pi, Fn.log_softmax(lg, -1))
        seq = lg.shape[1]
        first_fire = (acts > 0).float().argmax(0)
        pos_idx = torch.arange(seq).unsqueeze(0)
        mask = pos_idx >= first_fire.unsqueeze(1)
        kls = kls.masked_fill(~mask, float("-inf"))
        eff[sl.numpy(), pi] = kls.max(1).values.numpy()
    if pi % 40 == 0:
        print(f"  prompt {pi}/{P}  {time.time()-t0:.0f}s", flush=True)

U = np.nanmean(eff, 1)
layer_median = float(np.median(U))
q20, q80 = np.percentile(U, [20, 80])
dead = np.where(U <= q20)[0]
topq = np.where(U >= q80)[0]
thr = 3.0 * layer_median
rows = []
for i in dead:
    e = eff[i]; ctx = ~np.isnan(e)
    n_hi = int((e[ctx] >= thr).sum())
    rows.append(dict(latent=int(sample[i]), n_ctx=int(ctx.sum()), n_high=n_hi,
                     max_eff=round(float(np.nanmax(e)), 5), U=round(float(U[i]), 6)))
diag = dict(
    repro_layer_median=round(layer_median, 6), repro_q20=round(float(q20), 6),
    repro_oracle_max=round(float(U.max()), 5),
    repro_oracle_topq_median=round(float(np.median(U[topq])), 5),
    thr=round(thr, 6),
    n_dead=len(dead),
    dead_with_any_high_ctx=sum(1 for r in rows if r["n_high"] > 0),
    dead_with_ge3_high_ctx=sum(1 for r in rows if r["n_high"] >= 3),
    total_high_ctx_over_dead=sum(r["n_high"] for r in rows),
    per_dead=rows)
np.savez(OUT + "/eff_run2.npz", eff=eff, sample=sample, U=U)
json.dump(diag, open(OUT + "/diag.json", "w"), indent=1)
print(json.dumps({k: v for k, v in diag.items() if k != "per_dead"}, indent=1))
