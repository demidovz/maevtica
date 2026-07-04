#!/usr/bin/env python3
"""Gated Feature falsification. Preregistration: ./PREREG.md (rule fixed there).
venv: ~/.local/state/mst/crc-venv311/bin/python"""
from __future__ import annotations
import glob, json, time
import numpy as np, torch
import torch.nn.functional as Fn
torch.set_grad_enabled(False); torch.set_num_threads(12)
from transformer_lens import HookedTransformer
from safetensors.torch import load_file

L = 7; HOOK = f"blocks.{L}.hook_resid_pre"
M_TARGET = 256; MIN_FIRE = 40; MIN_FIRE_FB = 25
HIGH_MULT = 3.0; PREC = 0.7; MAX_LIT = 5
MIN_TRAIN_SUP = 6; MIN_TEST_SUP = 3; TRAIN_FRAC = 0.6
MIN_ORACLE_MAX = 0.01; MIN_ORACLE_TOPQ = 0.005; NOISE_MAX = 1e-6
CHUNK = 128
OUT = __file__.rsplit("/", 1)[0]

CORPUS = [
 "The weather today is quite","I went to the store to buy","She looked at him and",
 "In the news this morning a","The scientist explained that the","History teaches us that every",
 "My favorite thing about summer is","The old house at the end of","He opened the letter and read",
 "Economists worry that the market","The recipe calls for two cups of","Children love to play in the",
 "The government announced a new","After the long journey they finally","Music has the power to",
 "The mountain rose sharply above","During the meeting the manager","A cold wind blew across the",
 "The book on the table was","Technology continues to change how","The garden was full of",
 "When the sun set over the","The team celebrated their victory","Doctors recommend that patients",
 "The river flowed gently past","At the airport travelers waited","The painting showed a quiet",
 "Farmers depend on the rain to","The city streets were crowded with","Under the ocean strange creatures",
 "The teacher wrote on the board","A sudden storm forced them to","The company reported that profits",
 "Deep in the forest there lived","The engineer designed a bridge that","Every morning she drinks a cup of",
 "The soldiers marched across the","Tourists gathered to watch the","The chef prepared a plate of",
 "In ancient times people believed","The dog ran across the","The debate about climate change",
 "A letter arrived from a distant","The factory produces thousands of","On the shelf sat an old",
 "The lawyer argued that the case","Birds migrate south when the","The astronaut looked back at",
 "The market was full of fresh","A quiet melody drifted through","The king summoned his advisors",
 "The students studied hard for the","Rain fell steadily on the","The detective examined the",
 "The bakery smelled of fresh","Across the valley the lights of","The philosopher pondered the",
 "The nurse checked the patient's","A gentle breeze moved the","The captain steered the ship",
 "My dog loves to chase the","The war between the two nations","She fell in love with the",
 "The computer crashed during the","He scored the winning goal in","Grandmother baked cookies for the",
 "The stock price fell after the","They drove all night to reach","The volcano erupted without any",
 "A famous actor arrived at the","The librarian sorted the returned","Snow covered the entire",
 "The judge sentenced the man to","Bees carry pollen from flower to","The orchestra tuned their",
 "His salary barely covered the","The spaceship landed on the","Waves crashed against the",
 "The toddler refused to eat her","Firefighters rushed into the burning","An ancient scroll described the",
 "The election results surprised the","She planted tomatoes in the","The plumber fixed the leaking",
 "Lightning struck the tall","The professor graded the final","A stray cat wandered into the",
 "The bank approved their loan for","Miners dug deep into the","The ballet dancer practiced her",
 "Thieves broke into the museum and","The baby slept through the","Fishermen hauled their nets onto",
 "The senator proposed a bill to","Her perfume smelled of roses and","The glacier melted faster than",
 "Workers assembled the new","The pianist played a sad","A rainbow appeared after the",
 "The surgeon washed his hands before","Trucks delivered supplies to the","The monk lived alone on the",
 "Inflation pushed the price of","The gardener trimmed the","A wolf howled somewhere in the",
 "The tailor measured the fabric","Students protested outside the","The pilot announced a delay",
 "Grapes grow best in a","The blacksmith hammered the hot","A comet streaked across the",
 "The waiter brought a bottle of","Sailors feared the coming","The architect drew plans for a",
 "Her diary contained secrets about","The referee blew the whistle","Dust settled on the abandoned",
 "The poet wrote about the","Camels crossed the endless","The mayor cut the ribbon at",
]
assert len(CORPUS) == 120, len(CORPUS)

print("loading gpt2 ...", flush=True)
m = HookedTransformer.from_pretrained("gpt2", device="cpu"); m.eval()
d_model = m.cfg.d_model

base = glob.glob("/home/friemann/.cache/huggingface/hub/models--jbloom--GPT2-Small-SAEs-Reformatted/snapshots/*/" + HOOK)[0]
sd = load_file(base + "/sae_weights.safetensors")
W_enc = sd["W_enc"].float(); b_enc = sd["b_enc"].float()
W_dec = sd["W_dec"].float(); b_dec = sd["b_dec"].float()
d_sae = W_dec.shape[0]

# ---- cache activations + SAE features per prompt ----
print("caching activations ...", flush=True)
toks_list, feats = [], []
for p in CORPUS:
    t = m.to_tokens(p)
    _, c = m.run_with_cache(t, names_filter=HOOK)
    r = c[HOOK][0].float()                       # [seq, d]
    f = torch.relu((r - b_dec) @ W_enc + b_enc)  # [seq, d_sae]
    f[0] = 0.0                                   # exclude BOS
    toks_list.append(t); feats.append(f)
P = len(CORPUS)

fires = torch.stack([(f > 0).any(0) for f in feats])  # [P, d_sae] bool
fire_count = fires.sum(0)                             # [d_sae]
qual = (fire_count >= MIN_FIRE).nonzero().flatten()
min_fire_used = MIN_FIRE
if qual.numel() < M_TARGET:
    qual = (fire_count >= MIN_FIRE_FB).nonzero().flatten()
    min_fire_used = MIN_FIRE_FB
rng = np.random.default_rng(0)
qn = qual.numpy()
sample = np.sort(rng.choice(qn, size=min(M_TARGET, len(qn)), replace=False))
M = len(sample)
print(f"qualifying={len(qn)} (min_fire={min_fire_used})  sampled M={M}", flush=True)

# ---- clean logits + noise floor ----
clean_logp = []
for t in toks_list:
    clean_logp.append(Fn.log_softmax(m(t)[0, -1].float(), -1))
clean_logp = torch.stack(clean_logp)             # [P, vocab]
def kl(clean_lp, abl_lp):                        # KL(clean || ablated)
    return (clean_lp.exp() * (clean_lp - abl_lp)).sum(-1)
zdelta = torch.zeros(1, toks_list[0].shape[1], d_model)
def zfn(r, hook): return r + zdelta.to(r.dtype)
nz = Fn.log_softmax(m.run_with_hooks(toks_list[0], fwd_hooks=[(HOOK, zfn)])[0, -1].float(), -1)
noise = float(kl(clean_logp[0], nz))
print(f"noise_floor={noise:.2e}", flush=True)

# ---- mediation effects ----
print("measuring mediation effects ...", flush=True)
sample_t = torch.tensor(sample)
eff = np.full((M, P), np.nan, np.float32)        # nan = latent doesn't fire here
t0 = time.time()
for pi, (t, f) in enumerate(zip(toks_list, feats)):
    live = (f[:, sample_t] > 0).any(0).nonzero().flatten()  # idx into sample
    for s in range(0, live.numel(), CHUNK):
        sl = live[s:s + CHUNK]; bs = sl.numel()
        lat = sample_t[sl]                                   # global latent ids
        acts = f[:, lat]                                     # [seq, bs]
        delta = -acts.permute(1, 0).unsqueeze(-1) * W_dec[lat].unsqueeze(1)  # [bs, seq, d]
        def fn(r, hook, d=delta): return r + d.to(r.dtype)
        lg = m.run_with_hooks(t.repeat(bs, 1), fwd_hooks=[(HOOK, fn)])[:, -1, :].float()
        eff[sl.numpy(), pi] = kl(clean_logp[pi].unsqueeze(0), Fn.log_softmax(lg, -1)).numpy()
    if pi % 20 == 0:
        print(f"  prompt {pi}/{P}  {time.time()-t0:.0f}s", flush=True)

U = np.nanmean(eff, 1)                           # [M] unconditional effect
layer_median = float(np.median(U))
q20, q80 = np.percentile(U, [20, 80])
dead = np.where(U <= q20)[0]
topq = np.where(U >= q80)[0]
oracle_max = float(U.max()); oracle_topq = float(np.median(U[topq]))
print(f"layer_median={layer_median:.5f}  q20={q20:.5f}  dead={len(dead)}  "
      f"oracle_max={oracle_max:.4f} oracle_topq_median={oracle_topq:.4f}", flush=True)

fires_np = fires[:, sample_t].numpy()            # [P, M] context features

def try_rescue(i, eff_i, split_rng):
    """i: idx into sample. eff_i: [P] effects (nan where not firing)."""
    ctx = np.where(~np.isnan(eff_i))[0]
    perm = split_rng.permutation(len(ctx))
    ntr = int(round(TRAIN_FRAC * len(ctx)))
    tr, te = ctx[perm[:ntr]], ctx[perm[ntr:]]
    thr = HIGH_MULT * layer_median
    lab = eff_i >= thr
    if lab[tr].sum() == 0:
        return False, None
    feat_cols = [j for j in range(M) if j != i]
    Ftr = fires_np[tr][:, feat_cols]; Fte = fires_np[te][:, feat_cols]
    cur_tr = np.ones(len(tr), bool)
    lits = []
    cands = [(0.0 if cur_tr.sum() == 0 else lab[tr][cur_tr].mean(), int(cur_tr.sum()), list(lits), cur_tr.copy())]
    for _ in range(MAX_LIT):
        best = None
        for jc in range(Ftr.shape[1]):
            for pol in (True, False):
                g = cur_tr & (Ftr[:, jc] == pol)
                sup = int(g.sum())
                if sup < MIN_TRAIN_SUP:
                    continue
                prec = lab[tr][g].mean()
                key = (prec, sup)
                if best is None or key > best[0]:
                    best = (key, jc, pol, g)
        if best is None:
            break
        (prec, sup), jc, pol, g = best
        if lits and prec <= cands[-1][0]:
            break
        cur_tr = g; lits = lits + [(feat_cols[jc], pol)]
        cands.append((prec, sup, list(lits), cur_tr.copy()))
        if prec == 1.0:
            break
    cands.sort(key=lambda c: (c[0], c[1]), reverse=True)
    prec_tr, sup_tr, lits, _ = cands[0]
    if prec_tr < PREC:
        return False, None
    gte = np.ones(len(te), bool)
    for (gj, pol) in lits:
        col = feat_cols.index(gj)
        gte &= (Fte[:, col] == pol)
    if gte.sum() < MIN_TEST_SUP:
        return False, None
    prec_te = float(lab[te][gte].mean())
    cond_eff = float(eff_i[te[gte]].mean())
    ok = (prec_te >= PREC) and (cond_eff >= thr)
    return ok, dict(n_lit=len(lits), train_prec=round(float(prec_tr), 3),
                    test_prec=round(prec_te, 3), test_sup=int(gte.sum()),
                    cond_eff=round(cond_eff, 5), U=round(float(U[i]), 5))

print("learning gates (real) ...", flush=True)
rescued, details = 0, []
for i in dead:
    ok, d = try_rescue(int(i), eff[i], np.random.default_rng(1000 + int(i)))
    rescued += int(ok)
    if d: details.append(dict(latent=int(sample[i]), rescued=bool(ok), **d))
rescue_rate = rescued / len(dead)

print("learning gates (shuffle control) ...", flush=True)
sh_rng = np.random.default_rng(2)
sh_rescued = 0
for i in dead:
    e = eff[int(i)].copy()
    ctx = np.where(~np.isnan(e))[0]
    e[ctx] = e[ctx[sh_rng.permutation(len(ctx))]]
    ok, _ = try_rescue(int(i), e, np.random.default_rng(1000 + int(i)))
    sh_rescued += int(ok)
sh_rate = sh_rescued / len(dead)

if oracle_max < MIN_ORACLE_MAX or oracle_topq < MIN_ORACLE_TOPQ or noise > NOISE_MAX:
    verdict = "BROKEN_MEASUREMENT"
elif rescue_rate >= 0.25:
    verdict = "SUPPORTED" if sh_rate < max(0.05, 0.5 * rescue_rate) else "INCONCLUSIVE_OVERFIT"
elif rescue_rate < 0.05:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE" if sh_rate < max(0.05, 0.5 * rescue_rate) else "INCONCLUSIVE_OVERFIT"

report = dict(
    config=dict(L=L, M=M, P=P, min_fire=min_fire_used, HIGH_MULT=HIGH_MULT,
                PREC=PREC, MAX_LIT=MAX_LIT),
    layer_median=round(layer_median, 6), q20=round(float(q20), 6),
    oracle_max=round(oracle_max, 5), oracle_topq_median=round(oracle_topq, 5),
    noise_floor=noise, n_dead=len(dead),
    rescued=rescued, rescue_rate=round(rescue_rate, 4),
    shuffle_rescued=sh_rescued, shuffle_rate=round(sh_rate, 4),
    verdict=verdict, gate_details=details[:20])
print("=== REPORT ==="); print(json.dumps(report, indent=1))
json.dump(report, open(OUT + "/result.json", "w"), indent=1)
