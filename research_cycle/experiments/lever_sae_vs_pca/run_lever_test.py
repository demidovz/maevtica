#!/usr/bin/env python3
"""Behavioral-lever falsification: trained SAE vs PCA vs random dict (matched N).

Preregistration: ./PREREGISTRATION.md  (decision rule fixed there).
venv: ~/.local/state/mst/crc-venv311/bin/python
"""
from __future__ import annotations
import glob, json, sys, os
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(8)
from transformer_lens import HookedTransformer
from safetensors.torch import load_file

L = 7
HOOK = f"blocks.{L}.hook_resid_pre"
N = int(os.environ.get("LEVER_N", 512))   # matched dictionary count
TAU = 3.0                    # lever magnitude threshold (logits)
MIN_ORACLE = 3.0
MONO_TOL = 0.05
BEHAVIORS = {"dog": 3290, "war": 1175, "love": 1842}  # token ids verified
DOSE_ABS = float(os.environ.get("LEVER_D", 80))  # max dose (abs coef); calibrated: all 3 oracles monotonic & >=3 up to here, war saturates beyond
CHUNK = 384
rng = np.random.default_rng(0)

print("loading gpt2 ...", flush=True)
m = HookedTransformer.from_pretrained("gpt2", device="cpu"); m.eval()
d_model = m.cfg.d_model

# ---- corpus (diverse short prompts) ----
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
]

def cache(prompts, hook):
    out=[]
    for p in prompts:
        _,c=m.run_with_cache(m.to_tokens(p), names_filter=hook)
        out.append(c[hook][0])   # [seq, d]
    return out

print("collecting activations ...", flush=True)
acts_list = cache(CORPUS, HOOK)
X = torch.cat(acts_list, 0).float()          # [Ntok, d]
mean_norm = X.norm(dim=-1).mean().item()
D = DOSE_ABS
DOSES = np.array([0.0, D/4, D/2, D], dtype=np.float32)
print(f"Ntok={X.shape[0]} mean_norm={mean_norm:.1f} D={D:.1f} doses={DOSES.round(1)}", flush=True)

# ---- SAE dict: top-N live features' decoder rows ----
base = glob.glob("/home/friemann/.cache/huggingface/hub/models--jbloom--GPT2-Small-SAEs-Reformatted/snapshots/*/"+HOOK)[0]
sd = load_file(base+"/sae_weights.safetensors")
W_enc = sd["W_enc"].float(); b_enc = sd["b_enc"].float(); W_dec = sd["W_dec"].float(); b_dec = sd["b_dec"].float()
# encoder activations over corpus to rank features (standard SAE: relu(x W_enc + b_enc), x centered by b_dec)
feat = torch.relu((X - b_dec) @ W_enc + b_enc)     # [Ntok, d_sae]
mean_act = feat.mean(0)
top_sae = torch.topk(mean_act, N).indices
sae_dirs = W_dec[top_sae]                            # [N, d]
sae_dirs = (sae_dirs / sae_dirs.norm(dim=-1, keepdim=True)).numpy()
n_live = int((mean_act > 1e-6).sum())
print(f"SAE live features={n_live}/{W_dec.shape[0]}  selected top-{N}", flush=True)

# ---- PCA dict: top-N PCs ----
Xc = (X - X.mean(0)).numpy()
U,S,Vt = np.linalg.svd(Xc, full_matrices=False)
pca_dirs = Vt[:N]                                    # [N, d] already unit rows
pca_dirs = pca_dirs / (np.linalg.norm(pca_dirs,axis=1,keepdims=True)+1e-9)

# ---- RANDOM dict ----
rand_dirs = rng.standard_normal((N, d_model)).astype(np.float32)
rand_dirs = rand_dirs / (np.linalg.norm(rand_dirs,axis=1,keepdims=True)+1e-9)

# ---- Oracle directions per behavior (diff-in-means concept vs neutral) ----
NEUTRAL = ["The","It was","I think that","We went to","There is a","People often say"]
CONCEPT = {
 "dog":  ["The dog barked at the","My dog loves to","A puppy and its","The dog wagged its tail","Dogs are loyal","She walked her dog in the"],
 "war":  ["The war between the two","Soldiers fought in the","The battle raged on the","War brings destruction and","The army prepared for war","Generals planned the war"],
 "love": ["I love you so","She fell in love with","Love is a powerful","They shared a deep love","He wrote a love letter to","Love conquers all and"],
}
neutral_resid = torch.stack([a[-1] for a in cache(NEUTRAL, HOOK)]).float().mean(0)
oracle_dirs = {}
for b, prom in CONCEPT.items():
    cr = torch.stack([a[-1] for a in cache(prom, HOOK)]).float().mean(0)
    v = (cr - neutral_resid).numpy(); oracle_dirs[b] = v/ (np.linalg.norm(v)+1e-9)

# ---- probes ----
PROBES = ["The next word is","Yesterday I saw a","My favorite is the","She talked about the","He pointed at the","They discovered a"]
probe_toks = [m.to_tokens(p) for p in PROBES]

# clean last-token logits per probe
clean = np.stack([m(t)[0,-1].float().numpy() for t in probe_toks])   # [P, vocab]

def steered_logits(dirs):
    """dirs [K,d] -> delta logits [K, P, vocab, n_nonzero_doses] using +sign only.
    Returns full-vocab delta at each nonzero dose for each (dir,probe)."""
    K = dirs.shape[0]; P = len(probe_toks)
    nz = DOSES[1:]                                   # nonzero doses
    dt = torch.tensor(dirs, dtype=torch.float32)     # [K,d]
    out = np.zeros((K, P, clean.shape[1], len(nz)), np.float32)
    for pi,t in enumerate(probe_toks):
        tb = t.repeat(1,1)  # [1,seq]
        for di,c in enumerate(nz):
            add = (dt * float(c))                     # [K,d]
            for s in range(0, K, CHUNK):
                sl = slice(s, min(s+CHUNK, K))
                bs = sl.stop - sl.start
                tt = t.repeat(bs,1)
                addc = add[sl].to(torch.float32)
                def fn(r, hook, a=addc):
                    r[:, -1, :] = r[:, -1, :] + a.to(r.dtype); return r
                lg = m.run_with_hooks(tt, fwd_hooks=[(HOOK, fn)])[:, -1, :].float().numpy()
                out[sl, pi, :, di] = lg - clean[pi][None,:]
    return out   # [K,P,vocab,ndose]

def eval_dict(name, dirs):
    """Return per-behavior arrays. Test both signs by symmetry: delta(-d) at logit = using +d? no.
    We steer +d only; to get -d we re-run. Cheaper: run +d and -d together by stacking."""
    both = np.concatenate([dirs, -dirs], 0)          # [2N,d]
    dl = steered_logits(both)                        # [2N,P,vocab,ndose]
    res = {}
    K = dirs.shape[0]
    for b, tid in BEHAVIORS.items():
        dt = dl[:, :, tid, :]                         # [2N,P,ndose]  delta logit_T
        # mean over probes -> [2N, ndose]; prepend dose0=0
        mp = dt.mean(1)                               # [2N,ndose]
        full = np.concatenate([np.zeros((mp.shape[0],1),np.float32), mp], 1)  # [2N,4]
        # pick better sign per original direction
        pos = full[:K]; neg = full[K:]
        pick_pos = pos[:, -1] >= neg[:, -1]
        chosen = np.where(pick_pos[:,None], pos, neg) # [K,4]
        # also keep per-probe delta at max dose for chosen sign (for bootstrap CI)
        dt_max_pos = dt[:K, :, -1]; dt_max_neg = dt[K:, :, -1]
        chosen_probe_max = np.where(pick_pos[:,None], dt_max_pos, dt_max_neg)  # [K,P]
        # lever criteria
        incr = np.diff(chosen, axis=1)                # [K,3]
        mono = (incr > MONO_TOL).all(1)
        mag = chosen[:, -1] >= TAU
        lever = mono & mag
        best_idx = int(np.argmax(chosen[:, -1]))
        res[b] = dict(count=int(lever.sum()),
                      best=float(chosen[best_idx, -1]),
                      best_idx=best_idx,
                      best_probe_max=chosen_probe_max[best_idx].copy(),
                      chosen_maxdose=chosen[:, -1].copy())
    return res

print("evaluating SAE ...", flush=True);  sae = eval_dict("SAE", sae_dirs)
print("evaluating PCA ...", flush=True);  pca = eval_dict("PCA", pca_dirs)
print("evaluating RANDOM ...", flush=True); rnd = eval_dict("RANDOM", rand_dirs)
print("evaluating ORACLE ...", flush=True)
oracle = {b: eval_dict("ORACLE_"+b, oracle_dirs[b][None,:]) for b in BEHAVIORS}

def boot_ci(probe_max, iters=2000):
    P = len(probe_max)
    bs = [probe_max[rng.integers(0,P,P)].mean() for _ in range(iters)]
    return float(np.percentile(bs,2.5)), float(np.percentile(bs,97.5))

report = {"config": dict(L=L, N=N, TAU=TAU, MIN_ORACLE=MIN_ORACLE, D=float(D),
                         mean_norm=mean_norm, sae_live=n_live), "behaviors": {}}
refute_flags = []
for b in BEHAVIORS:
    oracle_best = oracle[b][b]["best"]
    sae_lo, sae_hi = boot_ci(sae[b]["best_probe_max"])
    pca_lo, pca_hi = boot_ci(pca[b]["best_probe_max"])
    strictly_more = (sae[b]["count"] >= 2*pca[b]["count"]) and (sae[b]["count"]-pca[b]["count"] >= 5)
    strictly_stronger = sae[b]["best"] > pca_hi
    refuted = strictly_more or strictly_stronger
    refute_flags.append(refuted)
    report["behaviors"][b] = dict(
        oracle_best=round(oracle_best,3),
        sae=dict(count=sae[b]["count"], best=round(sae[b]["best"],3), ci=[round(sae_lo,3),round(sae_hi,3)]),
        pca=dict(count=pca[b]["count"], best=round(pca[b]["best"],3), ci=[round(pca_lo,3),round(pca_hi,3)]),
        random=dict(count=rnd[b]["count"], best=round(rnd[b]["best"],3)),
        sae_within_pca_ci=bool(pca_lo <= sae[b]["best"] <= pca_hi),
        pca_wins=bool(pca[b]["best"] > sae[b]["best"]),
        strictly_more=bool(strictly_more), strictly_stronger=bool(strictly_stronger),
        refuted=bool(refuted))

oracle_min = min(oracle[b][b]["best"] for b in BEHAVIORS)
if oracle_min < MIN_ORACLE:
    verdict = "BROKEN_MEASUREMENT"
elif sum(refute_flags) >= 2:
    verdict = "REFUTED"
elif sum(refute_flags) <= 1:
    verdict = "SUPPORTED"
else:
    verdict = "INCONCLUSIVE"
report["oracle_min"] = round(oracle_min,3)
report["verdict"] = verdict
print("=== REPORT ===")
print(json.dumps(report, indent=1))
json.dump(report, open(__file__.rsplit("/",1)[0]+"/result.json","w"), indent=1)
