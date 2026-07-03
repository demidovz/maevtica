#!/usr/bin/env python3
"""CRC external test — ROUND 4: ablation transfer + third family.

Attacks two round-3 limits:
 (A) ABLATION as well as additive steering — is causal role intervention-general?
 (B) THREE families (GPT-2 / Pythia-160m / OPT-125m, 3 tokenizers, all d_model
     768) tested on all 3 cross-family pairs — robustness, not one lucky pair.

Same honest guards: DECOUPLED metric (role fingerprint on broad neutral words,
answer words masked), STRONG geometry baseline (paired-activation alignment).
Prints ORACLE transfer per condition so a zero-effect bug can't fake a verdict.

Run: crc-venv311/bin/python crc_transfer_test_r4.py
"""
from __future__ import annotations
import json, os, sys, time, itertools
from pathlib import Path
import numpy as np, torch

torch.set_grad_enabled(False); torch.set_num_threads(max(1, os.cpu_count() or 4))
OUT = Path(__file__).parent; FRAC = 0.5; STEER_C = 0.5; SEEDS = list(range(8)); MIN_W = 4
MODELS = ["gpt2", "pythia-160m", "opt-125m"]

# reuse the round-3 category battery
import importlib.util
spec = importlib.util.spec_from_file_location("r3", OUT/"crc_transfer_test_r3.py")
r3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r3)
CATEGORIES, NEUTRAL, ALIGN_PROMPTS, BROAD = r3.CATEGORIES, r3.NEUTRAL, r3.ALIGN_PROMPTS, r3.BROAD
# probes for the signature: neutrals + one eliciting prompt per category (gives
# ablation something to remove); answers are masked out of the metric anyway.
PROBES = NEUTRAL + [tmpl[0] for _, tmpl in CATEGORIES.values()]

def log(*a): print(*a, file=sys.stderr, flush=True)
def stid(m, w):
    t = m.to_tokens(" "+w, prepend_bos=False)[0]; return int(t[0]) if t.shape[0] == 1 else None
def resid_last(m, ps, L):
    h = f"blocks.{L}.hook_resid_post"; out = []
    for p in ps:
        _, c = m.run_with_cache(m.to_tokens(p), names_filter=h); out.append(c[h][0, -1].float())
    return torch.stack(out)
def steer_vecs(m, L):
    per = {k: resid_last(m, t, L) for k, (_, t) in CATEGORIES.items()}
    g = torch.cat(list(per.values()), 0).mean(0)
    d = {k: per[k].mean(0)-g for k in CATEGORIES}
    n = resid_last(m, NEUTRAL, L).norm(dim=-1).mean().item()
    return {k: d[k]/d[k].norm()*(STEER_C*n) for k in d}
def deltas(m, vecs, L, probes):
    """Return clean and per-probe full-vocab deltas for ADD and ABLATE."""
    h = f"blocks.{L}.hook_resid_post"; keys = list(vecs); tl = [m.to_tokens(p) for p in probes]
    clean = torch.stack([m(t)[0, -1].float() for t in tl])
    Dadd = np.zeros((len(keys), len(probes), clean.shape[1]), np.float32)
    Dabl = np.zeros_like(Dadd)
    for ki, k in enumerate(keys):
        v = vecs[k]; u = v/v.norm()
        def f_add(r, hook, v=v): r[:, -1, :] = r[:, -1, :]+v.to(r.dtype); return r
        def f_abl(r, hook, u=u):
            r[:, -1, :] = r[:, -1, :]-(r[:, -1, :] @ u.to(r.dtype)).unsqueeze(-1)*u.to(r.dtype); return r
        for pi, t in enumerate(tl):
            Dadd[ki, pi] = (m.run_with_hooks(t, fwd_hooks=[(h, f_add)])[0, -1].float()-clean[pi]).numpy()
            Dabl[ki, pi] = (m.run_with_hooks(t, fwd_hooks=[(h, f_abl)])[0, -1].float()-clean[pi]).numpy()
    return keys, Dadd, Dabl
def cos_rows(A, B):
    A = A/(np.linalg.norm(A, axis=1, keepdims=True)+1e-9); B = B/(np.linalg.norm(B, axis=1, keepdims=True)+1e-9)
    return A @ B.T
def top1(S): return float((S.argmax(1) == np.arange(S.shape[0])).mean())

def main():
    t0 = time.time()
    M = {}
    for name in MODELS:
        log(f"[load] {name}"); from transformer_lens import HookedTransformer
        m = HookedTransformer.from_pretrained(name, device="cpu"); m.eval()
        L = round(m.cfg.n_layers*FRAC)
        vecs = steer_vecs(m, L)
        log(f"[deltas] {name}"); keys, Dadd, Dabl = deltas(m, vecs, L, PROBES)
        M[name] = dict(m=m, L=L, vecs=vecs, keys=keys, Dadd=Dadd, Dabl=Dabl)
    keys = list(CATEGORIES); K = len(keys)

    results = {}
    for a, b in itertools.combinations(MODELS, 2):
        ma, mb = M[a]["m"], M[b]["m"]
        # shared answer + broad-probe strings (single-token in both tokenizers)
        ans_a, ans_b, ans_by_cat_b = [], [], {}
        for k in keys:
            ida = [stid(ma, w) for w in CATEGORIES[k][0]]; idb = [stid(mb, w) for w in CATEGORIES[k][0]]
            good = [(x, y) for x, y in zip(ida, idb) if x is not None and y is not None]
            ans_by_cat_b[k] = [y for _, y in good]
            ans_a += [x for x, _ in good]; ans_b += [y for _, y in good]
        ans_words = {w for k in keys for w in CATEGORIES[k][0]}
        broad = [(stid(ma, w), stid(mb, w)) for w in dict.fromkeys(BROAD) if w not in ans_words]
        broad = [(x, y) for x, y in broad if x is not None and y is not None]
        bA = np.array([x for x, _ in broad]); bB = np.array([y for _, y in broad])
        # strong geometry (768->768) alignment on paired activations
        W = torch.linalg.lstsq(resid_last(ma, ALIGN_PROMPTS, M[a]["L"]),
                               resid_last(mb, ALIGN_PROMPTS, M[b]["L"])).solution.numpy()
        VA = np.stack([M[a]["vecs"][k].numpy() for k in keys]); VB = np.stack([M[b]["vecs"][k].numpy() for k in keys])
        acc_geo = top1(cos_rows(VA @ W, VB))
        pair = {"broad_probe": len(broad), "geometry": acc_geo, "interventions": {}}
        for iv, key in (("add", "Dadd"), ("ablate", "Dabl")):
            Da, Db = M[a][key], M[b][key]
            S = cos_rows(Da.mean(1)[:, bA], Db.mean(1)[:, bB])
            acc = top1(S)
            # seeds via bootstrap over probes
            accs = []
            for s in SEEDS:
                idx = np.random.default_rng(s).integers(0, Da.shape[1], Da.shape[1])
                accs.append(top1(cos_rows(Da[:, idx].mean(1)[:, bA], Db[:, idx].mean(1)[:, bB])))
            pred = S.argmax(1)
            tr = float(np.mean([Db[int(pred[i])][:, ans_by_cat_b[keys[i]]].mean() for i in range(K)]))
            tro = float(np.mean([Db[i][:, ans_by_cat_b[keys[i]]].mean() for i in range(K)]))
            pair["interventions"][iv] = {"crc_mean": float(np.mean(accs)), "crc_std": float(np.std(accs)),
                                          "crc_point": acc, "transfer_crc": tr, "transfer_oracle": tro}
        results[f"{a}__{b}"] = pair
        log(f"[pair {a}/{b}] geom={acc_geo:.2f} add={pair['interventions']['add']['crc_mean']:.2f} "
            f"abl={pair['interventions']['ablate']['crc_mean']:.2f} (oracle add tr="
            f"{pair['interventions']['add']['transfer_oracle']:.2f} abl tr="
            f"{pair['interventions']['ablate']['transfer_oracle']:.2f})")

    out = {"round": 4, "models": MODELS, "K": K, "chance": 1.0/K, "pairs": results,
           "seconds": round(time.time()-t0, 1)}
    (OUT/"results_r4.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False)); return 0

if __name__ == "__main__":
    sys.exit(main())
