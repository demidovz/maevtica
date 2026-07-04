#!/usr/bin/env python3
"""Behavioral Quotient Feature — minimal falsification per PREREG.md.

Clause (i): effective rank k90 of latent-steering effects, 32k vs 128k SAE.
Clause (ii): same-quotient-coordinate latents with decoder |cos|<0.1 steer
coherently (signed effect cos > 0.8, above a fair random-pair null).

venv: ~/.local/state/mst/crc-venv311/bin/python
"""
import os, json, sys, numpy as np, torch

torch.set_grad_enabled(False)
torch.set_num_threads(max(1, os.cpu_count() or 4))
rng = np.random.default_rng(0)

SAE_DIR = os.environ.get("SAE_DIR", "/tmp/claude-1000/-home-friemann-workspace-maestratica/b9bd7464-4c05-40a5-ba2b-b36c0ae2836c/scratchpad/saes")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result.json")
LAYER, N_SAMPLE, TOPK = 6, 256, 32

PROBES = [
    "The weather today is",
    "In the middle of the city there was a",
    "She looked at him and said",
    "The most important thing about science is",
    "Yesterday the government announced",
    "My favorite food has always been",
    "The history of the war shows that",
    "He opened the door and saw",
]

CORPUS = [
    "The dog ran across the park chasing a ball.",
    "Quantum mechanics describes the behavior of particles.",
    "She poured coffee into the ceramic mug slowly.",
    "The stock market fell sharply on Tuesday morning.",
    "Ancient Rome built roads across the entire empire.",
    "My grandmother bakes bread every Sunday afternoon.",
    "The spacecraft entered orbit around the red planet.",
    "Lawyers argued the case before the supreme court.",
    "Heavy rain flooded the streets of the small town.",
    "The orchestra played a symphony by Beethoven.",
    "Children laughed while playing in the schoolyard.",
    "The recipe calls for two cups of flour and sugar.",
    "Soldiers marched through the gates at dawn.",
    "The algorithm sorts numbers in logarithmic time.",
    "Fishermen returned with a large catch of salmon.",
    "The painting depicted a stormy sea at sunset.",
    "Voters lined up outside the polling stations early.",
    "The chemistry teacher demonstrated a reaction.",
    "Wolves howled in the distance under the full moon.",
    "The startup raised millions in venture funding.",
    "A gentle breeze moved through the autumn leaves.",
    "The surgeon completed the operation successfully.",
    "Tourists photographed the ancient stone temple.",
    "The novel explores themes of memory and loss.",
    "Engineers tested the bridge for structural flaws.",
    "The cat slept curled up on the warm windowsill.",
    "Prices of oil rose after the announcement.",
    "The choir sang hymns in the old cathedral.",
    "Farmers harvested wheat before the storm arrived.",
    "The professor explained the theory of evolution.",
]


def log(*a):
    print(*a, flush=True)


def load_sae(path):
    sd = torch.load(path, map_location="cpu")
    if not isinstance(sd, dict):
        sd = sd.state_dict()
    keys = {k.lower(): k for k in sd.keys()}
    log("SAE keys:", list(sd.keys())[:8])
    def get(*names):
        for n in names:
            if n in sd: return sd[n].float()
        for n in names:
            if n.lower() in keys: return sd[keys[n.lower()]].float()
        return None
    enc = get("encoder.weight", "W_enc", "encoder")
    dec = get("decoder.weight", "W_dec", "decoder")
    pre = get("pre_bias", "b_dec", "bias")
    lat = get("latent_bias", "b_enc")
    d_model = 768
    if enc is not None and enc.shape[0] == d_model and enc.shape[1] != d_model:
        enc = enc.T
    if dec is not None and dec.shape[0] != d_model and dec.shape[1] == d_model:
        dec = dec.T  # want [d_model, n_latents]
    return enc, dec, pre, lat


def main():
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    model.eval()
    hookname = f"blocks.{LAYER}.hook_resid_post"
    V = model.cfg.d_vocab

    saes = {}
    for tag, fn in [("32k", "sae32k_L6.pt"), ("128k", "sae128k_L6.pt")]:
        enc, dec, pre, lat = load_sae(os.path.join(SAE_DIR, fn))
        log(tag, "enc", None if enc is None else tuple(enc.shape),
            "dec", None if dec is None else tuple(dec.shape))
        saes[tag] = (enc, dec, pre, lat)

    # ---- corpus activations for alive selection ----
    X = []
    for p in CORPUS:
        _, c = model.run_with_cache(model.to_tokens(p), names_filter=hookname)
        X.append(c[hookname][0].float())
    X = torch.cat(X, 0)  # [T, 768]
    log("corpus acts:", tuple(X.shape))

    picked = {}
    for tag, (enc, dec, pre, lat) in saes.items():
        Z = X - (pre if pre is not None else 0)
        A = Z @ enc.T
        if lat is not None:
            A = A + lat
        n_lat = A.shape[1]
        thresh = torch.topk(A, TOPK, dim=1).values[:, -1:]
        alive = ((A >= thresh) & (A > 0)).any(0).nonzero().squeeze(-1).numpy()
        log(tag, f"alive latents (in top-{TOPK} at least once): {len(alive)} / {n_lat}")
        idx = rng.choice(alive, size=min(N_SAMPLE, len(alive)), replace=False)
        Dm = dec[:, idx].T.clone()  # [N, 768]
        Dm = Dm / Dm.norm(dim=1, keepdim=True)
        picked[tag] = (idx, Dm)

    # ---- probe batch ----
    toks = [model.to_tokens(p)[0] for p in PROBES]
    lens = [t.shape[0] for t in toks]
    L = max(lens)
    pad_id = model.tokenizer.eos_token_id
    batch = torch.full((len(toks), L), pad_id, dtype=torch.long)
    for i, t in enumerate(toks):
        batch[i, : t.shape[0]] = t
    last = torch.tensor([l - 1 for l in lens])

    def last_logits(tokens, hooks=None):
        if hooks:
            out = model.run_with_hooks(tokens, fwd_hooks=hooks)
        else:
            out = model(tokens)
        return out[torch.arange(len(lens)), last].float()  # [8, V]

    clean = last_logits(batch)
    clean2 = last_logits(batch)
    noise_floor = float((clean2 - clean).norm(dim=-1).mean())

    _, c = model.run_with_cache(batch, names_filter=hookname)
    rn = c[hookname][torch.arange(len(lens)), last].norm(dim=-1).mean().item()
    ALPHA = 0.5 * rn
    log(f"mean clean resid norm L{LAYER}: {rn:.2f}  alpha={ALPHA:.2f}  noise_floor={noise_floor:.2e}")

    def steer_effect(vec, alpha=None):
        a = ALPHA if alpha is None else alpha
        v = (vec / vec.norm() * a)
        def fn(r, hook, v=v):
            return r + v.to(r.dtype)
        d = last_logits(batch, hooks=[(hookname, fn)]) - clean  # [8, V]
        return d.numpy().astype(np.float32)

    # ---- steering runs ----
    eff_mean, eff_h1, eff_h2, eff_norm = {}, {}, {}, {}
    for tag in ("32k", "128k"):
        idx, Dm = picked[tag]
        N = Dm.shape[0]
        Em = np.zeros((N, V), np.float32)
        H1 = np.zeros((N, V), np.float32)
        H2 = np.zeros((N, V), np.float32)
        for i in range(N):
            d = steer_effect(Dm[i])
            Em[i] = d.mean(0)
            H1[i] = d[:4].mean(0)
            H2[i] = d[4:].mean(0)
            if (i + 1) % 32 == 0:
                log(f"{tag}: {i+1}/{N}")
        eff_mean[tag], eff_h1[tag], eff_h2[tag] = Em, H1, H2
        eff_norm[tag] = float(np.linalg.norm(Em, axis=1).mean())

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    # ---- O1 split-half ----
    sh = []
    for tag in ("32k", "128k"):
        for i in range(len(eff_h1[tag])):
            sh.append(cos(eff_h1[tag][i], eff_h2[tag][i]))
    o1_median = float(np.median(sh))

    # ---- O2 near-duplicate cross-SAE pairs ----
    D32, D128 = picked["32k"][1].numpy(), picked["128k"][1].numpy()
    C = D32 @ D128.T
    dup = np.argwhere(C > 0.9)
    o2_mode, o2_median, o2_n = None, None, 0
    if len(dup) >= 10:
        dup = dup[rng.permutation(len(dup))[:50]]
        vals = [cos(eff_mean["32k"][i], eff_mean["128k"][j]) for i, j in dup]
        o2_mode, o2_median, o2_n = "cross_sae_duplicates", float(np.median(vals)), len(vals)
    else:
        vals = []
        sel = rng.choice(N_SAMPLE, 20, replace=False)
        for i in sel:
            d2 = steer_effect(picked["32k"][1][i], alpha=1.1 * ALPHA).mean(0)
            vals.append(cos(eff_mean["32k"][i], d2))
        o2_mode, o2_median, o2_n = "rescale_1.1alpha", float(np.median(vals)), len(vals)
        o2_dup_found = len(dup)
    log(f"O1 split-half median={o1_median:.3f}  O2[{o2_mode}] median={o2_median:.3f} n={o2_n} (dup pairs found: {len(dup)})")

    broken = (o1_median < 0.6) or (o2_median < (0.8 if o2_mode == "cross_sae_duplicates" else 0.9))

    # ---- clause (i): k90 ----
    def k90(M):
        Mc = M - M.mean(0, keepdims=True)
        G = Mc @ Mc.T
        ev = np.linalg.eigvalsh(G)[::-1]
        ev = np.clip(ev, 0, None)
        cs = np.cumsum(ev) / ev.sum()
        return int(np.searchsorted(cs, 0.90) + 1)

    k32, k128 = k90(eff_mean["32k"]), k90(eff_mean["128k"])
    kd32, kd128 = k90(D32), k90(D128)
    ratio = max(k32, k128) / min(k32, k128)
    rel = abs(k128 - k32) / min(k32, k128)
    sample_limited = (k32 >= 0.8 * N_SAMPLE) and (k128 >= 0.8 * N_SAMPLE)
    if sample_limited:
        ci = "inconclusive_sample_limited"
    elif rel <= 0.10:
        ci = "supported"
    elif ratio >= 1.5:
        ci = "refuted"
    else:
        ci = "inconclusive"
    log(f"clause i: k90_32k={k32} k90_128k={k128} ratio={ratio:.3f} rel={rel:.3f} -> {ci}  (decoder-geometry k90: {kd32}/{kd128})")

    # ---- clause (ii): quotient coords from 32k ridge fit, tested on 128k ----
    E32 = eff_mean["32k"]
    G = D32.T @ D32
    lam = 1e-2 * np.trace(G) / 768
    WT = np.linalg.solve(G + lam * np.eye(768), D32.T @ E32)  # [768, V]
    U, S, _ = np.linalg.svd(WT, full_matrices=False)
    K = k32
    UK = U[:, :K]  # quotient coordinates (residual space)

    P = D128 @ UK  # [N, K]
    dom = P**2 / (np.sum(P**2, axis=1, keepdims=True) + 1e-12)
    ci_idx = np.argmax(np.abs(P), axis=1)
    sigma = dom[np.arange(len(P)), ci_idx]

    def collect_pairs(thr):
        pairs = []
        for a in range(len(P)):
            if sigma[a] < thr: continue
            for b in range(a + 1, len(P)):
                if sigma[b] < thr or ci_idx[a] != ci_idx[b]: continue
                if abs(float(D128[a] @ D128[b])) >= 0.1: continue
                pairs.append((a, b))
        return pairs

    thr_used = 0.5
    pairs = collect_pairs(0.5)
    if len(pairs) < 5:
        thr_used = 0.4
        pairs = collect_pairs(0.4)
    if len(pairs) > 200:
        pairs = [pairs[i] for i in rng.permutation(len(pairs))[:200]]

    signed = []
    for a, b in pairs:
        s = np.sign(P[a, ci_idx[a]] * P[b, ci_idx[b]])
        signed.append(s * cos(eff_mean["128k"][a], eff_mean["128k"][b]))
    med_signed = float(np.median(signed)) if signed else None

    # fair null: random 128k pairs with decoder |cos|<0.1, no coordinate requirement
    null_vals = []
    tries = 0
    while len(null_vals) < 200 and tries < 20000:
        a, b = rng.integers(0, len(P), 2)
        tries += 1
        if a == b or abs(float(D128[a] @ D128[b])) >= 0.1: continue
        null_vals.append(abs(cos(eff_mean["128k"][a], eff_mean["128k"][b])))
    null_med = float(np.median(null_vals))

    if med_signed is None or len(pairs) < 5:
        cii = "inconclusive_no_pairs"
    elif null_med > 0.8:
        cii = "inconclusive_null_too_high"
    elif med_signed > 0.8 and (med_signed - null_med) >= 0.2:
        cii = "supported"
    elif med_signed < 0.4:
        cii = "refuted"
    else:
        cii = "inconclusive"
    log(f"clause ii: n_pairs={len(pairs)} (thr={thr_used}) median_signed_cos={med_signed} null={null_med:.3f} -> {cii}")

    # ---- EXPLORATORY (post-hoc, NOT decisive; preregistered rule stands) ----
    # Graded version of clause ii's spirit: among low-decoder-cos 128k pairs,
    # does quotient-profile similarity predict steering-effect cosine?
    Q = P * S[:K]  # singular-value-weighted quotient profiles [N, K]
    expl_pairs, q_sims, e_coss = [], [], []
    tries = 0
    while len(expl_pairs) < 500 and tries < 50000:
        a, b = rng.integers(0, len(P), 2)
        tries += 1
        if a == b or abs(float(D128[a] @ D128[b])) >= 0.1: continue
        expl_pairs.append((a, b))
        q_sims.append(cos(Q[a], Q[b]))
        e_coss.append(cos(eff_mean["128k"][a], eff_mean["128k"][b]))
    from scipy.stats import spearmanr
    try:
        rho, _ = spearmanr(q_sims, e_coss)
        rho = float(rho)
    except Exception:
        rho = None
    q_sims, e_coss = np.array(q_sims), np.array(e_coss)
    hi = q_sims > 0.8
    expl = dict(
        n_pairs=len(expl_pairs),
        spearman_qsim_vs_effectcos=rho,
        n_qsim_gt_08=int(hi.sum()),
        median_effectcos_when_qsim_gt_08=(float(np.median(e_coss[hi])) if hi.any() else None),
        median_effectcos_all=float(np.median(e_coss)),
    )
    log("EXPLORATORY:", json.dumps(expl))

    np.savez_compressed(os.path.join(os.path.dirname(OUT), "effects.npz"),
                        E32=eff_mean["32k"], E128=eff_mean["128k"],
                        D32=D32, D128=D128, UK=UK, S=S[:K])

    if broken:
        verdict = "BROKEN_MEASUREMENT"
    elif ci == "refuted" or cii == "refuted":
        verdict = "REFUTED"
    elif ci == "supported" and cii == "supported":
        verdict = "SUPPORTED"
    else:
        verdict = "INCONCLUSIVE"

    res = dict(
        model="gpt2-small", layer=LAYER, alpha=ALPHA, n_sample=N_SAMPLE,
        noise_floor=noise_floor, mean_effect_norm=eff_norm,
        oracle=dict(o1_splithalf_median=o1_median, o2_mode=o2_mode,
                    o2_median=o2_median, o2_n=o2_n, broken=bool(broken)),
        clause_i=dict(k90_32k=k32, k90_128k=k128, ratio=ratio, rel_diff=rel,
                      decoder_k90_32k=kd32, decoder_k90_128k=kd128,
                      sample_limited=bool(sample_limited), verdict=ci),
        clause_ii=dict(n_pairs=len(pairs), dominance_thr=thr_used, K=K,
                       median_signed_cos=med_signed, null_median=null_med,
                       n_dominant=int((sigma >= thr_used).sum()), verdict=cii),
        exploratory_not_decisive=expl,
        verdict=verdict,
    )
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    log(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
