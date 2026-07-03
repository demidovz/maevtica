#!/usr/bin/env python3
"""External test of the "Causal Role Carrier" (CRC) concept.

Prediction #1 (preregistered, see PREREGISTRATION.md):
  Matching mechanisms across models by intervention-stable causal ROLE
  transfers steering effects better than matching by activation-similarity
  (geometry) or feature label.

Design: K token-promotion steering vectors per model (Pythia-160m, -410m).
Hide the correspondence, try to recover it by 3 methods, measure top-1
matching accuracy and behavioral transfer.

Run:  crc-venv311/bin/python crc_transfer_test.py
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch

torch.set_grad_enabled(False)
torch.set_num_threads(max(1, os.cpu_count() or 4))
DEVICE = "cpu"
OUT = Path(__file__).parent
SEEDS = [0, 1, 2, 3, 4]
FRAC_LAYER = 0.5      # steer at middle layer (same fraction in both models)
STEER_C = 0.5         # injected vector norm = STEER_C * mean resid norm (fixed a priori)

# ── K behaviors: single-token category members + eliciting templates ──────────
CATEGORIES = {
    "animals":  (["cat","dog","horse","lion","bird","fish","cow","bear","wolf","sheep","goat","fox"],
                 ["The farmer fed the","At the zoo we saw a","My pet is a","In the forest lived a","The vet examined the","On the farm there was a"]),
    "colors":   (["red","blue","green","yellow","black","white","brown","pink","purple","orange","grey"],
                 ["Her favorite color is","The sky slowly turned","He painted the wall","The old car was","My new shirt is","The bright flower was"]),
    "numbers":  (["one","two","three","four","five","six","seven","eight","nine","ten","twelve"],
                 ["The total came to","She counted up to","He was only","There were exactly","I bought just","The answer is"]),
    "countries":(["France","Japan","Brazil","Egypt","China","India","Spain","Italy","Canada","Germany","Mexico"],
                 ["Last summer we flew to","She was born in","The capital city of","He is travelling to","The team came from","My visa is for"]),
    "body":     (["hand","foot","head","arm","leg","eye","nose","ear","finger","knee","mouth"],
                 ["The doctor examined my","He gently touched her","She injured her left","A tattoo covered his","The glove fit his","She raised her"]),
    "fruits":   (["apple","banana","orange","grape","lemon","peach","cherry","melon","mango","plum","pear"],
                 ["For dessert she ate an","The basket was full of","He picked a ripe","My favorite fruit is the","She squeezed a fresh","On the tree grew a"]),
    "metals":   (["gold","iron","silver","copper","steel","tin","lead","bronze","zinc","nickel"],
                 ["The ring was made of","The bridge is built from","They mined for","The coin was pure","The pipe is made of","The sword was forged from"]),
}
NEUTRAL = ["The","I think that","Yesterday,","She said that","It was a","We went to","Here is the","They found the",
           "After a while,","In the morning","He looked at the","Later that day"]

def log(*a): print(*a, file=sys.stderr, flush=True)

def load_model(name):
    from transformer_lens import HookedTransformer
    log(f"[load] {name} …")
    m = HookedTransformer.from_pretrained(name, device=DEVICE)
    m.eval()
    return m

def single_token_words(model, words):
    """Keep words that encode to exactly one token with a leading space."""
    keep = []
    for w in words:
        ids = model.to_tokens(" " + w, prepend_bos=False)[0]
        if ids.shape[0] == 1:
            keep.append((w, int(ids[0])))
    return keep

def resid_last(model, prompts, layer):
    """Mean residual (resid_post) at the last token, over prompts, at `layer`."""
    hook = f"blocks.{layer}.hook_resid_post"
    accs = []
    for p in prompts:
        toks = model.to_tokens(p)  # prepend BOS
        _, cache = model.run_with_cache(toks, names_filter=hook)
        accs.append(cache[hook][0, -1].float())
    return torch.stack(accs)  # [n_prompts, d_model]

def steering_vectors(model, cats, layer):
    """v[k] = mean resid on category-k eliciting prompts  -  grand mean over all."""
    per_cat = {k: resid_last(model, tmpl, layer) for k, (_, tmpl) in cats.items()}
    grand = torch.cat(list(per_cat.values()), 0).mean(0)
    vecs = {k: (per_cat[k].mean(0) - grand) for k in cats}
    return vecs

def resid_norm(model, prompts, layer):
    r = resid_last(model, prompts, layer)
    return r.norm(dim=-1).mean().item()

def add_hook(vec, layer):
    name = f"blocks.{layer}.hook_resid_post"
    def fn(resid, hook):
        resid[:, -1, :] = resid[:, -1, :] + vec.to(resid.dtype)
        return resid
    return name, fn

def steered_logit_delta(model, vec, layer, probes, vocab_ids):
    """Mean over probes of (steered - clean) last-token logits, restricted to vocab_ids."""
    name, fn = add_hook(vec, layer)
    deltas = []
    for p in probes:
        toks = model.to_tokens(p)
        clean = model(toks)[0, -1]
        steer = model.run_with_hooks(toks, fwd_hooks=[(name, fn)])[0, -1]
        deltas.append((steer - clean)[vocab_ids].float())
    return torch.stack(deltas).mean(0)  # [len(vocab_ids)]

def align_map(model_a, model_b, shared_ids_a, shared_ids_b):
    """Least-squares W: A-space -> B-space from paired input embeddings over shared vocab."""
    Ea = model_a.W_E[shared_ids_a].float()  # [V, d_a]
    Eb = model_b.W_E[shared_ids_b].float()  # [V, d_b]
    # solve Ea @ W ≈ Eb
    W = torch.linalg.lstsq(Ea, Eb).solution   # [d_a, d_b]
    return W

def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))

def topk_match_accuracy(score_matrix):
    """score_matrix[k, j] similarity of A-behavior k to B-behavior j. Top-1 = argmax_j == k."""
    K = score_matrix.shape[0]
    pred = score_matrix.argmax(1)
    return float((pred == torch.arange(K)).float().mean())

def main():
    t0 = time.time()
    ma = load_model("pythia-160m")
    mb = load_model("pythia-410m")
    La = round(ma.cfg.n_layers * FRAC_LAYER)
    Lb = round(mb.cfg.n_layers * FRAC_LAYER)
    log(f"[layers] A={ma.cfg.n_layers}(steer {La})  B={mb.cfg.n_layers}(steer {Lb})")

    # single-token filtered category members (shared tokenizer -> same ids in both)
    cats_a, cats_b, cat_ids = {}, {}, {}
    for k, (words, tmpl) in CATEGORIES.items():
        stw = single_token_words(ma, words)
        ids = [tid for _, tid in stw]
        cat_ids[k] = ids
        cats_a[k] = (words, tmpl); cats_b[k] = (words, tmpl)
    K = len(CATEGORIES)
    shared_vocab = sorted({tid for ids in cat_ids.values() for tid in ids})
    log(f"[vocab] K={K} categories, shared vocab {len(shared_vocab)} tokens")

    # steering vectors per model
    va = steering_vectors(ma, cats_a, La)
    vb = steering_vectors(mb, cats_b, Lb)
    na = resid_norm(ma, NEUTRAL, La); nb = resid_norm(mb, NEUTRAL, Lb)
    # scale each steering vector to fixed injected norm
    for d, n in ((va, na), (vb, nb)):
        for k in d:
            d[k] = d[k] / d[k].norm() * (STEER_C * n)

    keys = list(CATEGORIES.keys())

    # alignment map for geometry baseline (fair chance): A-space -> B-space
    W = align_map(ma, mb, shared_vocab, shared_vocab)

    # ── causal effect signatures over shared vocab (comparable across models) ──
    eff_a = torch.stack([steered_logit_delta(ma, va[k], La, NEUTRAL, shared_vocab) for k in keys])
    eff_b = torch.stack([steered_logit_delta(mb, vb[k], Lb, NEUTRAL, shared_vocab) for k in keys])

    # ── score matrices ──
    # CRC: cosine of causal effect signatures (shared output space, no alignment)
    crc_S = torch.zeros(K, K)
    for i in range(K):
        for j in range(K):
            crc_S[i, j] = cos(eff_a[i], eff_b[j])
    # geometry: cosine of aligned residual directions
    geo_S = torch.zeros(K, K)
    va_mapped = {k: (va[k] @ W) for k in keys}
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            geo_S[i, j] = cos(va_mapped[ki], vb[kj])

    acc_crc = topk_match_accuracy(crc_S)
    acc_geo = topk_match_accuracy(geo_S)

    # ── transfer effect: use B-mechanism selected by each method, measure ──
    # logit mass moved toward the INTENDED category k's tokens, in model B.
    def transfer(pred_idx):
        vals = []
        for i, ki in enumerate(keys):
            j = int(pred_idx[i])
            eff = steered_logit_delta(mb, vb[keys[j]], Lb, NEUTRAL, cat_ids[ki])
            vals.append(eff.mean().item())     # mean logit change on category-ki tokens
        return float(np.mean(vals))
    crc_pred = crc_S.argmax(1); geo_pred = geo_S.argmax(1)
    oracle_pred = torch.arange(K)
    tr_crc = transfer(crc_pred); tr_geo = transfer(geo_pred); tr_oracle = transfer(oracle_pred)

    # ── seed variance via bootstrap of probe set (recompute effect sigs) ──
    rng = np.random.default_rng(0)
    accs_crc, accs_geo = [], []
    for s in SEEDS:
        rs = np.random.default_rng(s)
        probes = list(rs.choice(NEUTRAL, size=len(NEUTRAL), replace=True))
        ea = torch.stack([steered_logit_delta(ma, va[k], La, probes, shared_vocab) for k in keys])
        eb = torch.stack([steered_logit_delta(mb, vb[k], Lb, probes, shared_vocab) for k in keys])
        S = torch.zeros(K, K)
        for i in range(K):
            for j in range(K):
                S[i, j] = cos(ea[i], eb[j])
        accs_crc.append(topk_match_accuracy(S))
        accs_geo.append(acc_geo)  # geometry doesn't depend on probes
    acc_crc_mean = float(np.mean(accs_crc)); acc_crc_std = float(np.std(accs_crc))

    # ── verdict per preregistration ──
    rand = 1.0 / K
    gap = acc_crc_mean - acc_geo
    if gap >= 0.15 and acc_crc_mean > rand + 1e-9:
        verdict = "SUPPORTED"
    elif gap <= 0.05:
        verdict = "REJECTED (renamed bundle)"
    else:
        verdict = "INCONCLUSIVE"

    result = {
        "models": ["pythia-160m", "pythia-410m"],
        "K_categories": K, "categories": keys,
        "shared_vocab_tokens": len(shared_vocab),
        "steer_layers": {"A": La, "B": Lb}, "steer_norm_c": STEER_C,
        "random_baseline_acc": rand,
        "match_accuracy": {
            "crc_mean": acc_crc_mean, "crc_std": acc_crc_std,
            "crc_point": acc_crc, "geometry": acc_geo,
        },
        "transfer_effect_logit": {"crc": tr_crc, "geometry": tr_geo, "oracle": tr_oracle},
        "crc_minus_geometry": gap,
        "verdict": verdict,
        "crc_predictions": [keys[int(j)] for j in crc_pred],
        "geo_predictions": [keys[int(j)] for j in geo_pred],
        "seconds": round(time.time() - t0, 1),
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log("\n=== RESULT ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    sys.exit(main())
