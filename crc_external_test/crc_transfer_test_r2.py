#!/usr/bin/env python3
"""CRC external test — ROUND 2 (hardened).

Fixes the three round-1 weaknesses:
 (1) DECOUPLED metric: causal-role signature measured over the full vocabulary
     with every category's answer-tokens MASKED OUT — matching must rely on the
     side-effect fingerprint, not on boosting the category's own tokens.
 (2) STRONG geometry baseline: align A/B residual spaces via least-squares on
     PAIRED activations (same prompts fed to both models) — the standard strong
     representational-alignment method — then cosine-match steering directions.
 (3) HARDER discrimination: 12 categories incl. confusable pairs (fruits/veg,
     metals/colors); chance = 1/12 ≈ 0.083.

Efficiency: per-(behavior,probe) full-vocab logit deltas computed ONCE; seeds
are bootstrap over the probe axis in numpy (no extra model runs).

Run: crc-venv311/bin/python crc_transfer_test_r2.py
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
SEEDS = list(range(8))
FRAC_LAYER = 0.5
STEER_C = 0.5

CATEGORIES = {
    "animals":   (["cat","dog","horse","lion","bird","fish","cow","bear","wolf","sheep","goat"],
                  ["The farmer fed the","At the zoo we saw a","My pet is a","In the forest lived a","The vet examined the"]),
    "fruits":    (["apple","banana","orange","grape","lemon","peach","cherry","melon","mango","pear"],
                  ["For dessert she ate an","The basket was full of","He picked a ripe","My favorite fruit is the","On the tree grew a"]),
    "vegetables":(["carrot","potato","onion","tomato","pepper","bean","pea","cabbage","corn","garlic"],
                  ["In the soup she added","The garden grew fresh","He chopped up a","The salad had some","She planted a row of"]),
    "colors":    (["red","blue","green","yellow","black","white","brown","pink","purple","orange"],
                  ["Her favorite color is","The sky slowly turned","He painted the wall","The old car was","My new shirt is"]),
    "metals":    (["gold","iron","silver","copper","steel","tin","lead","bronze","zinc"],
                  ["The ring was made of","The bridge is built from","They mined for","The coin was pure","The sword was forged from"]),
    "numbers":   (["one","two","three","four","five","six","seven","eight","nine","ten"],
                  ["The total came to","She counted up to","He was only","There were exactly","I bought just"]),
    "countries": (["France","Japan","Brazil","Egypt","China","India","Spain","Italy","Canada","Germany"],
                  ["Last summer we flew to","She was born in","He is travelling to","The team came from","My visa is for"]),
    "cities":    (["London","Paris","Tokyo","Berlin","Moscow","Rome","Madrid","Boston","Chicago","Sydney"],
                  ["The conference is held in","She moved to","The train arrives in","He grew up in","We landed in"]),
    "body":      (["hand","foot","head","arm","leg","eye","nose","ear","finger","knee"],
                  ["The doctor examined my","He gently touched her","She injured her left","The glove fit his","She raised her"]),
    "clothing":  (["shirt","coat","hat","shoe","dress","sock","glove","scarf","jacket","belt"],
                  ["In winter she wears a","He bought a new","On the hook hung a","She knitted a warm","He forgot his"]),
    "furniture": (["chair","table","sofa","bed","desk","shelf","lamp","couch","stool","cabinet"],
                  ["In the corner stood a","She sat down on the","He built a wooden","The room had a large","They bought a new"]),
    "weather":   (["rain","snow","wind","storm","fog","sun","cloud","frost","hail","thunder"],
                  ["Tomorrow expect heavy","The forecast calls for","Outside there was","The morning brought","We were caught in the"]),
}
NEUTRAL = ["The","I think that","Yesterday,","She said that","It was a","We went to","Here is the","They found the",
           "After a while,","In the morning","He looked at the","Later that day"]
ALIGN_PROMPTS = NEUTRAL + [
    "The quick brown fox jumps","History shows that","In science we learn","Once upon a time there",
    "The economy grew because","Music can make people","When the sun rises","A good book will",
    "The river flowed past","Children love to play","The machine started to","Under the old bridge",
    "Every year the town","Scientists recently discovered","The ancient castle stood","On a clear night you can",
]

def log(*a): print(*a, file=sys.stderr, flush=True)

def load_model(name):
    from transformer_lens import HookedTransformer
    log(f"[load] {name} …")
    m = HookedTransformer.from_pretrained(name, device=DEVICE); m.eval()
    return m

def single_token_ids(model, words):
    ids = []
    for w in words:
        t = model.to_tokens(" " + w, prepend_bos=False)[0]
        if t.shape[0] == 1:
            ids.append(int(t[0]))
    return ids

def resid_last(model, prompts, layer):
    hook = f"blocks.{layer}.hook_resid_post"
    out = []
    for p in prompts:
        _, c = model.run_with_cache(model.to_tokens(p), names_filter=hook)
        out.append(c[hook][0, -1].float())
    return torch.stack(out)

def steering_vectors(model, cats, layer):
    per = {k: resid_last(model, tmpl, layer) for k, (_, tmpl) in cats.items()}
    grand = torch.cat(list(per.values()), 0).mean(0)
    return {k: per[k].mean(0) - grand for k in cats}

def per_probe_full_delta(model, vecs, layer, probes):
    """Return array [K, n_probes, vocab] of (steered-clean) last-token logits."""
    name = f"blocks.{layer}.hook_resid_post"
    keys = list(vecs.keys())
    clean = []
    toks_list = [model.to_tokens(p) for p in probes]
    for toks in toks_list:
        clean.append(model(toks)[0, -1].float())
    clean = torch.stack(clean)  # [P, V]
    D = np.zeros((len(keys), len(probes), clean.shape[1]), dtype=np.float32)
    for ki, k in enumerate(keys):
        v = vecs[k]
        def fn(resid, hook, v=v):
            resid[:, -1, :] = resid[:, -1, :] + v.to(resid.dtype); return resid
        for pi, toks in enumerate(toks_list):
            steer = model.run_with_hooks(toks, fwd_hooks=[(name, fn)])[0, -1].float()
            D[ki, pi] = (steer - clean[pi]).numpy()
    return keys, D

def align_map(ma, mb, layer_a, layer_b, prompts):
    Xa = resid_last(ma, prompts, layer_a)   # [N, da]
    Xb = resid_last(mb, prompts, layer_b)   # [N, db]
    W = torch.linalg.lstsq(Xa, Xb).solution  # [da, db]
    return W

def cos_rows(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return A @ B.T

def top1(S):
    K = S.shape[0]
    return float((S.argmax(1) == np.arange(K)).mean())

def main():
    t0 = time.time()
    ma, mb = load_model("pythia-160m"), load_model("pythia-410m")
    La, Lb = round(ma.cfg.n_layers*FRAC_LAYER), round(mb.cfg.n_layers*FRAC_LAYER)
    keys = list(CATEGORIES.keys()); K = len(keys)
    cat_ids = {k: single_token_ids(ma, CATEGORIES[k][0]) for k in keys}
    all_cat_ids = sorted({i for ids in cat_ids.values() for i in ids})
    shared_cat = np.array(all_cat_ids)
    log(f"[cfg] K={K} layers A{La}/B{Lb}  masked answer-tokens={len(all_cat_ids)}")

    va, vb = steering_vectors(ma, CATEGORIES, La), steering_vectors(mb, CATEGORIES, Lb)
    na = resid_last(ma, NEUTRAL, La).norm(dim=-1).mean().item()
    nb = resid_last(mb, NEUTRAL, Lb).norm(dim=-1).mean().item()
    for d, n in ((va, na), (vb, nb)):
        for k in d: d[k] = d[k]/d[k].norm()*(STEER_C*n)

    # per-probe full-vocab deltas (once)
    log("[deltas] model A …"); _, Da = per_probe_full_delta(ma, va, La, NEUTRAL)
    log("[deltas] model B …"); _, Db = per_probe_full_delta(mb, vb, Lb, NEUTRAL)

    V = Da.shape[2]
    mask = np.ones(V, dtype=bool); mask[shared_cat] = False   # DECOUPLED: hide answer tokens

    # ── point estimates (mean over probes) ──
    sigA_dec = Da.mean(1)[:, mask]; sigB_dec = Db.mean(1)[:, mask]
    sigA_cat = Da.mean(1)[:, shared_cat]; sigB_cat = Db.mean(1)[:, shared_cat]
    acc_crc_dec = top1(cos_rows(sigA_dec, sigB_dec))     # honest metric
    acc_crc_cat = top1(cos_rows(sigA_cat, sigB_cat))     # round-1-style (home field)

    # strong geometry baseline
    W = align_map(ma, mb, La, Lb, ALIGN_PROMPTS).numpy()
    VA = np.stack([va[k].numpy() for k in keys])         # [K, da]
    VB = np.stack([vb[k].numpy() for k in keys])         # [K, db]
    acc_geo = top1(cos_rows(VA @ W, VB))

    # ── transfer effect (behavioral) via decoupled-CRC vs geometry choices ──
    pred_dec = cos_rows(sigA_dec, sigB_dec).argmax(1)
    pred_geo = cos_rows(VA @ W, VB).argmax(1)
    def transfer(pred):
        vals = []
        for i, ki in enumerate(keys):
            j = int(pred[i])
            eff = Db[j][:, cat_ids[ki]].mean()   # B mech j steers toward intended ki tokens
            vals.append(float(eff))
        return float(np.mean(vals))
    tr_dec, tr_geo, tr_oracle = transfer(pred_dec), transfer(pred_geo), transfer(np.arange(K))

    # ── seeds: bootstrap over probe axis (free) ──
    accs_dec = []
    for s in SEEDS:
        rs = np.random.default_rng(s)
        idx = rs.integers(0, Da.shape[1], Da.shape[1])
        sA = Da[:, idx].mean(1)[:, mask]; sB = Db[:, idx].mean(1)[:, mask]
        accs_dec.append(top1(cos_rows(sA, sB)))
    dec_mean, dec_std = float(np.mean(accs_dec)), float(np.std(accs_dec))

    rand = 1.0/K
    gap = dec_mean - acc_geo
    verdict = ("SUPPORTED" if gap >= 0.15 and dec_mean > rand+1e-9
               else "REJECTED (renamed bundle)" if gap <= 0.05
               else "INCONCLUSIVE")

    res = {
        "round": 2, "models": ["pythia-160m","pythia-410m"], "K": K, "categories": keys,
        "chance": rand, "steer_layers": {"A": La, "B": Lb},
        "match_accuracy": {
            "crc_decoupled_mean": dec_mean, "crc_decoupled_std": dec_std,
            "crc_decoupled_point": acc_crc_dec,
            "crc_categorytokens_point": acc_crc_cat,   # round-1-style, for contrast
            "geometry_strong": acc_geo,
        },
        "transfer_effect_logit": {"crc_decoupled": tr_dec, "geometry": tr_geo, "oracle": tr_oracle},
        "crc_decoupled_minus_geometry": gap,
        "verdict_on_decoupled_metric": verdict,
        "crc_decoupled_predictions": [keys[int(j)] for j in pred_dec],
        "geometry_predictions": [keys[int(j)] for j in pred_geo],
        "seconds": round(time.time()-t0, 1),
    }
    (OUT/"results_r2.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    sys.exit(main())
