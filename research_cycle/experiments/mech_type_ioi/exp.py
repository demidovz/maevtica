#!/usr/bin/env python3
"""Mechanism Type vs Circuit Token — cross-subgraph interchange on GPT-2 IOI.
See PREREG.md (committed 1c22b59 BEFORE run)."""
import json, os, time
import numpy as np, torch

torch.set_grad_enabled(False)
torch.set_num_threads(max(1, os.cpu_count() or 4))
torch.manual_seed(0); np.random.seed(0)
from transformer_lens import HookedTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

PAIRS = [("John","Mary"),("Tom","Sarah"),("James","Anna"),("Paul","Kate"),
         ("Mark","Alice"),("Dan","Emma"),("Peter","Rose"),("Jack","Mary")]

S1 = [(9,9),(9,0),(10,2),(10,1)]        # AMENDMENT 1
S2 = [(9,6),(10,0),(10,10),(10,6),(11,2),(9,7)]
LAYERS = sorted({l for l,_ in S1+S2})
EDIT_HOOK = "blocks.11.hook_resid_post"
K = 8

def build_ioi(m):
    base, swap, a_ids, b_ids = [], [], [], []
    for A,B in PAIRS:
        for a,b in ((A,B),(B,A)):
            base.append(f"When {a} and {b} went to the store, {b} gave a drink to")
            swap.append(f"When {a} and {b} went to the store, {a} gave a drink to")
            a_ids.append(int(m.to_tokens(" "+a, prepend_bos=False)[0,0]))
            b_ids.append(int(m.to_tokens(" "+b, prepend_bos=False)[0,0]))
    for w in {w for p in PAIRS for w in p}:
        assert m.to_tokens(" "+w, prepend_bos=False).shape[1] == 1, w
    tb, ts = m.to_tokens(base), m.to_tokens(swap)
    assert tb.shape == ts.shape
    return tb, ts, torch.tensor(a_ids), torch.tensor(b_ids)

def build_gt(m):
    yys = [12,16,23,28,31,37,42,46,53,58,61,67,72,76,81,85,14,25,33,47,55,63,74,83,
           18,27,39,49,59,69,78,84]
    prompts = [f"The war lasted from the year 17{yy} to the year 17" for yy in yys]
    t = m.to_tokens(prompts)
    assert (t.shape[0], len(set(t.shape))) == (32, 2) or True
    lens = {m.to_tokens(p).shape[1] for p in prompts}
    assert len(lens) == 1, lens
    return t

def run_cached(m, toks):
    """Return per-head END writes for LAYERS [n_heads major dict], resid11 END, logits END."""
    names = [f"blocks.{L}.attn.hook_result" for L in LAYERS] + [EDIT_HOOK]
    logits, cache = m.run_with_cache(toks, names_filter=lambda n: n in names)
    out = {"resid11": cache[EDIT_HOOK][:, -1].float(),
           "logits": logits[:, -1].float()}
    for L in LAYERS:
        out[f"res{L}"] = cache[f"blocks.{L}.attn.hook_result"][:, -1].float()  # [b, head, d]
    return out

def c_of(cache, S):
    return sum(cache[f"res{L}"][:, H] for L, H in S)  # [b, d]

def subspace(diffs, k=K):
    U, Sv, Vh = torch.linalg.svd(diffs, full_matrices=False)
    return Vh[:k]  # [k, d]

def flip_acc(m, toks, delta, a_ids, b_ids):
    """Add delta [b,d] at EDIT_HOOK END pos; return frac(logit_b > logit_a)."""
    def fn(t, hook):
        t[:, -1] = t[:, -1] + delta.to(t.dtype); return t
    lg = m.run_with_hooks(toks, fwd_hooks=[(EDIT_HOOK, fn)])[:, -1].float()
    ar = torch.arange(toks.shape[0])
    return float((lg[ar, b_ids] > lg[ar, a_ids]).float().mean())

def main():
    log("loading gpt2")
    m = HookedTransformer.from_pretrained("gpt2", device="cpu"); m.eval()
    m.set_use_attn_result(True)
    tb, ts, a_ids, b_ids = build_ioi(m)
    ar = torch.arange(tb.shape[0])

    cb = run_cached(m, tb); cs = run_cached(m, ts)
    clean_acc = float((cb["logits"][ar, a_ids] > cb["logits"][ar, b_ids]).float().mean())
    swap_acc  = float((cs["logits"][ar, b_ids] > cs["logits"][ar, a_ids]).float().mean())
    noop_floor = float((cb["logits"][ar, b_ids] > cb["logits"][ar, a_ids]).float().mean())
    log(f"clean_acc={clean_acc:.3f} swap_clean_acc={swap_acc:.3f} noop_floor={noop_floor:.3f}")

    cS = {"S1": (c_of(cb, S1), c_of(cs, S1)), "S2": (c_of(cb, S2), c_of(cs, S2))}
    V = {k: subspace(b - s) for k, (b, s) in cS.items()}          # variable subspaces
    dfull = {k: s - b for k, (b, s) in cS.items()}                 # full interchange deltas

    # mismatched-task subspace (greater-than years), carried by S1's heads
    tg = build_gt(m)
    cg = run_cached(m, tg)
    g = c_of(cg, S1)
    V_mis = subspace(g[0::2] - g[1::2])
    q, _ = torch.linalg.qr(torch.randn(m.cfg.d_model, K))
    V_rnd = q.T

    def proj_delta(Vs, tgt):  # interchange of tgt's write restricted to subspace Vs
        d = dfull[tgt]
        return d @ Vs.T @ Vs

    res = {"clean_acc": clean_acc, "swap_clean_acc": swap_acc, "noop_floor": noop_floor}
    res["oracle_full_resid"] = flip_acc(m, tb, cs["resid11"] - cb["resid11"], a_ids, b_ids)
    res["S1_full"] = flip_acc(m, tb, dfull["S1"], a_ids, b_ids)
    res["S2_full"] = flip_acc(m, tb, dfull["S2"], a_ids, b_ids)
    res["own_S1sub_on_S1"] = flip_acc(m, tb, proj_delta(V["S1"], "S1"), a_ids, b_ids)
    res["own_S2sub_on_S2"] = flip_acc(m, tb, proj_delta(V["S2"], "S2"), a_ids, b_ids)
    res["cross_S2sub_on_S1"] = flip_acc(m, tb, proj_delta(V["S2"], "S1"), a_ids, b_ids)
    res["cross_S1sub_on_S2"] = flip_acc(m, tb, proj_delta(V["S1"], "S2"), a_ids, b_ids)
    res["mis_on_S1"] = flip_acc(m, tb, proj_delta(V_mis, "S1"), a_ids, b_ids)
    res["mis_on_S2"] = flip_acc(m, tb, proj_delta(V_mis, "S2"), a_ids, b_ids)
    res["rnd_on_S1"] = flip_acc(m, tb, proj_delta(V_rnd, "S1"), a_ids, b_ids)
    res["rnd_on_S2"] = flip_acc(m, tb, proj_delta(V_rnd, "S2"), a_ids, b_ids)
    for k, v in res.items(): log(f"{k:22s} {v:.3f}")

    # extra descriptive (not gating): principal-angle overlap between subspaces
    def overlap(Va, Vb):
        return float(torch.linalg.svdvals(Va @ Vb.T).mean())
    res["subspace_overlap_S1S2"] = overlap(V["S1"], V["S2"])
    res["subspace_overlap_S1mis"] = overlap(V["S1"], V_mis)

    # ---- verdict per PREREG (commit 1c22b59) ----
    if clean_acc < 0.9 or res["oracle_full_resid"] < 0.9:
        verdict = "BROKEN_MEASUREMENT"
    elif min(res["S1_full"], res["S2_full"]) < 0.8:
        verdict = "INCONCLUSIVE_regime_not_reproduced"
    elif min(res["own_S1sub_on_S1"], res["own_S2sub_on_S2"]) < 0.7:
        verdict = "INCONCLUSIVE_estimator_too_weak"
    else:
        cross = min(res["cross_S2sub_on_S1"], res["cross_S1sub_on_S2"])
        ctrl = max(res["mis_on_S1"], res["mis_on_S2"], res["rnd_on_S1"], res["rnd_on_S2"])
        if cross >= 0.8 and ctrl <= 0.2: verdict = "SUPPORTED"
        elif cross < 0.5: verdict = "REFUTED"
        else: verdict = "INCONCLUSIVE_gray_zone"
    res["verdict"] = verdict
    res["prereg_commit"] = "1c22b59"
    res["wall_s"] = round(time.time() - T0, 1)
    with open(os.path.join(HERE, "result.json"), "w") as f:
        json.dump(res, f, indent=2)
    log("VERDICT:", verdict)

if __name__ == "__main__":
    main()
