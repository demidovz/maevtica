#!/usr/bin/env python3
"""IICF on GPT-2-small IOI — see PREREG.md (committed 1e82498 before run)."""
import json, os, sys, time
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
ABC_NAMES = ("Steve","Kevin","Carl")

def build_prompts(m):
    prompts, corr_abc, corr_swap, io_ids, s_ids = [], [], [], [], []
    for A, B in PAIRS:
        for a, b in ((A, B), (B, A)):
            prompts.append(f"When {a} and {b} went to the store, {b} gave a drink to")
            corr_abc.append(f"When {ABC_NAMES[0]} and {ABC_NAMES[1]} went to the store, {ABC_NAMES[2]} gave a drink to")
            corr_swap.append(f"When {a} and {b} went to the store, {a} gave a drink to")
            io_ids.append(int(m.to_tokens(" "+a, prepend_bos=False)[0,0]))
            s_ids.append(int(m.to_tokens(" "+b, prepend_bos=False)[0,0]))
    for w in [w for p in PAIRS for w in p] + list(ABC_NAMES):
        assert m.to_tokens(" "+w, prepend_bos=False).shape[1] == 1, w
    toks = m.to_tokens(prompts); t_abc = m.to_tokens(corr_abc); t_swap = m.to_tokens(corr_swap)
    assert toks.shape == t_abc.shape == t_swap.shape
    return toks, t_abc, t_swap, torch.tensor(io_ids), torch.tensor(s_ids)

# ---------- node bookkeeping ----------
NL, NH = 12, 12
UP = ["embed"] + [f"a{L}.{H}" for L in range(NL) for H in range(NH)] + [f"m{L}" for L in range(NL)]
DN = [f"a{L}.{H}" for L in range(NL) for H in range(NH)] + [f"m{L}" for L in range(NL)] + ["logits"]
UPi = {n:i for i,n in enumerate(UP)}; DNi = {n:i for i,n in enumerate(DN)}
def ustage(n):
    if n=="embed": return 0
    if n[0]=="a": L=int(n[1:].split(".")[0]); return 2*L+1
    return 2*int(n[1:])+2
def vstage(n):
    if n=="logits": return 2*NL+1
    if n[0]=="a": return 2*int(n[1:].split(".")[0])+1
    return 2*int(n[1:])+2
VALID = np.zeros((len(UP), len(DN)), bool)
for i,u in enumerate(UP):
    for j,v in enumerate(DN):
        VALID[i,j] = ustage(u) < vstage(v)
log("valid edges:", int(VALID.sum()))

UP_HOOKS = ["blocks.0.hook_resid_pre"] + [f"blocks.{L}.attn.hook_result" for L in range(NL)] + [f"blocks.{L}.hook_mlp_out" for L in range(NL)]
DN_HOOKS = [f"blocks.{L}.hook_attn_in" for L in range(NL)] + [f"blocks.{L}.hook_mlp_in" for L in range(NL)] + [f"blocks.{NL-1}.hook_resid_post"]

def stack_up(cache):
    """[|UP|, b, p, d] float32 from a dict of hook tensors."""
    parts = [cache["blocks.0.hook_resid_pre"].unsqueeze(0)]
    for L in range(NL):
        parts.append(cache[f"blocks.{L}.attn.hook_result"].permute(2,0,1,3))
    for L in range(NL):
        parts.append(cache[f"blocks.{L}.hook_mlp_out"].unsqueeze(0))
    # order must match UP: embed, all heads (L-major), all mlps
    heads = torch.cat(parts[1:1+NL], 0)  # [NL*NH, b, p, d]
    return torch.cat([parts[0], heads] + parts[1+NL:], 0).float()

def stack_dn(gr):
    parts = []
    for L in range(NL):
        parts.append(gr[f"blocks.{L}.hook_attn_in"].permute(2,0,1,3))
    heads = torch.cat(parts, 0)
    mlps = torch.cat([gr[f"blocks.{L}.hook_mlp_in"].unsqueeze(0) for L in range(NL)], 0)
    return torch.cat([heads, mlps, gr[f"blocks.{NL-1}.hook_resid_post"].unsqueeze(0)], 0).float()

def run_model(m, toks, io_ids, s_ids):
    """Returns clean caches, grads per metric, corrupt up-stacks, clean LD/acc."""
    ar = torch.arange(toks.shape[0])
    def ld_fn(logits):
        last = logits[:, -1].float()
        return last[ar, io_ids] - last[ar, s_ids]
    def lp_fn(logits):
        return logits[:, -1].float().log_softmax(-1)[ar, io_ids]
    res = {}
    for mi, (mname, mfn) in enumerate([("ld", ld_fn), ("lp", lp_fn)]):
        cache, grads = {}, {}
        fwd = [(h, (lambda t, hook: cache.__setitem__(hook.name, t.detach().clone()))) for h in UP_HOOKS]
        bwd = [(h, (lambda g, hook: grads.__setitem__(hook.name, g.detach().clone()))) for h in DN_HOOKS]
        with torch.enable_grad():
            with m.hooks(fwd_hooks=fwd, bwd_hooks=bwd):
                logits = m(toks)
                metric = mfn(logits).mean()
                metric.backward()
        m.zero_grad(set_to_none=True)
        if mi == 0:
            ldv = ld_fn(logits.detach())
            res["clean_ld"] = float(ldv.mean()); res["acc"] = float((ldv>0).float().mean())
            res["U_clean"] = stack_up(cache)
        res[f"G_{mname}"] = stack_dn(grads)
        del logits
    return res

def corrupt_up(m, toks):
    cache = {}
    fwd = [(h, (lambda t, hook: cache.__setitem__(hook.name, t.detach().clone()))) for h in UP_HOOKS]
    with m.hooks(fwd_hooks=fwd):
        m(toks)
    return stack_up(cache)

def eap_scores(U_corr, U_clean, G):
    D = (U_corr - U_clean).reshape(len(UP), -1)
    Gm = G.reshape(len(DN), -1)
    S = (D @ Gm.T).numpy()
    S[~VALID] = 0.0
    return S

def circuits_for(m, toks, t_abc, t_swap, io_ids, s_ids, tag):
    r = run_model(m, toks, io_ids, s_ids)
    log(tag, "clean LD", round(r["clean_ld"],3), "acc", r["acc"])
    U_abc = corrupt_up(m, t_abc); U_swap = corrupt_up(m, t_swap)
    U_mean = r["U_clean"].mean(1, keepdim=True).expand_as(r["U_clean"]).contiguous()
    S = {}
    for cname, Uc in [("abc",U_abc),("swap",U_swap),("mean",U_mean)]:
        for mname in ("ld","lp"):
            S[(cname,mname)] = eap_scores(Uc, r["U_clean"], r[f"G_{mname}"])
    circ = {}
    for (cname,mname), sc in S.items():
        a = np.abs(sc); flat = a.ravel()
        order = np.argsort(-flat)
        for k in (50,100,200):
            idx = order[:k]
            circ[(cname,k,mname)] = set(map(int, idx))
    return r, S, circ, U_abc

def edge_uv(eidx): return eidx // len(DN), eidx % len(DN)

def main():
    log("loading gpt2 (model A)")
    mA = HookedTransformer.from_pretrained("gpt2", device="cpu"); mA.eval()
    mA.set_use_attn_result(True); mA.set_use_attn_in(True); mA.set_use_hook_mlp_in(True)
    toks, t_abc, t_swap, io_ids, s_ids = build_prompts(mA)
    rA, SA, circA, U_abc_A = circuits_for(mA, toks, t_abc, t_swap, io_ids, s_ids, "A")

    allsets = list(circA.values())
    from collections import Counter
    cnt = Counter(e for s in allsets for e in s)
    CORE = sorted(e for e,c in cnt.items() if c == len(allsets))
    UNIQUE = sorted(e for e,c in cnt.items() if c == 1)
    log("|CORE|", len(CORE), "|UNIQUE|", len(UNIQUE), "union", len(cnt))

    # ---- part (a): single-edge ABC corruption patch necessity ----
    ar = torch.arange(toks.shape[0])
    def ld_mean(logits):
        last = logits[:, -1].float()
        return float((last[ar, io_ids] - last[ar, s_ids]).mean())
    DELTA = U_abc_A - rA["U_clean"]  # [|UP|, b, p, d]
    def patch_edge(eidx):
        u, v = edge_uv(eidx)
        d = DELTA[u]
        vn = DN[v]
        if vn == "logits":
            h = f"blocks.{NL-1}.hook_resid_post"
            fn = lambda t, hook: t + d.to(t.dtype)
        elif vn[0] == "a":
            L, H = map(int, vn[1:].split("."))
            h = f"blocks.{L}.hook_attn_in"
            def fn(t, hook, H=H, d=d):
                t[:, :, H, :] = t[:, :, H, :] + d.to(t.dtype); return t
        else:
            h = f"blocks.{int(vn[1:])}.hook_mlp_in"
            fn = lambda t, hook: t + d.to(t.dtype)
        lg = mA.run_with_hooks(toks, fwd_hooks=[(h, fn)])
        return rA["clean_ld"] - ld_mean(lg)

    rng = np.random.default_rng(0)
    core_test = CORE if len(CORE) <= 40 else sorted(rng.choice(CORE, 40, replace=False).tolist())
    uniq_test = sorted(rng.choice(UNIQUE, min(40, len(UNIQUE)), replace=False).tolist())

    # oracle: top-|EAP| edge of (abc, ld) spec
    sc = np.abs(SA[("abc","ld")])
    top_edge = int(np.argmax(sc.ravel()))
    oracle_drop = patch_edge(top_edge)
    u,v = edge_uv(top_edge)
    log("oracle edge", UP[u], "->", DN[v], "drop", round(oracle_drop,3))
    oracle_ok = rA["clean_ld"] > 1.0 and rA["acc"] > 0.9 and abs(oracle_drop) >= 0.3

    drops_core = {e: patch_edge(e) for e in core_test}; log("core drops done")
    drops_uniq = {e: patch_edge(e) for e in uniq_test}; log("uniq drops done")
    md_core = float(np.mean(list(drops_core.values()))) if drops_core else float("nan")
    md_uniq = float(np.mean(list(drops_uniq.values()))) if drops_uniq else float("nan")
    tested = core_test + uniq_test
    def spearman(x, y):
        rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
        return float(np.corrcoef(rx, ry)[0, 1])
    abs_eap = np.abs(SA[("abc","ld")]).ravel()
    rho = spearman(np.array([abs_eap[e] for e in tested]),
                   np.array([abs(drops_core.get(e, drops_uniq.get(e))) for e in tested])) if len(tested) > 2 else None

    sup_a = (len(CORE) > 0) and (md_core >= 2*max(md_uniq, 0.0)) and (md_core >= 0.05)

    # ---- part (b): transfer to differently-seeded model ----
    log("loading model B (alias-gpt2-small-x21)")
    del DELTA, U_abc_A
    b_res = {}
    try:
        mB = HookedTransformer.from_pretrained("stanford-crfm/alias-gpt2-small-x21", device="cpu"); mB.eval()
        mB.set_use_attn_result(True); mB.set_use_attn_in(True); mB.set_use_hook_mlp_in(True)
        toksB, t_abcB, t_swapB, io_idsB, s_idsB = build_prompts(mB)
        rB, SB, circB, _ = circuits_for(mB, toksB, t_abcB, t_swapB, io_idsB, s_idsB, "B")
        b_res["clean_ld"] = rB["clean_ld"]; b_res["acc"] = rB["acc"]
        if rB["acc"] >= 0.75:
            RB = set().union(*circB.values())
            rate = lambda E: (len(set(E) & RB) / len(E)) if E else float("nan")
            rate_core = rate(CORE)
            rates_spec = {str(s): rate(circA[s]) for s in circA}
            sup_b = all(rate_core > r for r in rates_spec.values()) if CORE else False
            b_res.update(dict(computable=True, rate_core=rate_core, rates_spec=rates_spec,
                              max_spec_rate=max(rates_spec.values()), RB_size=len(RB), sup_b=sup_b))
        else:
            b_res.update(dict(computable=False, reason="model B acc < 0.75"))
    except Exception as ex:
        b_res.update(dict(computable=False, reason=f"model B failed: {ex}"))

    # ---- verdict per prereg ----
    if not oracle_ok:
        verdict = "BROKEN_MEASUREMENT"
    elif b_res.get("computable"):
        verdict = "SUPPORTED" if (sup_a and b_res["sup_b"]) else "REFUTED"
    else:
        verdict = "INCONCLUSIVE"

    out = dict(
        prereg_commit="1e82498",
        clean_ld_A=rA["clean_ld"], acc_A=rA["acc"],
        n_core=len(CORE), n_unique=len(UNIQUE), n_union=len(cnt),
        core_edges=[f"{UP[edge_uv(e)[0]]}->{DN[edge_uv(e)[1]]}" for e in CORE][:60],
        oracle_edge=f"{UP[u]}->{DN[v]}", oracle_drop=oracle_drop, oracle_ok=bool(oracle_ok),
        mean_drop_core=md_core, mean_drop_unique=md_uniq,
        median_drop_core=float(np.median(list(drops_core.values()))) if drops_core else None,
        median_drop_unique=float(np.median(list(drops_uniq.values()))) if drops_uniq else None,
        n_core_tested=len(core_test), n_unique_tested=len(uniq_test),
        spearman_absEAP_vs_absdrop=rho,
        sup_a=bool(sup_a), part_b=b_res, verdict=verdict,
        wall_s=round(time.time()-T0,1),
    )
    with open(os.path.join(HERE, "result.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    log(json.dumps({k:v for k,v in out.items() if k not in ("core_edges","part_b")}, indent=2))
    log("part_b:", json.dumps({k:v for k,v in b_res.items() if k!="rates_spec"}, default=str))
    log("VERDICT:", verdict)

if __name__ == "__main__":
    main()
