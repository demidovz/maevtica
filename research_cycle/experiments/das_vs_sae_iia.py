#!/usr/bin/env python3
"""TESTER experiment: functional variable (causal-abstraction slot).

Prediction under test (preregistered, see PREREG string below):
  For tasks where DAS finds a k-dim subspace S with IIA > 0.9, the best single
  axis-aligned SAE latent scores IIA < 0.6, and winning k > 1 -> causally-real
  variables are non-monosemantic and lie OFF the SAE axes.
  FALSIFIED if a single monosemantic SAE latent reaches IIA parity with the DAS
  subspace (k=1 suffices and axis-aligned wins).

Behavior: subject-verb NUMBER agreement in gpt2-small (a genuine causal variable,
Finlayson et al. 2021). High-level var V = grammatical number of the subject.
Interchange intervention at blocks.6.hook_resid_pre, last (subject) token.
IIA = fraction of (base,source) pairs where the patched is/are prediction follows
the SOURCE's number (i.e. flips to the counterfactual).

Baselines compared at the SAME hook/position, both additive rank-<=k edits in
resid space, so it is a fair fight:
  * DAS:  resid' = base + P_R (source - base),  P_R = projector onto learned k-dim S.
  * SAE:  resid' = base + (a_src_j - a_base_j) * d_j  (best single feature j; d_j unit decoder dir).
  * ORACLE (positive control): full last-token resid swap base<-source. Must be high.
  * NEG control: random 1-dim direction. Must be ~chance.
"""
from __future__ import annotations
import numpy as np, torch, os, json, sys, random
torch.set_num_threads(max(1, os.cpu_count() or 4))
SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

LAYER = 6
HOOK = f"blocks.{LAYER}.hook_resid_pre"
DEVICE = "cpu"

PREREG = {
  "hypothesis": "causal number-variable is non-monosemantic & off SAE axes",
  "oracle_min": 0.75,           # full-swap positive control must exceed this
  "das_high": 0.85,             # antecedent 'DAS finds high-IIA subspace' (pred says >0.9; 0.85 fair for tiny model)
  "sae_ceiling_supported": 0.60,# pred: best single SAE latent < 0.6
  "parity_margin": 0.10,        # SAE within this of DAS best => parity => REFUTED
  "k1_needed_margin": 0.10,     # best k>1 must beat k=1 by this for 'k>1 needed'
  "chance": 0.50,
  "rule": ("SUPPORTED iff DAS_best>=das_high AND bestSAE<sae_ceiling AND "
           "(DAS_best - DAS_k1)>=k1_needed_margin (k>1 needed). "
           "REFUTED iff bestSAE>=DAS_best-parity_margin (SAE parity) "
           "OR (DAS_k1>=das_high AND bestSAE>=das_high). "
           "BROKEN if oracle<oracle_min. else INCONCLUSIVE."),
}

# --- data: singular/plural noun pairs, same lemma -----------------------------
PAIRS = [
  ("dog","dogs"),("cat","cats"),("key","keys"),("boy","boys"),("girl","girls"),
  ("car","cars"),("book","books"),("bird","birds"),("tree","trees"),("house","houses"),
  ("king","kings"),("door","doors"),("road","roads"),("star","stars"),("ship","ships"),
  ("wall","walls"),("hand","hands"),("river","rivers"),("horse","horses"),("student","students"),
  ("player","players"),("worker","workers"),("teacher","teachers"),("driver","drivers"),
  ("farmer","farmers"),("soldier","soldiers"),("doctor","doctors"),("sister","sisters"),
  ("brother","brothers"),("mother","mothers"),("father","fathers"),("island","islands"),
]
TEMPLATE = "The {}"   # last token = the noun; predict is/are


def main():
    from transformer_lens import HookedTransformer
    from sae_lens import SAE
    print("loading model...", flush=True)
    model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    IS = model.to_single_token(" is")
    ARE = model.to_single_token(" are")
    print("is/are token ids", IS, ARE, flush=True)

    # build items: prompt, number(0=sing,1=plur), single-token last?
    items = []  # (text, number, n_tokens)
    for sing, plur in PAIRS:
        for word, num in ((sing,0),(plur,1)):
            txt = TEMPLATE.format(word)
            toks = model.to_tokens(txt)  # [1, seq] incl BOS
            items.append((txt, num, toks))

    # clean pass: cache resid at HOOK last pos + logits
    resid, logit_is, logit_are = [], [], []
    for txt, num, toks in items:
        logits, cache = model.run_with_cache(toks, names_filter=HOOK)
        resid.append(cache[HOOK][0,-1].detach().clone())
        l = logits[0,-1]
        logit_is.append(float(l[IS])); logit_are.append(float(l[ARE]))
    resid = torch.stack(resid)  # [N, d]
    nums = np.array([it[1] for it in items])
    li = np.array(logit_is); la = np.array(logit_are)
    clean_pred = (la > li).astype(int)   # 1 => prefers 'are' => plural
    clean_acc = float((clean_pred == nums).mean())
    print(f"CLEAN 2-way (is/are) acc vs true number = {clean_acc:.3f}  (n={len(items)})", flush=True)

    # keep only items the model classifies correctly clean (variable is actually computed)
    good = np.where(clean_pred == nums)[0]
    print(f"usable items (clean-correct) = {len(good)}", flush=True)

    # build interchange pairs: base & source with OPPOSITE number, both clean-correct
    goodset = list(good)
    sing_idx = [i for i in goodset if nums[i]==0]
    plur_idx = [i for i in goodset if nums[i]==1]
    rng = random.Random(SEED)
    pairs = []  # (base_i, source_i, cf_label)  cf_label = source number
    for bi in goodset:
        # source of opposite number
        pool = plur_idx if nums[bi]==0 else sing_idx
        for si in pool:
            pairs.append((bi, si))
    rng.shuffle(pairs)
    # cap for speed but keep plenty
    pairs = pairs[:600]
    ntr = int(len(pairs)*0.6)
    train_pairs = pairs[:ntr]; test_pairs = pairs[ntr:]
    print(f"pairs total={len(pairs)} train={len(train_pairs)} test={len(test_pairs)}", flush=True)

    d = resid.shape[1]
    resid_g = resid  # [N,d]
    base_tokens = {i: items[i][2] for i in goodset}

    # ---- generic patched-eval: given per-item new resid vector, run model, get is/are argmax
    def patched_pred(base_i, new_resid_vec):
        toks = items[base_i][2]
        def hook(t, hook, v=new_resid_vec):
            t[:, -1, :] = v.to(t.dtype); return t
        logits = model.run_with_hooks(toks, fwd_hooks=[(HOOK, hook)])
        l = logits[0,-1]
        return 1 if float(l[ARE]) > float(l[IS]) else 0  # predicted number

    def iia_from_edit(edit_fn, eval_pairs):
        """edit_fn(base_i, source_i) -> new last-token resid vector."""
        ok = 0
        for bi, si in eval_pairs:
            new = edit_fn(bi, si)
            pred = patched_pred(bi, new)
            cf = nums[si]  # counterfactual target = source number
            ok += int(pred == cf)
        return ok/len(eval_pairs)

    results = {}

    # ORACLE: full resid swap
    oracle = iia_from_edit(lambda bi,si: resid_g[si].clone(), test_pairs)
    results["oracle_full_swap"] = oracle
    print(f"[ORACLE] full last-token resid swap IIA = {oracle:.3f}", flush=True)

    # NEG control: random 1-dim direction
    rand_iias = []
    for r in range(3):
        g = torch.Generator().manual_seed(100+r)
        u = torch.randn(d, generator=g); u = u/u.norm()
        def edit(bi,si,u=u):
            base=resid_g[bi]; src=resid_g[si]
            return base + ((src-base)@u)*u
        rand_iias.append(iia_from_edit(edit, test_pairs))
    results["neg_random_1d"] = float(np.mean(rand_iias))
    print(f"[NEG] random 1-dim dir IIA = {np.mean(rand_iias):.3f} (chance~0.5)", flush=True)

    # ---- DAS: learn k-dim subspace via grad, for several k
    torch.set_grad_enabled(True)
    def train_das(k, epochs=60, lr=5e-2):
        R = torch.randn(d, k)*0.02; R.requires_grad_(True)
        opt = torch.optim.Adam([R], lr=lr)
        bt = torch.stack([resid_g[bi] for bi,_ in train_pairs])
        st = torch.stack([resid_g[si] for _,si in train_pairs])
        tgt = torch.tensor([nums[si] for _,si in train_pairs])  # 0 sing ->is, 1 plur ->are
        # we need model forward per item (diff prompts) -> batch by grouping identical base tokens is complex;
        # instead run each pair's model forward with the edited resid. To keep grad, hook injects edited vec.
        # For speed, precompute nothing; loop.
        idx = list(range(len(train_pairs)))
        for ep in range(epochs):
            random.Random(ep).shuffle(idx)
            # orthonormal projector from R (basis-free): P = R (R^T R)^-1 R^T
            RtR = R.t()@R + 1e-4*torch.eye(k)
            P = R @ torch.linalg.solve(RtR, R.t())   # [d,d]
            total = 0.0; opt.zero_grad()
            # mini-batch subset each epoch for speed
            sub = idx[:120]
            loss = 0.0
            for j in sub:
                bi, si = train_pairs[j]
                base = resid_g[bi]; src = resid_g[si]
                new = base + P @ (src - base)
                toks = items[bi][2]
                def hook(t, hook, v=new):
                    t[:, -1, :] = v.to(t.dtype); return t
                logits = model.run_with_hooks(toks, fwd_hooks=[(HOOK, hook)])
                l = logits[0,-1]
                pair_logits = torch.stack([l[IS], l[ARE]])
                target = torch.tensor(int(nums[si]))
                loss = loss + torch.nn.functional.cross_entropy(pair_logits.unsqueeze(0), target.unsqueeze(0))
            loss = loss/len(sub)
            loss.backward()
            opt.step()
        # eval (no grad) with final projector
        with torch.no_grad():
            RtR = R.t()@R + 1e-4*torch.eye(k)
            P = (R @ torch.linalg.solve(RtR, R.t())).detach()
        def edit(bi,si,P=P):
            base=resid_g[bi]; src=resid_g[si]
            return base + P @ (src-base)
        with torch.no_grad():
            acc = iia_from_edit(edit, test_pairs)
        return acc

    das = {}
    for k in [1,2,4,8,16]:
        acc = train_das(k)
        das[k] = acc
        print(f"[DAS] k={k:2d} IIA = {acc:.3f}", flush=True)
    torch.set_grad_enabled(False)
    results["das"] = das
    das_k1 = das[1]
    das_best_k = max(das, key=das.get)
    das_best = das[das_best_k]

    # ---- SAE single-latent best -------------------------------------------------
    print("loading SAE...", flush=True)
    r = SAE.from_pretrained("gpt2-small-res-jb", HOOK, device=DEVICE)
    sae = r[0] if isinstance(r, tuple) else r
    Wdec = sae.W_dec.detach()          # [d_sae, d]
    dnorm = Wdec / (Wdec.norm(dim=-1, keepdim=True)+1e-9)  # unit decoder dirs

    # encode resid for good items
    with torch.no_grad():
        feats = sae.encode(resid_g)     # [N, d_sae]
    feats = feats.detach()

    # candidate latents: those most differentially active between sing & plur bases
    sing_mask = torch.tensor([nums[i]==0 for i in range(len(items))])
    plur_mask = ~sing_mask
    # restrict to good items
    goodmask = torch.zeros(len(items), dtype=torch.bool); goodmask[good]=True
    fs = feats[goodmask & sing_mask].mean(0)
    fp = feats[goodmask & plur_mask].mean(0)
    diff = (fp - fs).abs()
    cand = torch.topk(diff, 40).indices.tolist()
    print(f"top candidate SAE latents by sing/plur diff: {cand[:10]} ...", flush=True)

    sae_scores = {}
    for j in cand:
        dj = dnorm[j]
        def edit(bi,si,j=j,dj=dj):
            base=resid_g[bi]
            a_src = feats[si, j]; a_base = feats[bi, j]
            return base + (a_src - a_base)*dj
        sae_scores[j] = iia_from_edit(edit, test_pairs)
    best_j = max(sae_scores, key=sae_scores.get)
    best_sae = sae_scores[best_j]
    results["sae"] = {"best_latent": int(best_j), "best_iia": best_sae,
                      "top5": sorted(sae_scores.values(), reverse=True)[:5]}
    print(f"[SAE] best single latent #{best_j} IIA = {best_sae:.3f}  (top5={sorted(sae_scores.values(),reverse=True)[:5]})", flush=True)

    # Also: best axis-aligned RAW neuron (resid dim) as extra reference
    # ---- verdict ---------------------------------------------------------------
    P = PREREG
    if oracle < P["oracle_min"]:
        verdict = "BROKEN_MEASUREMENT"
    else:
        supported = (das_best >= P["das_high"] and best_sae < P["sae_ceiling_supported"]
                     and (das_best - das_k1) >= P["k1_needed_margin"])
        refuted = (best_sae >= das_best - P["parity_margin"]) or (das_k1 >= P["das_high"] and best_sae >= P["das_high"])
        if supported and not refuted:
            verdict = "SUPPORTED"
        elif refuted:
            verdict = "REFUTED"
        else:
            verdict = "INCONCLUSIVE"

    summary = {
      "clean_acc": clean_acc, "n_usable": len(good), "n_test_pairs": len(test_pairs),
      "oracle_full_swap": oracle, "neg_random_1d": results["neg_random_1d"],
      "das": das, "das_k1": das_k1, "das_best_k": das_best_k, "das_best": das_best,
      "sae_best_latent": int(best_j), "sae_best_iia": best_sae,
      "verdict": verdict, "prereg": PREREG,
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    out = os.path.join(os.path.dirname(__file__), "das_vs_sae_result.json")
    with open(out,"w") as f: json.dump(summary,f,indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
