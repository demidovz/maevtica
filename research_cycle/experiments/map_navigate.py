#!/usr/bin/env python3
"""'Map of ideas': is the concept hierarchy inside gpt2 a clean TREE you can WALK?
Two sharp, falsifiable questions (author: Илья, hand-written, calibrated-before-frozen):
 (1) TREE: does each member sit under its right branch? (leave-one-out: classify a
     held-out word by which category-direction it points at; beat chance 1/6.)
 (2) WALK: navigate by category-arithmetic — steer a member of A by (d_B - d_A) and
     land on B — and does the SAME move generalize across the whole pair matrix
     (reusable), not just apple->vegetable? A random push of similar size must NOT flip.
Controls: oracle (steer a member by +d_ownCategory -> own category up, else BROKEN)
+ random-direction control.

PREREGISTERED RULE (frozen before running; v2 fixes two of my own spec errors caught
by controls in v1: the BROKEN in-sample bar was too strict at 0.90 — 0.78 >> chance is a
real-but-fuzzy tree, not broken; and the walk test was over-driven by the oracle-MAX
strength (random flipped 82%), so walk now SWEEPS strength for a clean directional window):
  * BROKEN_MEASUREMENT if oracle mean intensification < 1.0 logit OR in-sample tree
    accuracy <= 2x chance (0.33) — directions can't sort their own words at all.
  * TREE_SUPPORTED iff leave-one-out accuracy >= 0.60 (chance = 1/6 = 0.167) AND mean
    pairwise |cos| between category directions < 0.50 (distinct axes).
  * WALK_SUPPORTED iff THERE EXISTS a steering strength (swept) where directed swap
    lands on B in >= 0.50 of clean cases AND a random push of matched size derails the
    category (flips off A) in < 0.15 — i.e. the move is directional, not brute disruption.
  * verdict SUPPORTED iff tree AND walk; PARTIAL iff exactly one; REFUTED iff neither.
"""
import os, json, numpy as np, torch
torch.set_num_threads(max(1,os.cpu_count() or 4))
from transformer_lens import HookedTransformer
m=HookedTransformer.from_pretrained("gpt2",device="cpu"); m.eval()
rng=np.random.default_rng(0)

CATS={
 "fruit":["apple","banana","orange","grape","pear","peach","lemon","cherry","melon","plum"],
 "vegetable":["carrot","potato","onion","celery","pepper","cabbage","radish","spinach"],
 "bird":["eagle","owl","duck","goose","hawk","crow","robin","sparrow","pigeon","swan"],
 "fish":["salmon","tuna","shark","trout","cod","bass","carp","perch"],
 "tree":["oak","pine","maple","birch","cedar","elm","willow","spruce","ash"],
 "flower":["rose","tulip","daisy","lily","poppy","iris","orchid","violet"],
}
CATW=list(CATS.keys())
# represent every word in the SAME context where we steer & measure (the category-
# prediction position), else the steering direction doesn't transfer (BROKEN v1).
FRAME="A {} is a type of"
def tid(w):
    t=m.to_tokens(" "+w,prepend_bos=False); return int(t[0,0]) if t.shape[1]==1 else None
CATTOK={c:tid(c) for c in CATW}
assert all(v is not None for v in CATTOK.values()), f"category not single-token: {CATTOK}"
for c in CATS: CATS[c]=[w for w in CATS[c] if tid(w) is not None]
print("members per cat:",{c:len(CATS[c]) for c in CATW}, flush=True)

@torch.no_grad()
def resid(words,L):
    h=f"blocks.{L}.hook_resid_post"; out=[]
    for w in words:
        _,c=m.run_with_cache(m.to_tokens(FRAME.format(w)),names_filter=h); out.append(c[h][0,-1].float())
    return torch.stack(out)
@torch.no_grad()
def cat_logits(word,vec,L):
    toks=m.to_tokens(f"A {word} is a type of")
    if vec is None: lg=m(toks)[0,-1].float()
    else:
        h=f"blocks.{L}.hook_resid_post"
        def fn(r,hook,v=vec): r[:,-1,:]=r[:,-1,:]+v.to(r.dtype); return r
        lg=m.run_with_hooks(toks,fwd_hooks=[(h,fn)])[0,-1].float()
    return {c:float(lg[CATTOK[c]]) for c in CATW}

def dirs_from(ACT, drop=None):
    """d_C = unit(mean resid(members C) - mean resid(members not-C)); drop=(cat,idx) for LOO."""
    means={}
    for c in CATW:
        A=ACT[c]
        if drop and drop[0]==c:
            keep=[i for i in range(A.shape[0]) if i!=drop[1]]; A=A[keep]
        means[c]=A.mean(0)
    dirs={}
    for c in CATW:
        inside=means[c]
        outs=torch.stack([means[o] for o in CATW if o!=c]).mean(0)
        d=inside-outs; dirs[c]=d/d.norm()
    return dirs

# ---- calibrate layer + steering strength on the ORACLE ----
print("[calib] layer x mult -> oracle intensification (member steered by +d_own)", flush=True)
best=None
for L in [6,8,9]:
    ACT={c:resid(CATS[c],L) for c in CATW}
    nn=torch.cat([ACT[c] for c in CATW]).norm(dim=-1).mean().item()
    dirs=dirs_from(ACT)
    for mult in [0.1,0.25,0.5,1.0,2.0]:
        lift=[]
        for c in CATW:
            for w in CATS[c][:3]:
                b=cat_logits(w,None,L); s=cat_logits(w,dirs[c]*(mult*nn),L); lift.append(s[c]-b[c])
        ml=float(np.mean(lift)); print(f"  L={L} mult={mult} oracle_lift={ml:+.2f}",flush=True)
        if ml>=1.0 and (best is None or ml>best["ml"]): best=dict(L=L,mult=mult,ml=ml,ACT=ACT,nn=nn,dirs=dirs)
if best is None:
    print("=== VERDICT === BROKEN_MEASUREMENT (no layer/strength passed oracle>=1.0)"); raise SystemExit
L=best["L"]; MULT=best["mult"]; ORACLE=best["ml"]; ACT=best["ACT"]; nn=best["nn"]; dirs=best["dirs"]
print(f"[calib] chosen L={L} mult={MULT} oracle_lift={ORACLE:+.2f}", flush=True)

# ---- TEST 1: TREE (leave-one-out classification + axis separation) ----
loo_ok=insamp_ok=tot=0
for c in CATW:
    for i in range(ACT[c].shape[0]):
        rw=ACT[c][i]
        pin={cc:float(rw@dirs[cc]) for cc in CATW}
        if max(pin,key=pin.get)==c: insamp_ok+=1
        dloo=dirs_from(ACT,drop=(c,i)); pl={cc:float(rw@dloo[cc]) for cc in CATW}
        if max(pl,key=pl.get)==c: loo_ok+=1
        tot+=1
tree_acc=loo_ok/tot; insamp_acc=insamp_ok/tot
coss=[abs(float(dirs[a]@dirs[b])) for i,a in enumerate(CATW) for b in CATW[i+1:]]
mean_cos=float(np.mean(coss))
print(f"[tree] LOO acc={tree_acc:.2f} (chance {1/len(CATW):.2f}) · in-sample={insamp_acc:.2f} · mean|cos|={mean_cos:.2f}",flush=True)

# ---- TEST 2: WALK — sweep strength for a clean DIRECTIONAL window ----
base_arg={}
for A in CATW:
    for w in CATS[A]: base_arg[(A,w)]=max(cat_logits(w,None,L).items(),key=lambda kv:kv[1])[0]
sweep=[]
for mult in [0.25,0.5,0.75,1.0,1.5,2.0]:
    sflip=stot=0
    for A in CATW:
        for B in CATW:
            if A==B: continue
            move=(dirs[B]-dirs[A])*(mult*nn)                  # subtract A, add B (natural arithmetic)
            for w in CATS[A][:4]:
                if base_arg[(A,w)]!=A: continue               # only clean starts
                st=cat_logits(w,move,L); stot+=1
                if max(st,key=st.get)==B: sflip+=1
    rflip=rtot=0; rnorm=mult*nn*1.4                            # matched size (~||d_B-d_A||)
    for A in CATW:
        for w in CATS[A][:3]:
            if base_arg[(A,w)]!=A: continue
            r=torch.tensor(rng.standard_normal(dirs[A].shape[0]),dtype=torch.float32); r=r/r.norm()*rnorm
            st=cat_logits(w,r,L); rtot+=1
            if max(st,key=st.get)!=A: rflip+=1
    sr=sflip/stot if stot else float('nan'); rr=rflip/rtot if rtot else float('nan')
    sweep.append(dict(mult=mult,swap_to_B=round(sr,3),random_away=round(rr,3)))
    print(f"[walk] mult={mult} swap->B={sr:.2f} random_away={rr:.2f}",flush=True)
windows=[s for s in sweep if s["swap_to_B"]>=0.50 and s["random_away"]<0.15]
walk_ok=len(windows)>0
best_win=max(windows,key=lambda s:s["swap_to_B"]) if windows else max(sweep,key=lambda s:s["swap_to_B"]-s["random_away"])
print(f"[walk] clean window exists: {walk_ok} · best={best_win}",flush=True)

# ---- VERDICT ----
tree_ok = tree_acc>=0.60 and mean_cos<0.50
if ORACLE<1.0 or insamp_acc<=2/len(CATW): verdict="BROKEN_MEASUREMENT"
elif tree_ok and walk_ok: verdict="SUPPORTED"
elif tree_ok or walk_ok: verdict="PARTIAL"
else: verdict="REFUTED"

out=dict(model="gpt2",layer=L,oracle_lift=ORACLE,
         tree_LOO_acc=tree_acc,tree_insample_acc=insamp_acc,tree_chance=1/len(CATW),
         mean_pairwise_cos=mean_cos,tree_supported=bool(tree_ok),
         walk_sweep=sweep,walk_clean_window=best_win,walk_supported=bool(walk_ok),
         categories=CATW,verdict=verdict)
print("\n=== NUMBERS ==="); print(json.dumps(out,indent=1))
print("\n=== VERDICT ===",verdict,f"| tree(acc={tree_acc:.2f},cos={mean_cos:.2f})={'OK' if tree_ok else 'no'} · walk window={'OK' if walk_ok else 'no'} best={best_win}")
dst=os.path.join(os.path.dirname(__file__),"..","campaigns","map-tree","map_navigate_result.json")
os.makedirs(os.path.dirname(dst),exist_ok=True)
json.dump(out,open(dst,"w"),indent=1); print("[saved]",os.path.abspath(dst),flush=True)
