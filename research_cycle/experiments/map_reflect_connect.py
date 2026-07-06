#!/usr/bin/env python3
"""Connect Brick 2 (the concept TREE) to Brick 1 (routing reflection to errors):
use the MAP as the where-to-reflect signal. The model sorts words into category
branches but mis-sorts ~1/3 (its own concept-mistakes). Question: does a MAP signal
-- "this word's activation is SMEARED across branches rather than cleanly on one" --
catch those mistakes better than the model's OUTPUT confidence (its category logits)?
Both are read from the SAME forward state: map = geometry onto our learned category
axes (leave-one-out, no peeking), output = the native category logits it produces.

PREREGISTERED RULE (frozen before running):
  For each entity: error = argmax category logit != true category.
  MAP signal   = entropy of softmax(projections of resid onto the 6 LOO category dirs).
  OUTPUT signal= entropy of softmax(the 6 category logits).  (both: high => likely wrong)
  random = noise; oracle = true error label (AUC 1.0 by construction, sanity only).
  * BROKEN if errors < 6 or correct < 6 (degenerate) or oracle AUC < 0.99.
  * SUPPORTED iff MAP_AUC - OUTPUT_AUC >= 0.05 AND paired-bootstrap 95% CI excludes 0
    AND MAP_AUC > 0.55.  (output being weak is fine -- that's the pro-hypothesis case.)
  * REFUTED otherwise.
"""
import os, json, numpy as np, torch
torch.set_num_threads(max(1,os.cpu_count() or 4))
from transformer_lens import HookedTransformer
from sklearn.metrics import roc_auc_score
m=HookedTransformer.from_pretrained("gpt2",device="cpu"); m.eval()
rng=np.random.default_rng(0); L=9; FRAME="A {} is a type of"

CATS={
 "fruit":["apple","banana","orange","grape","pear","peach","lemon","cherry","melon","plum","kiwi","lime","fig","mango","berry"],
 "vegetable":["carrot","potato","onion","celery","pepper","cabbage","radish","spinach","tomato","corn","bean","pea","beet","leek","kale"],
 "bird":["eagle","owl","duck","goose","hawk","crow","robin","sparrow","pigeon","swan","hen","dove","finch","wren","quail"],
 "fish":["salmon","tuna","shark","trout","cod","bass","carp","perch","eel","ray","pike","sole","herring","catfish"],
 "tree":["oak","pine","maple","birch","cedar","elm","willow","spruce","ash","fir","teak","aspen","poplar","beech"],
 "flower":["rose","tulip","daisy","lily","poppy","iris","orchid","violet","aster","lotus","peony","dahlia","tulip"],
}
CATW=list(CATS.keys())
def tid(w):
    t=m.to_tokens(" "+w,prepend_bos=False); return int(t[0,0]) if t.shape[1]==1 else None
CATTOK={c:tid(c) for c in CATW}; assert all(v is not None for v in CATTOK.values())
for c in CATS: CATS[c]=sorted(set(w for w in CATS[c] if tid(w) is not None))
print("members per cat:",{c:len(CATS[c]) for c in CATW},flush=True)

@torch.no_grad()
def resid(words):
    h=f"blocks.{L}.hook_resid_post"; out=[]
    for w in words:
        _,c=m.run_with_cache(m.to_tokens(FRAME.format(w)),names_filter=h); out.append(c[h][0,-1].float())
    return torch.stack(out)
@torch.no_grad()
def cat_logits(word):
    lg=m(m.to_tokens(f"A {word} is a type of"))[0,-1].float()
    return np.array([float(lg[CATTOK[c]]) for c in CATW])

ACT={c:resid(CATS[c]) for c in CATW}
def dirs_from(drop=None):
    means={}
    for c in CATW:
        A=ACT[c]
        if drop and drop[0]==c:
            A=A[[i for i in range(A.shape[0]) if i!=drop[1]]]
        means[c]=A.mean(0)
    dd={}
    for c in CATW:
        d=means[c]-torch.stack([means[o] for o in CATW if o!=c]).mean(0); dd[c]=d/d.norm()
    return dd
def ent(v):
    p=np.exp(v-v.max()); p=p/p.sum(); return float(-(p*np.log(p+1e-12)).sum())

err=[]; map_sig=[]; out_sig=[]
for c in CATW:
    for i,w in enumerate(CATS[c]):
        lg=cat_logits(w)
        pred=CATW[int(np.argmax(lg))]
        err.append(1 if pred!=c else 0)
        out_sig.append(ent(lg))                                  # output confidence (native logits)
        loo=dirs_from(drop=(c,i))
        proj=np.array([float(ACT[c][i]@loo[cc]) for cc in CATW])
        map_sig.append(ent(proj))                                # MAP geometry signal
err=np.array(err); map_sig=np.array(map_sig); out_sig=np.array(out_sig)
rand_sig=rng.random(len(err))
N=len(err); n_err=int(err.sum())
print(f"[data] N={N} · miscategorized={n_err} ({n_err/N:.1%})",flush=True)

def auc(s):
    try: return float(roc_auc_score(err,s))
    except Exception: return float('nan')
map_auc=auc(map_sig); out_auc=auc(out_sig); rand_auc=auc(rand_sig); oracle_auc=auc(err.astype(float))
print(f"[detect error] MAP AUC={map_auc:.3f} · OUTPUT AUC={out_auc:.3f} · random={rand_auc:.3f} · oracle={oracle_auc:.3f}",flush=True)

idx=np.arange(N); diffs=[]
for _ in range(2000):
    b=rng.choice(idx,N,replace=True); e=err[b]
    if e.sum()<2 or (len(e)-e.sum())<2: continue
    try: diffs.append(roc_auc_score(e,map_sig[b])-roc_auc_score(e,out_sig[b]))
    except Exception: pass
ci=(float(np.percentile(diffs,2.5)),float(np.percentile(diffs,97.5))) if diffs else (float('nan'),)*2
delta=map_auc-out_auc

if n_err<6 or (N-n_err)<6 or oracle_auc<0.99: verdict="BROKEN_MEASUREMENT"
elif delta>=0.05 and ci[0]>0 and map_auc>0.55: verdict="SUPPORTED"
else: verdict="REFUTED"

out=dict(model="gpt2",layer=L,N=N,n_miscategorized=n_err,err_rate=n_err/N,
         MAP_AUC=map_auc,OUTPUT_AUC=out_auc,random_AUC=rand_auc,oracle_AUC=oracle_auc,
         map_minus_output_AUC=delta,delta_CI=ci,verdict=verdict)
print("\n=== NUMBERS ==="); print(json.dumps(out,indent=1))
print("\n=== VERDICT ===",verdict,f"| MAP-OUTPUT={delta:+.3f} CI[{ci[0]:+.3f},{ci[1]:+.3f}] map={map_auc:.3f} out={out_auc:.3f}")
dst=os.path.join(os.path.dirname(__file__),"..","campaigns","map-tree","connect_result.json")
json.dump(out,open(dst,"w"),indent=1); print("[saved]",os.path.abspath(dst),flush=True)
