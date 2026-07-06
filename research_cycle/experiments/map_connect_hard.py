#!/usr/bin/env python3
"""'Добивай' — connect the bricks in a CONFIDENTLY-WRONG regime. Same test as
map_reflect_connect but with RARER / trickier single-token members, where gpt2 is
more likely to mis-sort *confidently*. Adds a regime diagnostic: how confident is the
voice ON ITS ERRORS vs on correct? If the voice is ~as peaked on errors (blind), that
is the regime where the MAP could win. Preregistered rule identical to the easy run.

RULE (frozen): error = argmax category logit != true category.
  MAP signal = entropy(softmax(proj of resid onto 6 LOO category dirs)).
  OUTPUT signal = entropy(softmax(6 category logits)). random; oracle=true label.
  * BROKEN if errors<8 or correct<8 or oracle AUC<0.99.
  * SUPPORTED iff MAP_AUC - OUTPUT_AUC >= 0.05 AND paired-bootstrap 95% CI excludes 0
    AND MAP_AUC > 0.55.
  * REFUTED otherwise.
Regime is REPORTED (voice entropy on err vs ok) but does NOT gate the verdict.
"""
import os, json, numpy as np, torch
torch.set_num_threads(max(1,os.cpu_count() or 4))
from transformer_lens import HookedTransformer
from sklearn.metrics import roc_auc_score
m=HookedTransformer.from_pretrained("gpt2",device="cpu"); m.eval()
rng=np.random.default_rng(0); L=9; FRAME="A {} is a type of"

# rarer / harder members (clear true category, but less common -> weaker/wronger priors)
CATS={
 "fruit":["fig","date","lime","quince","guava","papaya","lychee","apricot","damson","medlar","persimmon","mulberry"],
 "vegetable":["leek","kale","beet","chard","cress","turnip","swede","parsnip","marrow","endive","chicory","rutabaga"],
 "bird":["wren","finch","teal","grebe","snipe","plover","curlew","tern","gull","coot","rail","crane","egret","heron","ibis","godwit"],
 "fish":["eel","ray","pike","sole","carp","perch","tench","bream","roach","chub","dace","minnow","smelt","shad","bass","trout"],
 "tree":["larch","rowan","hazel","alder","aspen","yew","holly","fir","hornbeam","sycamore","chestnut","walnut"],
 "flower":["phlox","zinnia","aster","lotus","peony","dahlia","pansy","crocus","petunia","marigold","daffodil","primrose"],
}
CATW=list(CATS.keys())
def tid(w):
    t=m.to_tokens(" "+w,prepend_bos=False); return int(t[0,0]) if t.shape[1]==1 else None
CATTOK={c:tid(c) for c in CATW}; assert all(v is not None for v in CATTOK.values())
# members need NOT be single-token: we read the residual at the LAST token of
# "A {member} is a type of" and the category logits at single-token category words.
for c in CATS: CATS[c]=sorted(set(CATS[c]))
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
        if drop and drop[0]==c: A=A[[i for i in range(A.shape[0]) if i!=drop[1]]]
        means[c]=A.mean(0)
    return {c:(means[c]-torch.stack([means[o] for o in CATW if o!=c]).mean(0))/(means[c]-torch.stack([means[o] for o in CATW if o!=c]).mean(0)).norm() for c in CATW}
def ent(v):
    p=np.exp(v-v.max()); p=p/p.sum(); return float(-(p*np.log(p+1e-12)).sum())

err=[]; map_sig=[]; out_sig=[]
for c in CATW:
    for i,w in enumerate(CATS[c]):
        lg=cat_logits(w); pred=CATW[int(np.argmax(lg))]
        err.append(1 if pred!=c else 0); out_sig.append(ent(lg))
        loo=dirs_from(drop=(c,i)); proj=np.array([float(ACT[c][i]@loo[cc]) for cc in CATW]); map_sig.append(ent(proj))
err=np.array(err); map_sig=np.array(map_sig); out_sig=np.array(out_sig); rand_sig=rng.random(len(err))
N=len(err); n_err=int(err.sum())
# regime diagnostic: voice entropy on errors vs on correct (blind voice => similar)
voice_err=float(out_sig[err==1].mean()) if n_err else float('nan')
voice_ok=float(out_sig[err==0].mean()) if n_err<N else float('nan')
print(f"[data] N={N} · miscategorized={n_err} ({n_err/N:.1%})",flush=True)
print(f"[regime] voice entropy on ERRORS={voice_err:.2f} vs on CORRECT={voice_ok:.2f} (close => voice blind => confidently-wrong regime)",flush=True)

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
if n_err<8 or (N-n_err)<8 or oracle_auc<0.99: verdict="BROKEN_MEASUREMENT"
elif delta>=0.05 and ci[0]>0 and map_auc>0.55: verdict="SUPPORTED"
else: verdict="REFUTED"
out=dict(model="gpt2 (hard/rare members)",layer=L,N=N,n_miscategorized=n_err,err_rate=n_err/N,
         voice_entropy_on_errors=voice_err,voice_entropy_on_correct=voice_ok,
         MAP_AUC=map_auc,OUTPUT_AUC=out_auc,random_AUC=rand_auc,oracle_AUC=oracle_auc,
         map_minus_output_AUC=delta,delta_CI=ci,verdict=verdict)
print("\n=== NUMBERS ==="); print(json.dumps(out,indent=1))
print("\n=== VERDICT ===",verdict,f"| MAP-OUTPUT={delta:+.3f} CI[{ci[0]:+.3f},{ci[1]:+.3f}] map={map_auc:.3f} out={out_auc:.3f}")
dst=os.path.join(os.path.dirname(__file__),"..","campaigns","map-tree","connect_hard_result.json")
json.dump(out,open(dst,"w"),indent=1); print("[saved]",os.path.abspath(dst),flush=True)
