#!/usr/bin/env python3
"""Warmup for the 'hierarchy beyond lying' line: do FEELING/EVALUATION words sit as clean,
distinct internal DIRECTIONS the way concrete-noun categories do (Brick 2 machinery)?
Honest scope: this is CONCEPT-reading (does 'shame' have a direction like 'fruit'), NOT the
model feeling anything. If emotions form a clean navigable sub-tree, the main experiment
('says-against-its-own-grain' conflict signal) has a substrate to work with.

Method (same as map-navigate TREE test): frame -> residual at last token -> diff-in-means
category directions -> leave-one-out member->category classification + mean pairwise |cos|.
Run BOTH domains through identical code (each with its natural frame) for a reference compare.
CANARY: shuffle the category labels, rebuild -> LOO must collapse to ~chance (else the method
would 'find structure' in noise and nothing here is trustworthy).
"""
import os, json, numpy as np, torch
torch.set_num_threads(max(1,os.cpu_count() or 4))
from transformer_lens import HookedTransformer
m=HookedTransformer.from_pretrained("gpt2",device="cpu"); m.eval()
LAYERS=[6,8,9,10]

NOUNS=dict(
 fruit=["apple","banana","orange","grape","pear","peach","lemon","cherry"],
 vegetable=["carrot","potato","onion","celery","pepper","cabbage","radish"],
 bird=["eagle","owl","duck","goose","hawk","crow","robin","sparrow"],
 fish=["salmon","tuna","shark","trout","cod","carp","perch"],
 tree=["oak","pine","maple","birch","cedar","elm","willow"],
 flower=["rose","tulip","daisy","lily","poppy","iris","violet"])
EMOS=dict(
 fear=["fear","dread","terror","panic","horror","fright","anxiety"],
 anger=["anger","rage","fury","wrath","annoyance","resentment","irritation"],
 joy=["joy","happiness","delight","glee","elation","cheer","bliss"],
 shame=["shame","embarrassment","humiliation","disgrace","guilt","remorse"],
 pride=["pride","triumph","confidence","vanity","dignity","arrogance"],
 love=["love","affection","adoration","fondness","tenderness","devotion"])
FRAME_N="A {} is a type of"; FRAME_E="The feeling of {}"

@torch.no_grad()
def resid(words,frame,L):
    h=f"blocks.{L}.hook_resid_post"; out=[]
    for w in words:
        _,c=m.run_with_cache(m.to_tokens(frame.format(w)),names_filter=h); out.append(c[h][0,-1].float().numpy())
    return np.array(out)

def classify_acc(R,y,K,loo):
    N=len(y); correct=0
    for i in range(N):
        means=np.array([R[[j for j in range(N) if y[j]==c and (not loo or j!=i)]].mean(0) for c in range(K)])
        dirs=means-means.mean(0,keepdims=True); dirs=dirs/(np.linalg.norm(dirs,axis=1,keepdims=True)+1e-8)
        if int(np.argmax(dirs@R[i]))==y[i]: correct+=1
    return correct/N
def mean_abs_cos(R,y,K):
    means=np.array([R[y==c].mean(0) for c in range(K)]); D=means-means.mean(0,keepdims=True)
    D=D/(np.linalg.norm(D,axis=1,keepdims=True)+1e-8); cs=[]
    for a in range(K):
        for b in range(a+1,K): cs.append(abs(float(D[a]@D[b])))
    return float(np.mean(cs))

def run_domain(name,CATS,frame):
    cats=list(CATS); words=[]; y=[]
    for ci,c in enumerate(cats):
        for w in CATS[c]: words.append(w); y.append(ci)
    y=np.array(y); K=len(cats); chance=1/K
    best=None
    for L in LAYERS:
        R=resid(words,frame,L)
        insamp=classify_acc(R,y,K,loo=False); loo=classify_acc(R,y,K,loo=True); cosm=mean_abs_cos(R,y,K)
        print(f"[{name}] L={L}: in-sample={insamp:.2f} LOO={loo:.2f} mean|cos|={cosm:.2f}",flush=True)
        if best is None or insamp>best["insamp"]: best=dict(L=L,insamp=insamp,loo=loo,cos=cosm,R=R)
    yshuf=np.random.default_rng(0).permutation(y)
    loo_shuf=classify_acc(best["R"],yshuf,K,loo=True)
    clean = best["loo"]>=2*chance and best["cos"]<0.60 and loo_shuf<1.6*chance
    res=dict(domain=name,K=K,chance=chance,best_layer=best["L"],in_sample=best["insamp"],
             LOO=best["loo"],mean_abs_cos=best["cos"],shuffled_LOO_canary=loo_shuf,clean_tree=bool(clean))
    print(f"[{name}] BEST L={best['L']} · LOO={best['loo']:.2f} (chance {chance:.2f}) · |cos|={best['cos']:.2f} "
          f"· shuffle-canary={loo_shuf:.2f} · clean_tree={clean}\n",flush=True)
    return res

print("=== concrete nouns (reference) ==="); n=run_domain("nouns",NOUNS,FRAME_N)
print("=== feeling / evaluation words ==="); e=run_domain("emotions",EMOS,FRAME_E)
verdict = "EMOTIONS_FORM_CLEAN_TREE" if e["clean_tree"] else "EMOTIONS_FUZZY_OR_NULL"
out=dict(nouns=n,emotions=e,verdict=verdict,
         note="warmup: concept-reading, not feeling; frames differ per domain (natural priming)")
print("=== SUMMARY ==="); print(json.dumps({k:(v if not isinstance(v,dict) else {kk:vv for kk,vv in v.items()}) for k,v in out.items()},indent=1))
print("\n=== VERDICT ===",verdict,f"| nouns LOO={n['LOO']:.2f} emotions LOO={e['LOO']:.2f} (chance {e['chance']:.2f}) · emo|cos|={e['mean_abs_cos']:.2f} · emo shuffle={e['shuffled_LOO_canary']:.2f}")
d=os.path.join(os.path.dirname(__file__),"..","campaigns","map-tree"); os.makedirs(d,exist_ok=True)
json.dump(out,open(os.path.join(d,"emotions_tree_result.json"),"w"),indent=1); print("[saved]",flush=True)
