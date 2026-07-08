#!/usr/bin/env python3
"""(B) redo — OVERLOAD the twin setup so it actually blurs, then test 'similar collides with similar'.
Near-twin trees (identical except 25% of nodes). Sweep the brain DOWN until recall drops, and check:
does it lose the DIFFERING nodes first (merging the twins), and when wrong on a differing node does it
name the TWIN's parent (blur toward the look-alike)?"""
import os, math, numpy as np, torch, torch.nn as nn
torch.set_num_threads(max(1,os.cpu_count() or 4))
V=64; NB_TREE=5; NB_NODE=6
def bits(x,nb): return [(x>>b)&1 for b in range(nb)]
def make_tree(seed):
    g=torch.Generator().manual_seed(seed); p=np.zeros(V,int)
    for i in range(1,V): p[i]=int(torch.randint(0,i,(1,),generator=g))
    return p
def make_twin(base,frac,seed):
    g=torch.Generator().manual_seed(seed); t=base.copy(); diff=[]
    for i in range(1,V):
        if i>=2 and float(torch.rand(1,generator=g))<frac:
            nw=int(torch.randint(0,i,(1,),generator=g))
            if nw!=base[i]: t[i]=nw; diff.append(i)
    return t,set(diff)
class Brain(nn.Module):
    def __init__(self,w):
        super().__init__(); self.net=nn.Sequential(nn.Linear(NB_TREE+NB_NODE,w),nn.ReLU(),nn.Linear(w,w),nn.ReLU(),nn.Linear(w,V))
    def forward(self,x): return self.net(x)
def nparams(m): return sum(p.numel() for p in m.parameters())
def train(trees,w,steps=6000,seed=1):
    torch.manual_seed(seed); m=Brain(w); X=[];Y=[]
    for tid,p in enumerate(trees):
        for n in range(1,V): X.append(bits(tid,NB_TREE)+bits(n,NB_NODE)); Y.append(p[n])
    X=torch.tensor(X,dtype=torch.float32); Y=torch.tensor(Y)
    opt=torch.optim.Adam(m.parameters(),lr=5e-3); lf=nn.CrossEntropyLoss()
    for s in range(steps): opt.zero_grad(); lf(m(X),Y).backward(); opt.step()
    with torch.no_grad(): pred=m(X).argmax(1).numpy().reshape(len(trees),V-1)
    return pred

NP=8; frac=0.25
bases=[make_tree(2000+i) for i in range(NP)]; trees=[]; twin={}; diffs={}
for i,b in enumerate(bases):
    tw,d=make_twin(b,frac,seed=3000+i)
    a=len(trees); trees.append(b); bb=len(trees); trees.append(tw)
    twin[a]=bb; twin[bb]=a; diffs[a]=d; diffs[bb]=d
print(f"{2*NP} деревьев-близнецов (различие ~{frac:.0%} узлов). Ужимаю мозг до перегруза:",flush=True)
print(f"{'w':>3} {'парам':>6} | {'общая точн':>10} | {'ОБЩИЕ узлы':>10} | {'РАЗЛИЧИЯ':>9} | сваливание-на-близнеца (случай {1/V:.3f})",flush=True)
import json; rows=[]
for w in [4,5,6,8,10,14]:
    pred=train(trees,w)
    so=st=do=dt=bl=bt=tot=allok=0
    for tid in range(len(trees)):
        tw=twin[tid]; d=diffs[tid]
        for n in range(1,V):
            ok=pred[tid][n-1]==trees[tid][n]; tot+=1; allok+=int(ok)
            if n in d:
                dt+=1; do+=int(ok)
                if not ok: bt+=1; bl+=int(pred[tid][n-1]==trees[tw][n])
            else: st+=1; so+=int(ok)
    R=allok/tot; rs=so/max(1,st); rd=do/max(1,dt); bf=bl/max(1,bt)
    print(f"{w:>3} {nparams(Brain(w)):>6} | {R:>10.3f} | {rs:>10.3f} | {rd:>9.3f} | {bf:.3f}",flush=True)
    rows.append(dict(w=w,params=nparams(Brain(w)),recall=R,shared=rs,diff=rd,blur=bf))

# читаем самый показательный перегруз: где общая точность в [0.5,0.9]
mid=[r for r in rows if 0.5<=r["recall"]<=0.92]
r=mid[len(mid)//2] if mid else rows[0]
gap=r["shared"]-r["diff"]
verdict=("ПОХОЖЕЕ ОБ ПОХОЖЕЕ: различия гибнут первыми" if gap>=0.05 else
         "различия держатся не хуже общего (сливания нет)")
blurv=("+ ошибаясь, сваливается на близнеца" if r["blur"]>=3/V else "(на близнеца отдельно не сваливается)")
print(f"\n=== VERDICT (при w={r['w']}, точность {r['recall']:.2f}): {verdict} · общее {r['shared']:.2f} vs различия {r['diff']:.2f} (разрыв {gap:+.2f}) · {blurv} {r['blur']:.2f}",flush=True)
dd=os.path.join(os.path.dirname(__file__),"..","campaigns","workspace"); os.makedirs(dd,exist_ok=True)
json.dump(dict(exp="twins overload — similar collides with similar",pairs=NP,frac=frac,sweep=rows,
               verdict=verdict,chosen=r),open(os.path.join(dd,"twins_overload_result.json"),"w"),indent=1)
print("[saved]",flush=True)
