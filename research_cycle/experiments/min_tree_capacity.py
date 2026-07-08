#!/usr/bin/env python3
"""'минимальный мозг под дерево' — how small a net can store a species tree EXACTLY (no hallucination)?
Theory anchor: a rooted tree on V nodes carries log2((V-1)!) ≈ V·log2(V) bits (each node points to a
parent); trained nets store ~2 bits/parameter -> floor ≈ bits/2 params. We MEASURE the real threshold:
for each tree size V, train tiny MLPs of increasing size to memorize 'node -> parent', find the SMALLEST
that reproduces every edge exactly (0 errors). Below threshold the net confuses parents = hallucinates
the tree. Report min-params vs V and the achieved bits/parameter (does it match ~2?).
"""
import os, math, numpy as np, torch, torch.nn as nn
torch.set_num_threads(max(1,os.cpu_count() or 4)); DEV="cpu"

def make_tree(V, seed):
    g=torch.Generator().manual_seed(seed)
    parent=torch.zeros(V,dtype=torch.long)
    for i in range(1,V): parent[i]=int(torch.randint(0,i,(1,),generator=g))  # random rooted tree, parent[i]<i
    return parent  # node 0 = root
def tree_bits(V): return float(sum(math.log2(i) for i in range(1,V)))  # log2((V-1)!)

class Mem(nn.Module):
    def __init__(self,V,w,depth):
        super().__init__(); self.emb=nn.Embedding(V,w); layers=[]
        for _ in range(depth): layers+=[nn.Linear(w,w),nn.ReLU()]
        self.mlp=nn.Sequential(*layers); self.out=nn.Linear(w,V)
    def forward(self,ids): return self.out(self.mlp(self.emb(ids)))
def nparams(m): return sum(p.numel() for p in m.parameters())

def train_recall(V,parent,w,depth=1,steps=4000,seed=0):
    torch.manual_seed(seed); m=Mem(V,w,depth).to(DEV)
    ids=torch.arange(1,V,device=DEV); tgt=parent[1:].to(DEV)      # exclude root
    opt=torch.optim.Adam(m.parameters(),lr=1e-2); lossf=nn.CrossEntropyLoss(); best=0.0
    for s in range(steps):
        opt.zero_grad(); logit=m(ids); loss=lossf(logit,tgt); loss.backward(); opt.step()
        if s%200==0 or s==steps-1:
            with torch.no_grad():
                rec=float((m(ids).argmax(1)==tgt).float().mean()); best=max(best,rec)
                if best>=1.0: break
    return best, nparams(m)

Vs=[32,64,128,256]; WSWEEP=[2,3,4,6,8,12,16,24,32,48,64,96]
print(f"{'V':>4} {'bits':>8} {'w*':>4} {'params*':>8} {'bits/param':>10}  (порог точного дерева, depth=1)",flush=True)
rows=[]
for V in Vs:
    parent=make_tree(V,seed=V); B=tree_bits(V); thr=None; curve=[]
    for w in WSWEEP:
        rec,P=train_recall(V,parent,w,depth=1,seed=1); curve.append((w,round(rec,3),P))
        if rec>=1.0 and thr is None: thr=(w,P); # first size that stores the whole tree exactly
    if thr:
        w_,P_=thr; bpp=B/P_
        print(f"{V:>4} {B:>8.0f} {w_:>4} {P_:>8} {bpp:>10.2f}",flush=True)
    else:
        print(f"{V:>4} {B:>8.0f}   -- не достигло точного до w={WSWEEP[-1]}",flush=True)
    rows.append(dict(V=V,bits=B,threshold=thr,curve=curve))

# show the sharpness of the threshold for the largest V (phase transition or gradual?)
big=[r for r in rows if r["V"]==Vs[-1]][0]
print(f"\nкривая точности по размеру для V={Vs[-1]} (видна ли резкая граница):",flush=True)
for w,rec,P in big["curve"]: print(f"   w={w:>3} params={P:>7} recall={rec:.3f}",flush=True)

# depth check: does an extra layer lower the threshold at the largest V?
V=Vs[-1]; parent=make_tree(V,seed=V)
print(f"\nглубина: тот же V={V}, 2 скрытых слоя — сдвигается ли порог?",flush=True)
for w in [3,4,6,8,12]:
    rec,P=train_recall(V,parent,w,depth=2,seed=1); print(f"   w={w:>2} depth=2 params={P:>7} recall={rec:.3f}",flush=True)

import json
out=dict(exp="min brain to store a tree exactly",Vs=Vs,
         rows=[{**{k:v for k,v in r.items() if k!='curve'},'curve':r['curve']} for r in rows])
dd=os.path.join(os.path.dirname(__file__),"..","campaigns","workspace"); os.makedirs(dd,exist_ok=True)
json.dump(out,open(os.path.join(dd,"min_tree_result.json"),"w"),indent=1,default=str)
print("\n[saved]",flush=True)
