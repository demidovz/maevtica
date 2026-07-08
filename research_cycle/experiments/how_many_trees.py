#!/usr/bin/env python3
"""'сколько деревьев влезет, пока не смешаются' — pack many trees into ONE fixed-size brain.
(A) CAPACITY: recall vs number of trees N (fixed net) — when do they overflow and blur?
(B) SIMILARITY BLUR: near-twin trees (identical except a fraction of nodes). Does the brain lose
    exactly the nodes where twins DIFFER (merging them into one), and when wrong on a differing node
    does it predict the TWIN's parent (blur toward the look-alike)?
Encoding: input = bits(tree_id) ++ bits(node) -> logits over parent. Net params FIXED as N grows.
"""
import os, math, numpy as np, torch, torch.nn as nn
torch.set_num_threads(max(1,os.cpu_count() or 4)); DEV="cpu"
V=64; NB_TREE=5; NB_NODE=6                       # fixed input width (<=32 trees, 64 nodes)
def bits(x,nb): return [(x>>b)&1 for b in range(nb)]
def tree_bits(v): return sum(math.log2(i) for i in range(1,v))
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
        super().__init__(); din=NB_TREE+NB_NODE
        self.net=nn.Sequential(nn.Linear(din,w),nn.ReLU(),nn.Linear(w,w),nn.ReLU(),nn.Linear(w,V))
    def forward(self,x): return self.net(x)
def nparams(m): return sum(p.numel() for p in m.parameters())

def encode(tid,node): return bits(tid,NB_TREE)+bits(node,NB_NODE)
def train(trees,w,steps=6000,seed=1):
    torch.manual_seed(seed); m=Brain(w)
    X=[];Y=[]
    for tid,p in enumerate(trees):
        for n in range(1,V): X.append(encode(tid,n)); Y.append(p[n])
    X=torch.tensor(X,dtype=torch.float32); Y=torch.tensor(Y)
    opt=torch.optim.Adam(m.parameters(),lr=5e-3); lf=nn.CrossEntropyLoss()
    for s in range(steps):
        opt.zero_grad(); loss=lf(m(X),Y); loss.backward(); opt.step()
    with torch.no_grad(): pred=m(X).argmax(1).numpy()
    return m,pred

# ---------- (A) capacity: recall vs N ----------
W_A=32; import json
print(f"=== (A) сколько деревьев влезает (мозг фикс. w={W_A}) ===",flush=True)
mtmp=Brain(W_A); cap_params=nparams(mtmp); cap_bits=cap_params*0.5
print(f"мозг: {cap_params} параметров ~ {cap_bits:.0f} бит ёмкости (0.5 бита/парам). Дерево весит ~{tree_bits(V):.0f} бит.",flush=True)
print(f"{'N деревьев':>10} | {'общий вес(бит)':>14} | {'средняя точность':>16}",flush=True)
A=[]
for N in [1,2,4,6,8,12,16,24,32]:
    trees=[make_tree(1000+i) for i in range(N)]
    m,pred=train(trees,W_A); pred=pred.reshape(N,V-1)
    recs=[(pred[i]==np.array([trees[i][n] for n in range(1,V)])).mean() for i in range(N)]
    r=float(np.mean(recs)); need=N*tree_bits(V)
    print(f"{N:>10} | {need:>14.0f} | {r:>16.3f}",flush=True)
    A.append(dict(N=N,total_bits=need,recall=r))

# ---------- (B) similarity blur: near-twins ----------
print(f"\n=== (B) похожее об похожее (пары близнецов, различие 25% узлов) ===",flush=True)
NP=6; W_B=24; frac=0.25
bases=[make_tree(2000+i) for i in range(NP)]; trees=[]; twinmap={}; diffs={}
for i,b in enumerate(bases):
    tw,d=make_twin(b,frac,seed=3000+i)
    a_id=len(trees); trees.append(b); b_id=len(trees); trees.append(tw)
    twinmap[a_id]=b_id; twinmap[b_id]=a_id; diffs[a_id]=d; diffs[b_id]=d
m,pred=train(trees,W_B); pred=pred.reshape(len(trees),V-1)
shared_ok=shared_tot=diff_ok=diff_tot=0; blur=blur_tot=0
for tid in range(len(trees)):
    tw=twinmap[tid]; d=diffs[tid]
    for n in range(1,V):
        ok = pred[tid][n-1]==trees[tid][n]
        if n in d:  # nodes where the twins DIFFER
            diff_tot+=1; diff_ok+=int(ok)
            if not ok:
                blur_tot+=1; blur+=int(pred[tid][n-1]==trees[tw][n])  # predicted the twin's parent?
        else:
            shared_tot+=1; shared_ok+=int(ok)
rs=shared_ok/max(1,shared_tot); rd=diff_ok/max(1,diff_tot); blur_frac=blur/max(1,blur_tot)
print(f"мозг w={W_B} ({nparams(Brain(W_B))} парам), {len(trees)} деревьев-близнецов",flush=True)
print(f"точность на ОБЩИХ узлах (близнецы согласны): {rs:.3f}",flush=True)
print(f"точность на РАЗЛИЧАЮЩИХСЯ узлах (близнецы спорят): {rd:.3f}   <- теряет ли различия первыми",flush=True)
print(f"когда ошибся на различии — назвал родителя БЛИЗНЕЦА: {blur_frac:.3f} (случайно ~{1/V:.3f})",flush=True)

verdict_A="есть перегруз (точность падает с числом деревьев)" if A[-1]["recall"]<A[0]["recall"]-0.1 else "перегруза не видно"
verdict_B="ПОХОЖЕЕ ОБ ПОХОЖЕЕ: различия теряются первыми" if (rs-rd)>=0.05 else "различия держатся не хуже общего"
print(f"\n=== VERDICT (A) {verdict_A}",flush=True)
print(f"=== VERDICT (B) {verdict_B} · различия {rd:.2f} vs общее {rs:.2f} · сваливание на близнеца {blur_frac:.2f} vs случай {1/V:.3f}",flush=True)
dd=os.path.join(os.path.dirname(__file__),"..","campaigns","workspace"); os.makedirs(dd,exist_ok=True)
json.dump(dict(exp="how many trees fit before they blur",V=V,capacity=A,
               similarity=dict(shared_recall=rs,diff_recall=rd,blur_to_twin=blur_frac,chance=1/V),
               verdict_A=verdict_A,verdict_B=verdict_B),
          open(os.path.join(dd,"how_many_trees_result.json"),"w"),indent=1)
print("[saved]",flush=True)
