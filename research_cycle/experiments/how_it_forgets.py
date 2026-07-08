#!/usr/bin/env python3
"""'как оно забывает' — below the capacity threshold, WHICH tree edges break first?
Random, or STRUCTURED? Hypothesis: a too-small net keeps the TRUNK (coarse, high-importance edges)
and hallucinates the TWIGS (fine detail) — lossy compression that follows the tree's own importance.
We train tiny MLPs BELOW the exact-storage threshold and correlate edge-correctness with three
importance measures — node DEPTH (trunk=shallow), SUBTREE SIZE under the node (trunk=big), PARENT
POPULARITY (hub in-degree). And when WRONG, where does it point — to a HUB? to an ANCESTOR (climbs
its own lineage but blurs the rung)?
"""
import os, math, numpy as np, torch, torch.nn as nn
torch.set_num_threads(max(1,os.cpu_count() or 4)); DEV="cpu"

def make_tree(V,seed):
    g=torch.Generator().manual_seed(seed); parent=np.zeros(V,dtype=int)
    for i in range(1,V): parent[i]=int(torch.randint(0,i,(1,),generator=g))
    return parent
def tree_features(parent):
    V=len(parent); depth=np.zeros(V,int);
    for i in range(1,V): depth[i]=depth[parent[i]]+1
    size=np.ones(V,int)
    for i in range(V-1,0,-1): size[parent[i]]+=size[i]     # subtree size (descendants+self)
    indeg=np.zeros(V,int)
    for i in range(1,V): indeg[parent[i]]+=1               # how many children a node has
    def ancestors(i):
        a=set(); j=i
        while j!=0: j=parent[j]; a.add(j)
        return a
    return depth,size,indeg,ancestors

class Mem(nn.Module):
    def __init__(self,V,w):
        super().__init__(); self.emb=nn.Embedding(V,w); self.h=nn.Sequential(nn.Linear(w,w),nn.ReLU()); self.out=nn.Linear(w,V)
    def forward(self,ids): return self.out(self.h(self.emb(ids)))
def train_pred(V,parent,w,steps=5000,seed=1):
    torch.manual_seed(seed); m=Mem(V,w).to(DEV)
    ids=torch.arange(1,V); tgt=torch.tensor(parent[1:])
    opt=torch.optim.Adam(m.parameters(),lr=1e-2); lf=nn.CrossEntropyLoss()
    for s in range(steps):
        opt.zero_grad(); loss=lf(m(ids),tgt); loss.backward(); opt.step()
    with torch.no_grad(): pred=m(ids).argmax(1).numpy()
    return pred  # predicted parent for nodes 1..V-1

def auc(y,s):
    y=np.asarray(y); p=(y==1); a=int(p.sum()); b=int((~p).sum())
    if a==0 or b==0: return float('nan')
    o=np.argsort(np.argsort(s)); r=o.astype(float)+1
    return float((r[p].sum()-a*(a+1)/2)/(a*b))

V=256; parent=make_tree(V,seed=V); depth,size,indeg,anc=tree_features(parent)
nodes=np.arange(1,V); Dp=depth[nodes]; Sz=size[nodes]; Pin=indeg[parent[nodes]]  # per-edge features
hub_thresh=np.percentile(indeg,90); hubs=set(np.where(indeg>=hub_thresh)[0])
base_anc=float(np.mean([len(anc(i)) for i in nodes])/V)   # chance a random node is an ancestor
base_indeg=float(indeg.mean())
print(f"V={V} · avg depth {Dp.mean():.1f} · avg subtree {Sz.mean():.1f} · hubs(top10%) in-deg>={hub_thresh:.0f}",flush=True)
print(f"baselines: P(random node is ancestor)={base_anc:.3f} · mean in-degree of a random node={base_indeg:.2f}\n",flush=True)

import json; results=[]
for w in [2,3,4]:
    pred=train_pred(V,parent,w); true=parent[nodes]; correct=(pred==true).astype(int); rec=correct.mean()
    # is correctness predicted by importance? (AUC>0.5 => structured degradation, trunk survives)
    a_size=auc(correct,Sz); a_shallow=auc(correct,-Dp); a_pop=auc(correct,Pin)
    # permutation canary on the headline (subtree size)
    perm=np.mean([auc(np.random.default_rng(k).permutation(correct),Sz) for k in range(20)])
    # where do WRONG edges point?
    wrong=nodes[correct==0]; wp=pred[correct==0]
    if len(wrong)>0:
        f_hub=float(np.mean([p in hubs for p in wp]))
        f_anc=float(np.mean([wp[k] in anc(wrong[k]) for k in range(len(wrong))]))  # predicted is an ancestor of the node
        mean_wp_indeg=float(indeg[wp].mean())
    else: f_hub=f_anc=mean_wp_indeg=float('nan')
    print(f"w={w} (recall {rec:.2f}):",flush=True)
    print(f"   correctness ~ importance AUC:  subtree={a_size:.3f}  shallow={a_shallow:.3f}  popular-parent={a_pop:.3f}  (perm {perm:.3f}; 0.5=random)",flush=True)
    print(f"   correct-vs-wrong means:  subtree {Sz[correct==1].mean():.1f} vs {Sz[correct==0].mean():.1f} · depth {Dp[correct==1].mean():.1f} vs {Dp[correct==0].mean():.1f}",flush=True)
    print(f"   WRONG points to: hub {f_hub:.2f} (base {len(hubs)/V:.2f}) · ancestor-of-node {f_anc:.2f} (base {base_anc:.2f}) · pred in-deg {mean_wp_indeg:.1f} (base {base_indeg:.1f})\n",flush=True)
    results.append(dict(w=w,recall=float(rec),auc_subtree=a_size,auc_shallow=a_shallow,auc_popular=a_pop,perm=float(perm),
                        wrong_to_hub=f_hub,hub_base=len(hubs)/V,wrong_to_ancestor=f_anc,anc_base=base_anc,
                        pred_indeg=mean_wp_indeg,indeg_base=base_indeg))

# verdict: structured if importance predicts correctness (any AUC >=0.60) AND/OR wrong collapses to hubs/ancestors above base
r0=results[0]
structured = (max(r0['auc_subtree'],r0['auc_shallow'],r0['auc_popular'])>=0.60) or \
             (r0['wrong_to_hub']>=1.5*r0['hub_base']) or (r0['wrong_to_ancestor']>=1.5*r0['anc_base']) or (r0['pred_indeg']>=1.5*r0['indeg_base'])
verdict="STRUCTURED forgetting (keeps trunk/hubs, loses twigs)" if structured else "RANDOM-ish forgetting (no importance structure)"
print("=== VERDICT ===",verdict,flush=True)
dd=os.path.join(os.path.dirname(__file__),"..","campaigns","workspace"); os.makedirs(dd,exist_ok=True)
json.dump(dict(exp="how it forgets a tree below capacity",V=V,results=results,verdict=verdict),
          open(os.path.join(dd,"how_it_forgets_result.json"),"w"),indent=1)
print("[saved]",flush=True)
