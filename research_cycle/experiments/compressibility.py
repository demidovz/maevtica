#!/usr/bin/env python3
"""'сжимаемость' — does a net ARCHIVE a regular tree by finding its RULE (few params, generalizes),
while a random tree must be memorized (many params, no generalization)?
Node i is fed as its BINARY BITS so the net CAN do arithmetic on the index.
  REGULAR tree = binary heap: parent(i)=i//2  (== drop the last bit; a clean shift rule).
  RANDOM tree  = parent(i)=rand in [1..i-1]   (no rule).
Two measurements:
  (A) STORAGE THRESHOLD: min params for exact recall on all edges, vs V. Regular should be far smaller
      and grow far slower (rule doesn't grow with V); random ~ V·log V.
  (B) RULE READOUT via GENERALIZATION: train on 75% of nodes, test held-out 25%. Regular should
      generalize (it learned the shift rule); random cannot (no rule) -> held-out ≈ chance.
"""
import os, math, numpy as np, torch, torch.nn as nn
torch.set_num_threads(max(1,os.cpu_count() or 4)); DEV="cpu"

def bits_of(i,nb): return [(i>>b)&1 for b in range(nb)]
def make_regular(V): return np.array([0]+[i//2 for i in range(1,V)],dtype=int)  # heap: parent=i//2
def make_random(V,seed):
    g=torch.Generator().manual_seed(seed); p=np.zeros(V,int)
    for i in range(1,V): p[i]=int(torch.randint(0,i,(1,),generator=g))
    return p

class Net(nn.Module):
    def __init__(self,nb,w,V):
        super().__init__(); self.net=nn.Sequential(nn.Linear(nb,w),nn.ReLU(),nn.Linear(w,w),nn.ReLU(),nn.Linear(w,V))
    def forward(self,x): return self.net(x)
def nparams(m): return sum(p.numel() for p in m.parameters())

def train(V,parent,w,idx_train,steps=3000,wd=0.0,seed=1):
    nb=max(1,math.ceil(math.log2(V))); torch.manual_seed(seed)
    X=torch.tensor([bits_of(i,nb) for i in range(V)],dtype=torch.float32)
    m=Net(nb,w,V); opt=torch.optim.Adam(m.parameters(),lr=5e-3,weight_decay=wd); lf=nn.CrossEntropyLoss()
    it=torch.tensor(idx_train); tgt=torch.tensor(parent[idx_train])
    for s in range(steps):
        opt.zero_grad(); loss=lf(m(X[it]),tgt); loss.backward(); opt.step()
    with torch.no_grad(): pred=m(X).argmax(1).numpy()
    return pred, nparams(m)

Vs=[64,128,256,512]; WSWEEP=[2,3,4,6,8,12,16,24,32,48,64]
print("=== (A) порог точного хранения: параметров нужно ===",flush=True)
print(f"{'V':>4} | {'RANDOM w*/params':>18} | {'REGULAR w*/params':>18} | выигрыш",flush=True)
A=[]
for V in Vs:
    pr=make_random(V,seed=V); pg=make_regular(V); alln=np.arange(1,V)
    def thresh(parent):
        for w in WSWEEP:
            pred,P=train(V,parent,w,alln); rec=(pred[1:]==parent[1:]).mean()
            if rec>=1.0: return w,P
        return None,None
    wr,Pr=thresh(pr); wg,Pg=thresh(pg)
    gain = (Pr/Pg) if (Pr and Pg) else float('nan')
    print(f"{V:>4} | {str(wr)+'/'+str(Pr):>18} | {str(wg)+'/'+str(Pg):>18} | {gain:.1f}x" if Pr and Pg else f"{V:>4} | rnd {wr}/{Pr} | reg {wg}/{Pg}",flush=True)
    A.append(dict(V=V,rnd=(wr,Pr),reg=(wg,Pg),gain=gain))

print("\n=== (B) правило или таблица: обобщение на невиданные узлы (train 75%, test 25%) ===",flush=True)
print(f"{'V':>4} | {'RANDOM held-out':>16} | {'REGULAR held-out':>16}  (chance≈1/V)",flush=True)
B=[]
for V in Vs:
    pr=make_random(V,seed=V); pg=make_regular(V)
    rng=np.random.default_rng(0); nodes=np.arange(1,V); rng.shuffle(nodes)
    cut=int(0.75*len(nodes)); tr=nodes[:cut]; te=nodes[cut:]
    predr,_=train(V,pr,24,tr,steps=6000,wd=1e-2); predg,_=train(V,pg,24,tr,steps=6000,wd=1e-2)
    hr=(predr[te]==pr[te]).mean(); hg=(predg[te]==pg[te]).mean()
    print(f"{V:>4} | {hr:>16.3f} | {hg:>16.3f}  (chance {1/V:.3f})",flush=True)
    B.append(dict(V=V,rnd_heldout=float(hr),reg_heldout=float(hg),chance=1/V))

reg_gen=np.mean([b["reg_heldout"] for b in B]); rnd_gen=np.mean([b["rnd_heldout"] for b in B])
gains=[a["gain"] for a in A if a["gain"]==a["gain"]]
verdict = ("ARCHIVES THE RULE (regular compresses AND generalizes)" if (reg_gen>=0.6 and np.mean(gains)>=2)
           else "PARTIAL" if (reg_gen>=0.6 or np.mean(gains)>=2) else "MEMORIZES BOTH (no rule extraction)")
print(f"\n=== VERDICT ===",verdict,f"| порог-выигрыш ~{np.mean(gains):.1f}x · обобщение regular {reg_gen:.2f} vs random {rnd_gen:.2f}",flush=True)
import json; dd=os.path.join(os.path.dirname(__file__),"..","campaigns","workspace"); os.makedirs(dd,exist_ok=True)
json.dump(dict(exp="compressibility: rule vs table",thresholds=A,generalization=B,verdict=verdict),
          open(os.path.join(dd,"compressibility_result.json"),"w"),indent=1,default=str)
print("[saved]",flush=True)
