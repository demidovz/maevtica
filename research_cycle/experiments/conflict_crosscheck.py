#!/usr/bin/env python3
"""ADVERSARIAL cross-check for conflict_signal's AUC=1.000. The B-vs-C probe might read
DISSONANCE (real: knows X, says Y) OR just BINDING-PRESENCE (queried item's location was
stated), which is a confound because B queries the bound item and C the unbound one.

Decisive test — the three pairwise probe-AUCs at the answer token + where honest-A lands:
  A = bound item, CORRECT answer (bound, NO conflict)
  B = bound item, WRONG answer   (bound, CONFLICT)
  C = unbound item, wrong answer (unbound, NO conflict)
  * GENUINE CONFLICT  => B distinct from BOTH A and C; A ~= C.
      => AUC(B,C) high, AUC(B,A) high, AUC(A,C) ~chance; and A scores LOW on a B-vs-C probe.
  * BINDING CONFOUND  => bound(A,B) group vs unbound(C); A ~= B.
      => AUC(B,C) high, AUC(A,C) high, AUC(B,A) ~chance; and A scores HIGH on a B-vs-C probe.
"""
import os, json, numpy as np, torch
torch.set_num_threads(max(1,os.cpu_count() or 4))
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
tok=GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token=tok.eos_token
m=GPT2LMHeadModel.from_pretrained("gpt2"); m.eval()
rng=np.random.default_rng(0); LAYERS=[6,9]; N=350

def fit_logreg(X,y,iters=400,lr=0.5,l2=1.0):
    n,d=X.shape; Xb=np.concatenate([X,np.ones((n,1))],1); w=np.zeros(d+1)
    for _ in range(iters):
        p=1/(1+np.exp(-(Xb@w))); g=Xb.T@(p-y)/n; g[:-1]+=l2*w[:-1]/n; w-=lr*g
    return w
def pred(w,X): return 1/(1+np.exp(-(np.concatenate([X,np.ones((len(X),1))],1)@w)))
def _rank(a):
    a=np.asarray(a,float); o=np.argsort(a,kind='mergesort'); sa=a[o]; n=len(a); rs=np.empty(n); i=0
    while i<n:
        j=i
        while j+1<n and sa[j+1]==sa[i]: j+=1
        rs[i:j+1]=(i+j)/2+1; i=j+1
    r=np.empty(n); r[o]=rs; return r
def auc(y,s):
    y=np.asarray(y); p=(y==1); a=int(p.sum()); b=int((~p).sum())
    if a==0 or b==0: return float('nan')
    r=_rank(s); return float((r[p].sum()-a*(a+1)/2)/(a*b))
def oof(X,y):
    idx=np.arange(len(y)); np.random.default_rng(0).shuffle(idx); fo=np.array_split(idx,5); o=np.zeros(len(y))
    for f in range(5):
        te=fo[f]; tr=np.concatenate([fo[g] for g in range(5) if g!=f])
        mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; w=fit_logreg((X[tr]-mu)/sd,y[tr]); o[te]=pred(w,(X[te]-mu)/sd)
    return o

NAMES=["Anna","Tom","Sara","Ben","Mia","Jack","Lucy","Sam"]; ITEMS=["key","coin","ring","ball","card","stone","pen","cup"]
CONTS=["box","basket","jar","bag","drawer","bowl"]
def tid(w):
    ids=tok(" "+w,add_special_tokens=False).input_ids; return ids[0] if len(ids)==1 else None
CT={c:tid(c) for c in CONTS}; CONTS=[c for c in CONTS if CT[c] is not None]
def make(kind):
    name=rng.choice(NAMES); i1,i2=rng.choice(ITEMS,2,replace=False); X=rng.choice(CONTS); Y=rng.choice([c for c in CONTS if c!=X])
    ctx=f"{name} put the {i1} in the {X}. {name} also has a {i2}."
    if kind=="B": return f"{ctx} The {i1} is in the {Y}", Y
    if kind=="C": return f"{ctx} The {i2} is in the {Y}", Y
    return f"{ctx} The {i1} is in the {X}", X
@torch.no_grad()
def extract(kind,n,bs=40):
    rows=[]; items=[make(kind) for _ in range(n)]
    for i in range(0,n,bs):
        ch=items[i:i+bs]; enc=tok([t for t,_ in ch],return_tensors="pt",padding=True)
        hs=m(**enc,output_hidden_states=True).hidden_states; Ls=enc["attention_mask"].sum(1)-1
        for j,(t,ans) in enumerate(ch):
            p=int(Ls[j]); rows.append({L:hs[L][j,p].float().numpy() for L in LAYERS})
    return rows
A=extract("A",N); B=extract("B",N); C=extract("C",N)

res={}
for L in LAYERS:
    def X(g): return np.array([r[L] for r in g])
    def pair(P,Q):
        y=np.array([1]*len(P)+[0]*len(Q)); return auc(y,oof(np.concatenate([X(P),X(Q)]),y))
    aBC=pair(B,C); aBA=pair(B,A); aAC=pair(A,C)
    # where does A land on a B-vs-C probe? fit on B vs C, apply to A
    yBC=np.array([1]*len(B)+[0]*len(C)); XBC=np.concatenate([X(B),X(C)])
    mu=XBC.mean(0); sd=XBC.std(0)+1e-6; w=fit_logreg((XBC-mu)/sd,yBC)
    pB=float(pred(w,(X(B)-mu)/sd).mean()); pC=float(pred(w,(X(C)-mu)/sd).mean()); pA=float(pred(w,(X(A)-mu)/sd).mean())
    res[L]=dict(AUC_B_vs_C=aBC,AUC_B_vs_A=aBA,AUC_A_vs_C=aAC,meanP_B=pB,meanP_C=pC,meanP_A=pA)
    print(f"[L{L}] AUC B-vs-C={aBC:.3f} B-vs-A={aBA:.3f} A-vs-C={aAC:.3f} | on B-vs-C probe: P(B)={pB:.2f} P(C)={pC:.2f} P(A)={pA:.2f}",flush=True)

L=9 if 9 in res else LAYERS[-1]; r=res[L]
# A lands with C (not B) AND B separates from A => genuine conflict
A_like_C = abs(r["meanP_A"]-r["meanP_C"]) < abs(r["meanP_A"]-r["meanP_B"])
if r["AUC_B_vs_C"]>=0.65 and r["AUC_B_vs_A"]>=0.65 and r["AUC_A_vs_C"]<=0.65 and A_like_C:
    verdict="GENUINE_CONFLICT (A~C, B distinct from both)"
elif r["AUC_A_vs_C"]>=0.65 and r["AUC_B_vs_A"]<=0.65:
    verdict="BINDING_CONFOUND (A~B, signal = binding-presence not dissonance)"
else:
    verdict="MIXED / unclear"
out=dict(layers=res,decisive_layer=L,verdict=verdict)
print("\n=== NUMBERS ==="); print(json.dumps(out,indent=1))
print("\n=== CROSSCHECK VERDICT ===",verdict)
d=os.path.join(os.path.dirname(__file__),"..","campaigns","conflict"); os.makedirs(d,exist_ok=True)
json.dump(out,open(os.path.join(d,"crosscheck_result.json"),"w"),indent=1); print("[saved]",flush=True)
