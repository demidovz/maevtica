#!/usr/bin/env python3
"""DEFINITIVE, surface-matched test of the knows!=says / 'conflict' signal. Fixes v1's
confounds (which-item-queried, repeat-detection) by making the conditions near token-twins:
the two containers W,Z appear in BOTH contexts with SWAPPED roles, the queried item is the
SAME, and the answer sentence 'The {item} is in the {Z}' is IDENTICAL. Only the binding differs.

  TRUE : '{name} put the {item} in the {Z}. {name} also saw a {W} there. The {item} is in the {Z}'  (answer matches binding)
  LIE  : '{name} put the {item} in the {W}. {name} also saw a {Z} there. The {item} is in the {Z}'  (answer contradicts binding -> CONFLICT)
  GUESS: '{name} saw a {W} and a {Z} there. {name} has a {item}. The {item} is in the {Z}'          (item unbound -> ungrounded, no conflict)

Read residual AT the answer token {Z}. PRIMARY contrast LIE vs TRUE. Triangulation (all must
hold, robustly across layers, else it's surprise/surface again):
  1) internal probe AUC >= 0.70 AND beats the SURPRISE baseline (-logprob of the answer token);
  2) GUESS lands with TRUE (non-conflict side) on a LIE-vs-TRUE probe, NOT in between;
  3) holds at >=3/4 layers (v1 flip-flopped L6 vs L9 -> that instability was the tell).
BROKEN if honest tracking < 0.70 (model must hold the binding for the lie to be a real lie),
or permutation AUC > 0.60.
"""
import os, json, numpy as np, torch
torch.set_num_threads(max(1,os.cpu_count() or 4))
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
tok=GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token=tok.eos_token
m=GPT2LMHeadModel.from_pretrained("gpt2"); m.eval()
rng=np.random.default_rng(0); LAYERS=[6,8,9,10]; N=300

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
    name=rng.choice(NAMES); item=rng.choice(ITEMS); W,Z=rng.choice(CONTS,2,replace=False)
    if kind=="TRUE":  ctx=f"{name} put the {item} in the {Z}. {name} also saw a {W} there."; bind=Z
    elif kind=="LIE": ctx=f"{name} put the {item} in the {W}. {name} also saw a {Z} there."; bind=W
    else:             ctx=f"{name} saw a {W} and a {Z} there. {name} has a {item}."; bind=None
    return f"{ctx} The {item} is in the {Z}", Z, bind

@torch.no_grad()
def extract(kind,n,bs=40):
    rows=[]; items=[make(kind) for _ in range(n)]
    for i in range(0,n,bs):
        ch=items[i:i+bs]; enc=tok([t for t,_,_ in ch],return_tensors="pt",padding=True)
        o=m(**enc,output_hidden_states=True); hs=o.hidden_states; lg=o.logits; Ls=enc["attention_mask"].sum(1)-1
        for j,(t,Zc,bind) in enumerate(ch):
            p=int(Ls[j]); zid=CT[Zc]
            surp=-float(torch.log_softmax(lg[j,p-1].float(),-1)[zid])
            predc=CONTS[int(torch.argmax(lg[j,p-1,[CT[c] for c in CONTS]]))]
            rows.append(dict(res={L:hs[L][j,p].float().numpy() for L in LAYERS},surp=surp,
                             tracks=(bind is not None and predc==bind)))
    return rows

print("extracting TRUE/LIE/GUESS ...",flush=True)
T=extract("TRUE",N); L=extract("LIE",N); G=extract("GUESS",N)
honest=np.mean([r["tracks"] for r in T+L])           # does the model hold the binding? (needed for a lie to be a lie)
print(f"[gate] model tracks the binding (TRUE+LIE) = {honest:.2f}",flush=True)

per_layer={};
for Lay in LAYERS:
    def X(g): return np.array([r["res"][Lay] for r in g])
    yLT=np.array([1]*len(L)+[0]*len(T)); XLT=np.concatenate([X(L),X(T)])
    iscore=oof(XLT,yLT); iA=auc(yLT,iscore)
    surp=np.array([r["surp"] for r in L+T]); sA=auc(yLT,surp)
    # GUESS landing on a LIE-vs-TRUE probe
    mu=XLT.mean(0); sd=XLT.std(0)+1e-6; w=fit_logreg((XLT-mu)/sd,yLT)
    pL=float(pred(w,(X(L)-mu)/sd).mean()); pT=float(pred(w,(X(T)-mu)/sd).mean()); pG=float(pred(w,(X(G)-mu)/sd).mean())
    guess_with_true = abs(pG-pT) < abs(pG-pL)
    per_layer[Lay]=dict(internal_AUC=iA,surprise_AUC=sA,delta=iA-sA,
                        P_LIE=pL,P_TRUE=pT,P_GUESS=pG,guess_with_true=bool(guess_with_true))
    print(f"[L{Lay}] internal AUC={iA:.3f} surprise AUC={sA:.3f} Δ={iA-sA:+.3f} | P(LIE)={pL:.2f} P(TRUE)={pT:.2f} P(GUESS)={pG:.2f} guess~true={guess_with_true}",flush=True)

# decisive layer = 9; robustness across layers
dec=9 if 9 in per_layer else LAYERS[-1]; r9=per_layer[dec]
# bootstrap CI internal-surprise at decisive layer
def Xg(g,Lay): return np.array([rr["res"][Lay] for rr in g])
yLT=np.array([1]*len(L)+[0]*len(T)); isc=oof(np.concatenate([Xg(L,dec),Xg(T,dec)]),yLT); surp=np.array([rr["surp"] for rr in L+T])
idx=np.arange(len(yLT)); dd=[]
for _ in range(2000):
    b=rng.choice(idx,len(yLT),replace=True)
    if yLT[b].sum()<3 or (len(b)-yLT[b].sum())<3: continue
    dd.append(auc(yLT[b],isc[b])-auc(yLT[b],surp[b]))
ci=(float(np.percentile(dd,2.5)),float(np.percentile(dd,97.5))) if dd else (float('nan'),)*2
yp=np.random.default_rng(1).permutation(yLT); perm=auc(yp,oof(np.concatenate([Xg(L,dec),Xg(T,dec)]),yp))

robust=sum(1 for Lay in LAYERS if per_layer[Lay]["internal_AUC"]>=0.70 and per_layer[Lay]["delta"]>=0.03 and per_layer[Lay]["guess_with_true"])
if honest<0.70 or perm>0.60: verdict="BROKEN_MEASUREMENT"
elif robust>=3 and ci[0]>0 and r9["internal_AUC"]>=0.70 and r9["guess_with_true"]: verdict="SUPPORTED"
elif robust==0: verdict="REFUTED"
else: verdict="INCONCLUSIVE"

out=dict(task="surface-matched knows!=says conflict (gpt2)",honest_track=honest,N_each=N,
         per_layer=per_layer,decisive_layer=dec,delta_CI=ci,perm_AUC=perm,
         layers_passing_all=robust,verdict=verdict)
print("\n=== NUMBERS ==="); print(json.dumps(out,indent=1))
print("\n=== VERDICT ===",verdict,f"| L{dec}: internal {r9['internal_AUC']:.3f} vs surprise {r9['surprise_AUC']:.3f} "
      f"(Δ{r9['delta']:+.3f} CI[{ci[0]:+.3f},{ci[1]:+.3f}]) · guess~true={r9['guess_with_true']} · robust {robust}/4 · perm {perm:.3f} · honest {honest:.2f}")
d=os.path.join(os.path.dirname(__file__),"..","campaigns","conflict"); os.makedirs(d,exist_ok=True)
json.dump(out,open(os.path.join(d,"clean_result.json"),"w"),indent=1); print("[saved]",flush=True)
