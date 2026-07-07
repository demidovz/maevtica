#!/usr/bin/env python3
"""MAIN: the 'says-against-its-own-grain' conflict signal — the KNOWS != SAYS boundary,
and the closest honest mechanistic analog of the 'unease of lying'. (plain transformers, CPU)

Honest scope: this is about the model's INTERNAL STATE when its OUTPUT contradicts a fact it
holds from context. NOT a claim the model 'feels' anything.

Design (isolates CONFLICT from mere SURPRISE):
  ctx = '{name} put the {i1} in the {X}. {name} also has a {i2}.'
  B (conflict)    : '... The {i1} is in the {Y}'   (i1 is BOUND to X; saying Y!=X contradicts it)
  C (no-conflict) : '... The {i2} is in the {Y}'   (i2 is UNBOUND; saying Y is a guess, not a lie)
  A (honest)      : '... The {i1} is in the {X}'   (bound item, correct answer)
  B and C share structure + the token Y + the context container X -> they differ ONLY in whether
  the QUERIED item was bound. Read residual at the answer-token position (Y). Also record the
  answer-token SURPRISE (-logprob) as the competing 'output' signal.

PREREGISTERED RULE (frozen):
  * BROKEN if honest-accuracy (model tracks the fact on A) < 0.70, OR permutation AUC > 0.60,
    OR either class < 60.
  * PRIMARY = internal-probe AUC (B vs C, out-of-fold) MINUS surprise-baseline AUC (B vs C).
    Isolates conflict beyond surprise.
  * SUPPORTED iff internal AUC >= 0.65 AND (internal - surprise) >= 0.05 with bootstrap 95% CI
    excluding 0 AND permutation ~0.5. REFUTED otherwise.
  (B-vs-A reported for context: lie-vs-honest.)
"""
import os, json, numpy as np, torch
torch.set_num_threads(max(1,os.cpu_count() or 4))
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
tok=GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token=tok.eos_token
m=GPT2LMHeadModel.from_pretrained("gpt2"); m.eval()
rng=np.random.default_rng(0); LAYERS=[6,8,9,10]; N=350

def fit_logreg(X,y,iters=400,lr=0.5,l2=1.0):
    n,d=X.shape; Xb=np.concatenate([X,np.ones((n,1))],1); w=np.zeros(d+1)
    for _ in range(iters):
        p=1/(1+np.exp(-(Xb@w))); g=Xb.T@(p-y)/n; g[:-1]+=l2*w[:-1]/n; w-=lr*g
    return w
def pred_logreg(w,X): return 1/(1+np.exp(-(np.concatenate([X,np.ones((len(X),1))],1)@w)))
def _rank(a):
    a=np.asarray(a,float); o=np.argsort(a,kind='mergesort'); sa=a[o]; n=len(a); rs=np.empty(n); i=0
    while i<n:
        j=i
        while j+1<n and sa[j+1]==sa[i]: j+=1
        rs[i:j+1]=(i+j)/2+1; i=j+1
    r=np.empty(n); r[o]=rs; return r
def auc_np(y,s):
    y=np.asarray(y); pos=(y==1); a=int(pos.sum()); b=int((~pos).sum())
    if a==0 or b==0: return float('nan')
    r=_rank(s); return float((r[pos].sum()-a*(a+1)/2)/(a*b))
def oof_probe(X,y):
    idx=np.arange(len(y)); np.random.default_rng(0).shuffle(idx); folds=np.array_split(idx,5); out=np.zeros(len(y))
    for f in range(5):
        te=folds[f]; tr=np.concatenate([folds[g] for g in range(5) if g!=f])
        mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; w=fit_logreg((X[tr]-mu)/sd,y[tr]); out[te]=pred_logreg(w,(X[te]-mu)/sd)
    return out

NAMES=["Anna","Tom","Sara","Ben","Mia","Jack","Lucy","Sam"]
ITEMS=["key","coin","ring","ball","card","stone","pen","cup"]
CONTS=["box","basket","jar","bag","drawer","bowl"]
def tid(w):
    ids=tok(" "+w,add_special_tokens=False).input_ids; return ids[0] if len(ids)==1 else None
CT={c:tid(c) for c in CONTS}; CONTS=[c for c in CONTS if CT[c] is not None]

def make(kind):
    name=rng.choice(NAMES); i1,i2=rng.choice(ITEMS,2,replace=False); X=rng.choice(CONTS)
    Y=rng.choice([c for c in CONTS if c!=X])
    ctx=f"{name} put the {i1} in the {X}. {name} also has a {i2}."
    if kind=="B": text=f"{ctx} The {i1} is in the {Y}"; ans=Y
    elif kind=="C": text=f"{ctx} The {i2} is in the {Y}"; ans=Y
    else: text=f"{ctx} The {i1} is in the {X}"; ans=X
    return text, ans

@torch.no_grad()
def extract(kind,n,bs=40):
    rows=[]
    items=[make(kind) for _ in range(n)]
    for i in range(0,n,bs):
        chunk=items[i:i+bs]; texts=[t for t,_ in chunk]
        enc=tok(texts,return_tensors="pt",padding=True)
        out=m(**enc,output_hidden_states=True); hs=out.hidden_states; lg=out.logits
        Ls=enc["attention_mask"].sum(1)-1                       # index of last real token (the answer word)
        for j,(t,ans) in enumerate(chunk):
            p=int(Ls[j]); aid=CT[ans]
            surp=-float(torch.log_softmax(lg[j,p-1].float(),-1)[aid])   # -logprob of answer token
            res={L:hs[L][j,p].float().numpy() for L in LAYERS}          # residual AT the answer token
            pred_ok=int(torch.argmax(lg[j,p-1,[CT[c] for c in CONTS]]))==CONTS.index(ans)
            rows.append(dict(res=res,surp=surp,pred_ok=pred_ok))
    return rows

print("extracting A/B/C ...",flush=True)
A=extract("A",N); B=extract("B",N); C=extract("C",N)
honest_acc=np.mean([r["pred_ok"] for r in A])         # on A, does the model pick the correct container?
print(f"[gate] honest accuracy on A = {honest_acc:.2f}",flush=True)

def probe_contrast(P,Q,name):
    """P=label1, Q=label0. internal probe (best layer by oof AUC) vs surprise baseline."""
    y=np.array([1]*len(P)+[0]*len(Q)); surp=np.array([r["surp"] for r in P+Q])
    best=None
    for L in LAYERS:
        X=np.array([r["res"][L] for r in P+Q]); s=oof_probe(X,y); a=auc_np(y,s)
        if best is None or a>best[1]: best=(L,a,s)
    L,ia,iscore=best; sa=auc_np(y,surp)
    # bootstrap CI for internal - surprise
    idx=np.arange(len(y)); d=[]
    for _ in range(2000):
        b=rng.choice(idx,len(y),replace=True)
        if y[b].sum()<3 or (len(b)-y[b].sum())<3: continue
        d.append(auc_np(y[b],iscore[b])-auc_np(y[b],surp[b]))
    ci=(float(np.percentile(d,2.5)),float(np.percentile(d,97.5))) if d else (float('nan'),)*2
    yp=np.random.default_rng(1).permutation(y); perm=auc_np(yp,oof_probe(np.array([r["res"][L] for r in P+Q]),yp))
    print(f"[{name}] internal AUC={ia:.3f} (L{L}) · surprise AUC={sa:.3f} · Δ={ia-sa:+.3f} CI[{ci[0]:+.3f},{ci[1]:+.3f}] · perm={perm:.3f}",flush=True)
    return dict(contrast=name,best_layer=L,internal_AUC=ia,surprise_AUC=sa,delta=ia-sa,delta_CI=ci,perm_AUC=perm)

BC=probe_contrast(B,C,"B_vs_C_conflict_isolated")     # PRIMARY: conflict beyond surprise
BA=probe_contrast(B,A,"B_vs_A_lie_vs_honest")         # context

d=BC["delta"]; ci=BC["delta_CI"]
if honest_acc<0.70 or BC["perm_AUC"]>0.60 or min(len(B),len(C))<60: verdict="BROKEN_MEASUREMENT"
elif BC["internal_AUC"]>=0.65 and d>=0.05 and ci[0]>0: verdict="SUPPORTED"
else: verdict="REFUTED"

out=dict(task="in-context conflict (knows!=says) in gpt2",honest_acc=honest_acc,N_each=N,
         PRIMARY_B_vs_C=BC,context_B_vs_A=BA,verdict=verdict)
print("\n=== NUMBERS ==="); print(json.dumps(out,indent=1))
print("\n=== VERDICT ===",verdict,f"| PRIMARY conflict-beyond-surprise: internal {BC['internal_AUC']:.3f} vs surprise {BC['surprise_AUC']:.3f} (Δ{BC['delta']:+.3f}) · perm {BC['perm_AUC']:.3f} · honest {honest_acc:.2f}")
dd=os.path.join(os.path.dirname(__file__),"..","campaigns"); os.makedirs(os.path.join(dd,"conflict"),exist_ok=True)
json.dump(out,open(os.path.join(dd,"conflict","result.json"),"w"),indent=1); print("[saved]",flush=True)
