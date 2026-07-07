#!/usr/bin/env python3
"""E1 — 'Errors = the missing intermediate' (Brick 1 ⊕ J-space).
Substrate: our gpt2 adder, 2-digit + 2-digit. The unspoken intermediate for the TENS answer
digit is the UNITS CARRY (carry1 = units sum >= 10). At the position predicting the tens digit
we ask three things:
  (behavioral)  does the model err on the tens digit MORE when a carry is needed? (carry = the hard intermediate)
  (readable)    is carry1 represented in the workspace? (probe AUC)
  (Brick-1 ⊕)   does the INTERNAL state predict the tens-digit error BETTER than output confidence,
                and is a big share of errors 'forgot-the-carry' (off-by-carry)?  -> errors are missing intermediates.
PREREG: BROKEN if adder tens-acc not in [0.4,0.95] (need errors AND competence) or permutation>0.60.
  SUPPORTED(missing-intermediate) iff err(carry)-err(no-carry) >= +0.05 AND internal error-AUC >= 0.65
  AND internal beats output-entropy AUC (CI>0) AND off-by-carry share of carry-errors >= 0.40.
"""
import os, json, numpy as np, torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
DEV="cuda" if torch.cuda.is_available() else "cpu"; torch.set_num_threads(max(1,os.cpu_count() or 4))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
tok=GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token=tok.eos_token
model=GPT2LMHeadModel.from_pretrained("gpt2").to(DEV)
rng=np.random.default_rng(0); HS=8; BATCH=16
DIG=[tok(" "+str(d),add_special_tokens=False).input_ids[0] for d in range(10)]
def sd(n): return " ".join(list(str(n)))
def two(): return int(rng.integers(10,100))
def gen(n): return [(two(),two()) for _ in range(n)]

def fit_logreg(X,y,it=400,lr=0.5,l2=1.0):
    n,d=X.shape; Xb=np.concatenate([X,np.ones((n,1))],1); w=np.zeros(d+1)
    for _ in range(it): p=1/(1+np.exp(-(Xb@w))); g=Xb.T@(p-y)/n; g[:-1]+=l2*w[:-1]/n; w-=lr*g
    return w
def prd(w,X): return 1/(1+np.exp(-(np.concatenate([X,np.ones((len(X),1))],1)@w)))
def _rk(a):
    a=np.asarray(a,float); o=np.argsort(a,kind='mergesort'); sa=a[o]; n=len(a); rs=np.empty(n); i=0
    while i<n:
        j=i
        while j+1<n and sa[j+1]==sa[i]: j+=1
        rs[i:j+1]=(i+j)/2+1; i=j+1
    r=np.empty(n); r[o]=rs; return r
def auc(y,s):
    y=np.asarray(y); p=(y==1); a=int(p.sum()); b=int((~p).sum())
    if a==0 or b==0: return float('nan')
    r=_rk(s); return float((r[p].sum()-a*(a+1)/2)/(a*b))
def oof(X,y):
    idx=np.arange(len(y)); np.random.default_rng(0).shuffle(idx); fo=np.array_split(idx,5); o=np.zeros(len(y))
    for f in range(5):
        te=fo[f]; tr=np.concatenate([fo[g] for g in range(5) if g!=f])
        mu=X[tr].mean(0); sd_=X[tr].std(0)+1e-6; w=fit_logreg((X[tr]-mu)/sd_,y[tr]); o[te]=prd(w,(X[te]-mu)/sd_)
    return o

def bt(probs):
    seqs=[];masks=[]
    for a,b in probs:
        pj=tok(f"{sd(a)} + {sd(b)} =",add_special_tokens=False).input_ids; aj=tok(" "+sd(a+b),add_special_tokens=False).input_ids
        seqs.append(pj+aj+[tok.eos_token_id]); masks.append([-100]*len(pj)+aj+[tok.eos_token_id])
    L=max(len(s) for s in seqs); X=torch.full((len(seqs),L),tok.eos_token_id,dtype=torch.long)
    Y=torch.full((len(seqs),L),-100,dtype=torch.long); A=torch.zeros((len(seqs),L),dtype=torch.long)
    for i,(s,m) in enumerate(zip(seqs,masks)): X[i,:len(s)]=torch.tensor(s);Y[i,:len(m)]=torch.tensor(m);A[i,:len(s)]=1
    return X.to(DEV),Y.to(DEV),A.to(DEV)

# ---- train adder to partial competence (target tens-err ~0.3) ----
print(f"[train] adder on {DEV} ...",flush=True)
opt=torch.optim.AdamW(model.parameters(),lr=5e-4); model.train()
@torch.no_grad()
def tens_acc(n=200):
    model.eval(); ok=t=0
    for a,b in gen(n):
        pj=tok(f"{sd(a)} + {sd(b)} =",add_special_tokens=False).input_ids; ans=[int(c) for c in str(a+b)]
        ids=torch.tensor(pj+[DIG[d] for d in ans],device=DEV).unsqueeze(0); lg=model(ids).logits[0]; start=len(pj)
        k=len(ans)-2  # tens digit index
        ok+= 1 if int(torch.argmax(lg[start+k-1,DIG]))==ans[k] else 0; t+=1
    model.train(); return ok/t
for step in range(1,1801):
    X,Y,A=bt(gen(BATCH)); out=model(input_ids=X,attention_mask=A,labels=Y); out.loss.backward(); opt.step(); opt.zero_grad()
    if step%150==0:
        ta=tens_acc(); print(f"[train] step {step} tens_acc={ta:.3f}",flush=True)
        if ta>=0.70: break
del opt; torch.cuda.empty_cache() if DEV=="cuda" else None

# ---- collect at the TENS answer digit ----
@torch.no_grad()
def collect(n,bs=40):
    model.eval(); rows=[]; probs=gen(n)
    for i in range(0,n,bs):
        ch=probs[i:i+bs]; seqs=[]; meta=[]
        for a,b in ch:
            pj=tok(f"{sd(a)} + {sd(b)} =",add_special_tokens=False).input_ids; ans=[int(c) for c in str(a+b)]
            seqs.append(pj+[DIG[d] for d in ans]); meta.append((len(pj),ans,(a%10)+(b%10)>=10,a,b))
        Lm=max(len(s) for s in seqs); X=torch.full((len(seqs),Lm),tok.eos_token_id,dtype=torch.long); M=torch.zeros((len(seqs),Lm),dtype=torch.long)
        for j,s in enumerate(seqs): X[j,:len(s)]=torch.tensor(s); M[j,:len(s)]=1
        X=X.to(DEV); M=M.to(DEV); o=model(input_ids=X,attention_mask=M,output_hidden_states=True); hs=o.hidden_states[HS]; lg=o.logits
        for j,(start,ans,carry1,a,b) in enumerate(meta):
            k=len(ans)-2; p=start+k                      # position of tens digit
            z=lg[j,p-1,DIG].float(); pr=torch.softmax(z,-1); pred=int(torch.argmax(z)); true=ans[k]
            err=1 if pred!=true else 0
            offby=1 if (err and carry1 and pred==(true-1)%10) else 0   # 'forgot the carry' signature
            rows.append(dict(res=hs[j,p].float().cpu().numpy(),err=err,carry1=int(carry1),
                             ent=float(-(pr*torch.log(pr+1e-12)).sum()),offby=offby))
    return rows

R=collect(1000)
err=np.array([r["err"] for r in R]); carry=np.array([r["carry1"] for r in R]); ent=np.array([r["ent"] for r in R])
X=np.array([r["res"] for r in R]); tens_err=err.mean()
e_carry=err[carry==1].mean(); e_nocarry=err[carry==0].mean()
print(f"[data] N={len(R)} tens_err={tens_err:.3f} | err|carry={e_carry:.3f} err|no-carry={e_nocarry:.3f} (Δ={e_carry-e_nocarry:+.3f})",flush=True)
carry_auc=auc(carry,oof(X,carry))                                   # is the carry intermediate readable?
int_err=oof(X,err); int_auc=auc(err,int_err); out_auc=auc(err,ent)  # Brick-1: internal vs output at flagging errors
idx=np.arange(len(err)); dd=[]
for _ in range(2000):
    b=rng.choice(idx,len(err),replace=True)
    if err[b].sum()<3 or (len(b)-err[b].sum())<3: continue
    dd.append(auc(err[b],int_err[b])-auc(err[b],ent[b]))
ci=(float(np.percentile(dd,2.5)),float(np.percentile(dd,97.5))) if dd else (float('nan'),)*2
yp=np.random.default_rng(1).permutation(err); perm=auc(yp,oof(X,yp))
carry_errs=[r for r in R if r["err"] and r["carry1"]]; offby_share=(np.mean([r["offby"] for r in carry_errs]) if carry_errs else float('nan'))
print(f"[readable] carry-probe AUC={carry_auc:.3f}",flush=True)
print(f"[Brick1] internal error-AUC={int_auc:.3f} vs output={out_auc:.3f} Δ={int_auc-out_auc:+.3f} CI[{ci[0]:+.3f},{ci[1]:+.3f}] perm={perm:.3f}",flush=True)
print(f"[off-by-carry] share of carry-errors that = forgot-the-carry: {offby_share:.2f}",flush=True)

if not(0.05<=tens_err<=0.60) or perm>0.60: verdict="BROKEN_MEASUREMENT"
elif (e_carry-e_nocarry)>=0.05 and int_auc>=0.65 and ci[0]>0 and offby_share>=0.40: verdict="SUPPORTED"
else: verdict="REFUTED"
out=dict(exp="E1 missing intermediate (carry)",device=DEV,N=len(R),tens_err=tens_err,
         err_carry=e_carry,err_nocarry=e_nocarry,carry_gap=e_carry-e_nocarry,carry_probe_AUC=carry_auc,
         internal_err_AUC=int_auc,output_err_AUC=out_auc,delta=int_auc-out_auc,delta_CI=ci,perm_AUC=perm,
         offby_carry_share=offby_share,verdict=verdict)
print("\n=== NUMBERS ==="); print(json.dumps(out,indent=1))
print("\n=== VERDICT ===",verdict,f"| carry-gap {e_carry-e_nocarry:+.3f} · carry readable {carry_auc:.2f} · internal {int_auc:.2f} vs output {out_auc:.2f} (Δ{int_auc-out_auc:+.2f}) · off-by-carry {offby_share:.2f}")
d=os.path.join(os.path.dirname(__file__),"..","campaigns","workspace"); os.makedirs(d,exist_ok=True)
json.dump(out,open(os.path.join(d,"E1_result.json"),"w"),indent=1); print("[saved]",flush=True)
