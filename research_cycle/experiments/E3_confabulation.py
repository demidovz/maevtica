#!/usr/bin/env python3
"""E3 — 'Confabulation = empty workspace' (local hallucination-detector dry run).
Clean substrate that avoids the real-vs-fake-word confound: a SYNTHETIC memory. Teach gpt2 a
random table 'person -> city' but HIDE some persons from training. Then:
  GROUNDED = trained persons (model retrieves the right city).
  CONFAB   = held-out persons (model confidently names SOME city, but has no grounded knowledge).
All persons are the same kind of token; grounded/confab is a random person split -> no surface
confound, and no fixed token->label. Question: does an INTERNAL signal flag confabulation
(held-out) better than the model's OUTPUT confidence? (the Brick-1 / cloud-hallucination claim).

PREREG: BROKEN if grounded retrieval acc < 0.60 (didn't memorize) or person-grouped permutation
AUC > 0.62. SUPPORTED iff internal AUC >= 0.65 AND internal beats output-confidence AUC (CI>0).
Report whether confabulation is CONFIDENT (output entropy on confab vs grounded) — the money case
is confident confabulation, where internal must win. AUC~1.0 => adversarial suspicion.
"""
import os, json, numpy as np, torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
DEV="cpu"; torch.set_num_threads(max(1,os.cpu_count() or 4))
tok=GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token=tok.eos_token
model=GPT2LMHeadModel.from_pretrained("gpt2").to(DEV)
rng=np.random.default_rng(0); HS=8; BATCH=16

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
def oof_grouped(X,y,groups):
    """5-fold, splitting by GROUP (person) so no person is in both train and test."""
    gs=np.array(groups); uniq=np.array(sorted(set(gs))); gi=np.random.default_rng(0).permutation(len(uniq))
    folds=np.array_split(uniq[gi],5); o=np.zeros(len(y))
    for f in range(5):
        teg=set(folds[f].tolist()); te=np.array([i for i in range(len(y)) if gs[i] in teg]); tr=np.array([i for i in range(len(y)) if gs[i] not in teg])
        mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; w=fit_logreg((X[tr]-mu)/sd,y[tr]); o[te]=prd(w,(X[te]-mu)/sd)
    return o

NAMES=("Anna Tom Sara Ben Mia Jack Lucy Sam Kate Paul Emma Mark Rosa Nick Ella Dave Ivan Nina Leo Rita "
       "Adam Beth Carl Dora Erik Faye Gary Hana Igor Jane Karl Lena Milo Nora Omar Pia Ravi Sofi Theo Uma "
       "Vera Will Xena York Zoe Alex Bella Cody Dana Emil Fred Gina Hugo Iris Jon Kira Liam Maya Noel Opal "
       "Pete Quin Remy Suki Troy Umar Vlad Wade Yara Zack Abby Bram Cleo Drew Enzo Fira Glen Hope Inez Jude "
       "Kai Lars Moss Nell Otis Page Rex Skye Tara Uri Vin Wren Yuki Zane Ada Bo Cira Dex Evan Finn").split()
CITIES=["Paris","London","Rome","Berlin","Tokyo","Madrid","Cairo","Boston"]
def tid(w):
    ids=tok(" "+w,add_special_tokens=False).input_ids; return ids[0] if len(ids)==1 else None
CT={c:tid(c) for c in CITIES}; CITIES=[c for c in CITIES if CT[c] is not None]; CID=[CT[c] for c in CITIES]
home={p:CITIES[int(rng.integers(len(CITIES)))] for p in NAMES}
TRAINED=NAMES[:75]; HELD=NAMES[75:]
TEMPL=["{p} lives in","The home of {p} is in","{p} resides in"]
print(f"persons trained={len(TRAINED)} held={len(HELD)} cities={len(CITIES)}",flush=True)

def bt(pairs):  # pairs of (person, city) for "{p} lives in {c}."
    seqs=[];masks=[]
    for p,c in pairs:
        pj=tok(f"{p} lives in",add_special_tokens=False).input_ids; cj=tok(" "+c,add_special_tokens=False).input_ids
        seqs.append(pj+cj+[tok.eos_token_id]); masks.append([-100]*len(pj)+cj+[tok.eos_token_id])
    L=max(len(s) for s in seqs); X=torch.full((len(seqs),L),tok.eos_token_id,dtype=torch.long)
    Y=torch.full((len(seqs),L),-100,dtype=torch.long); A=torch.zeros((len(seqs),L),dtype=torch.long)
    for i,(s,m) in enumerate(zip(seqs,masks)): X[i,:len(s)]=torch.tensor(s);Y[i,:len(m)]=torch.tensor(m);A[i,:len(s)]=1
    return X.to(DEV),Y.to(DEV),A.to(DEV)

@torch.no_grad()
def grounded_acc():
    model.eval(); ok=0
    for p in TRAINED:
        lg=model(tok(f"{p} lives in",return_tensors="pt").input_ids.to(DEV)).logits[0,-1]
        ok+= 1 if CITIES[int(torch.argmax(lg[CID]))]==home[p] else 0
    model.train(); return ok/len(TRAINED)

print("[train] memorizing person->city ...",flush=True)
opt=torch.optim.AdamW(model.parameters(),lr=5e-4); model.train()
for step in range(1,1201):
    batch=[(p,home[p]) for p in [TRAINED[int(rng.integers(len(TRAINED)))] for _ in range(BATCH)]]
    X,Y,A=bt(batch); out=model(input_ids=X,attention_mask=A,labels=Y); out.loss.backward(); opt.step(); opt.zero_grad()
    if step%150==0:
        ga=grounded_acc(); print(f"[train] step {step} grounded_acc={ga:.3f}",flush=True)
        if ga>=0.90: break
del opt

@torch.no_grad()
def collect():
    model.eval(); rows=[]
    persons=[(p,0) for p in TRAINED]+[(p,1) for p in HELD]     # label 1 = confabulation (held-out)
    for p,lab in persons:
        for t in TEMPL:
            text=t.format(p=p); enc=tok(text,return_tensors="pt")
            o=model(enc.input_ids.to(DEV),output_hidden_states=True); h=o.hidden_states[HS][0,-1]; lg=o.logits[0,-1]
            cl=lg[CID].float(); pr=torch.softmax(cl,-1)
            top_correct=(CITIES[int(torch.argmax(cl))]==home[p])
            rows.append(dict(person=p,confab=lab,res=h.float().cpu().numpy(),
                             ent=float(-(pr*torch.log(pr+1e-12)).sum()),maxp=float(pr.max()),correct=int(top_correct)))
    return rows

R=collect()
grounded=[r for r in R if r["confab"]==0]; confab=[r for r in R if r["confab"]==1]
gacc=np.mean([r["correct"] for r in grounded])
# keep grounded = trained AND retrieved correctly (truly grounded) vs confab = held-out
G=[r for r in grounded if r["correct"]]; C=confab
y=np.array([0]*len(G)+[1]*len(C)); X=np.array([r["res"] for r in G+C]); grp=[r["person"] for r in G+C]
ent=np.array([r["ent"] for r in G+C]); maxp=np.array([r["maxp"] for r in G+C])
ent_g=np.mean([r["ent"] for r in G]); ent_c=np.mean([r["ent"] for r in C])
print(f"[data] grounded(correct) n={len(G)} confab n={len(C)} · grounded retrieval acc={gacc:.2f}",flush=True)
print(f"[confidence] output entropy: grounded={ent_g:.2f} confab={ent_c:.2f} (close => confident confabulation)",flush=True)
int_s=oof_grouped(X,y,grp); iA=auc(y,int_s)
oA=auc(y,ent)                                   # output confidence (entropy) as confab detector
idx=np.arange(len(y)); dd=[]
for _ in range(2000):
    b=rng.choice(idx,len(y),replace=True)
    if y[b].sum()<3 or (len(b)-y[b].sum())<3: continue
    dd.append(auc(y[b],int_s[b])-auc(y[b],ent[b]))
ci=(float(np.percentile(dd,2.5)),float(np.percentile(dd,97.5))) if dd else (float('nan'),)*2
yp=np.random.default_rng(1).permutation(y); perm=auc(yp,oof_grouped(X,yp,grp))
print(f"[detect confab] internal AUC={iA:.3f} vs output-conf AUC={oA:.3f} Δ={iA-oA:+.3f} CI[{ci[0]:+.3f},{ci[1]:+.3f}] perm={perm:.3f}",flush=True)

if gacc<0.60 or perm>0.62: verdict="BROKEN_MEASUREMENT"
elif iA>=0.65 and (iA-oA)>=0.03 and ci[0]>0: verdict="SUPPORTED"
else: verdict="REFUTED"
out=dict(exp="E3 confabulation = empty workspace (synthetic memory)",grounded_acc=gacc,n_grounded=len(G),n_confab=len(C),
         output_entropy_grounded=ent_g,output_entropy_confab=ent_c,internal_AUC=iA,output_conf_AUC=oA,
         delta=iA-oA,delta_CI=ci,perm_AUC=perm,verdict=verdict)
print("\n=== NUMBERS ==="); print(json.dumps(out,indent=1))
print("\n=== VERDICT ===",verdict,f"| internal {iA:.3f} vs output {oA:.3f} (Δ{iA-oA:+.3f}) · confident-confab entropy g={ent_g:.2f} c={ent_c:.2f} · perm {perm:.3f} · grounded_acc {gacc:.2f}")
d=os.path.join(os.path.dirname(__file__),"..","campaigns","workspace"); os.makedirs(d,exist_ok=True)
json.dump(out,open(os.path.join(d,"E3_result.json"),"w"),indent=1); print("[saved]",flush=True)
