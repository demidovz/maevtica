#!/usr/bin/env python3
"""'Учитель-провидец', честная версия — INTERLEAVE. The restrict-version
(teach_curriculum.py) showed that teaching ONLY the inner-flagged hard spots HURTS
(catastrophic narrowing). This is the fair test of the boss's idea: keep BROAD practice
but add EXTRA EMPHASIS on the inner-flagged (confidently-confused) spots. Does emphasis
WITHOUT restriction beat plain broad practice, at EQUAL budget?

Setup identical to teach_curriculum (gpt2 addition 0-99, partial-competence child C0,
Brick-1 inner probe). Arms differ only in how the K teaching problems are COMPOSED:
  broad     : K random problems (pure broad practice; last run's winner).
  inner0.3  : 30% top-inner-flagged + 70% random  (light emphasis on inward confusion).
  inner0.6  : 60% top-inner-flagged + 40% random  (heavy emphasis; dose context).
  output0.3 : 30% top-output-entropy + 70% random (light emphasis via the VOICE; inside-vs-listening).
Equal budget: same K, same steps, same gentle lr; 3 seeds; fixed held-out test.

PREREGISTERED RULE (frozen before running):
  * BROKEN_MEASUREMENT if C0 digit-err not in [0.30,0.70], OR probe oof AUC < 0.60, OR
    broad-arm did not beat C0 by >=0.5pt (teaching inert -> can't compare).
  * PRIMARY = inner0.3_acc - broad_acc (does light inner-emphasis beat plain breadth?).
  * SUPPORTED iff mean(inner0.3 - broad) >= +1.0 pt AND inner0.3 beats broad in
    >= ceil(0.75*SEEDS) seeds AND inner0.3_acc >= output0.3_acc (inside >= listening).
  * REFUTED otherwise. (inner0.6 reported as dose context; single primary dose = 0.3,
    chosen a priori because restrict/1.0 already hurt, so lighter emphasis is the best bet.)
"""
import os, json, numpy as np, torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

def fit_logreg(X,y,iters=400,lr=0.5,l2=1.0):
    n,d=X.shape; Xb=np.concatenate([X,np.ones((n,1))],1); w=np.zeros(d+1)
    for _ in range(iters):
        p=1/(1+np.exp(-(Xb@w))); g=Xb.T@(p-y)/n; g[:-1]+=l2*w[:-1]/n; w-=lr*g
    return w
def pred_logreg(w,X): return 1/(1+np.exp(-(np.concatenate([X,np.ones((len(X),1))],1)@w)))
def _rankdata(a):
    a=np.asarray(a,float); order=np.argsort(a,kind='mergesort'); sa=a[order]; n=len(a); rs=np.empty(n); i=0
    while i<n:
        j=i
        while j+1<n and sa[j+1]==sa[i]: j+=1
        rs[i:j+1]=(i+j)/2+1; i=j+1
    r=np.empty(n); r[order]=rs; return r
def auc_np(y,s):
    y=np.asarray(y); pos=(y==1); npos=int(pos.sum()); nneg=int((~pos).sum())
    if npos==0 or nneg==0: return float('nan')
    r=_rankdata(np.asarray(s,float)); return float((r[pos].sum()-npos*(npos+1)/2)/(npos*nneg))

DEV="cuda" if torch.cuda.is_available() else "cpu"
SMOKE=os.environ.get("SMOKE")=="1"
POOL_N=200 if SMOKE else 450; K=64 if SMOKE else 260; TEACH=20 if SMOKE else 50
SEEDS=1 if SMOKE else 3; TEST_N=100 if SMOKE else 150; PROBE_N=100 if SMOKE else 300
HS=8; BATCH=16; LR=5e-4; LR_TEACH=1e-4
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
print(f"device={DEV} SMOKE={SMOKE} | POOL={POOL_N} K={K} TEACH={TEACH}@lr{LR_TEACH} SEEDS={SEEDS}",flush=True)

tok=GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token=tok.eos_token
DIG=[tok(" "+str(d),add_special_tokens=False).input_ids[0] for d in range(10)]
def sd(n): return " ".join(list(str(n)))
def gen(n,rng): return [(int(rng.integers(0,100)),int(rng.integers(0,100))) for _ in range(n)]
def batch_tensors(probs):
    seqs=[];masks=[]
    for a,b in probs:
        pj=tok(f"{sd(a)} + {sd(b)} =",add_special_tokens=False).input_ids
        aj=tok(" "+sd(a+b),add_special_tokens=False).input_ids
        seqs.append(pj+aj+[tok.eos_token_id]); masks.append([-100]*len(pj)+aj+[tok.eos_token_id])
    L=max(len(s) for s in seqs)
    X=torch.full((len(seqs),L),tok.eos_token_id,dtype=torch.long)
    Y=torch.full((len(seqs),L),-100,dtype=torch.long); A=torch.zeros((len(seqs),L),dtype=torch.long)
    for i,(s,m) in enumerate(zip(seqs,masks)):
        X[i,:len(s)]=torch.tensor(s);Y[i,:len(m)]=torch.tensor(m);A[i,:len(s)]=1
    return X.to(DEV),Y.to(DEV),A.to(DEV)
def train_on(model,probs,steps,seed):
    opt=torch.optim.AdamW(model.parameters(),lr=LR_TEACH); model.train()
    r=np.random.default_rng(seed)
    for _ in range(steps):
        batch=[probs[i] for i in r.integers(0,len(probs),BATCH)]
        X,Y,A=batch_tensors(batch); out=model(input_ids=X,attention_mask=A,labels=Y)
        out.loss.backward(); opt.step(); opt.zero_grad()
    return model
def _pad_batch(chunk):
    """right-pad prompt+answer token seqs; return X, mask, and per-sample (start, ans_digits)."""
    seqs=[]; meta=[]
    for a,b in chunk:
        pj=tok(f"{sd(a)} + {sd(b)} =",add_special_tokens=False).input_ids
        ans=[int(c) for c in str(a+b)]
        seqs.append(pj+[DIG[d] for d in ans]); meta.append((len(pj),ans))
    L=max(len(s) for s in seqs)
    X=torch.full((len(seqs),L),tok.eos_token_id,dtype=torch.long); M=torch.zeros((len(seqs),L),dtype=torch.long)
    for j,s in enumerate(seqs): X[j,:len(s)]=torch.tensor(s); M[j,:len(s)]=1
    return X.to(DEV),M.to(DEV),meta
@torch.no_grad()
def eval_digit_acc(model,probs,bs=64):
    model.eval(); ok=tot=0
    for i in range(0,len(probs),bs):
        X,M,meta=_pad_batch(probs[i:i+bs]); lg=model(input_ids=X,attention_mask=M).logits
        for j,(start,ans) in enumerate(meta):
            for k,d in enumerate(ans): ok+= 1 if int(torch.argmax(lg[j,start+k-1,DIG]))==d else 0; tot+=1
    return ok/tot
@torch.no_grad()
def collect_digits(model,probs,bs=64):
    model.eval(); rows=[]
    for i in range(0,len(probs),bs):
        X,M,meta=_pad_batch(probs[i:i+bs])
        out=model(input_ids=X,attention_mask=M,output_hidden_states=True); lg=out.logits; hs=out.hidden_states[HS]
        for j,(start,ans) in enumerate(meta):
            for k,d in enumerate(ans):
                pos=start+k; z=lg[j,pos-1,DIG].float(); p=torch.softmax(z,-1)
                rows.append((i+j,hs[j,pos-1].float().cpu().numpy(),float(-(p*torch.log(p+1e-12)).sum()),1 if int(torch.argmax(z))!=d else 0))
    return rows
def per_problem(rows,vals,nprob):
    acc=np.zeros(nprob); cnt=np.zeros(nprob)
    for (pi,_,_,_),v in zip(rows,vals): acc[pi]+=v; cnt[pi]+=1
    return acc/np.maximum(cnt,1)
def build_set(pool,sig,rho,K,rng):
    """K problems = top-(rho*K) by sig (emphasis) + the rest drawn broadly at random."""
    if rho<=0: return [pool[i] for i in rng.choice(len(pool),K,replace=False)]
    kf=int(round(rho*K)); order=np.argsort(-sig)
    flagged=list(order[:kf]); broad=list(rng.choice(order[kf:],K-kf,replace=False))
    return [pool[i] for i in flagged+broad]

# ---- child C0 (adaptive to partial competence) ----
print("[C0] training partial-competence child...",flush=True)
model=GPT2LMHeadModel.from_pretrained("gpt2").to(DEV)
C0_MAX=(80 if SMOKE else 1500); C0_EVAL=(40 if SMOKE else 100); TARGET=0.55
TEST=gen(TEST_N,np.random.default_rng(999))
opt=torch.optim.AdamW(model.parameters(),lr=LR); model.train()
_r=np.random.default_rng(1); _pool=gen(20000,np.random.default_rng(0))
for step in range(1,C0_MAX+1):
    batch=[_pool[i] for i in _r.integers(0,len(_pool),BATCH)]
    X,Y,A=batch_tensors(batch); out=model(input_ids=X,attention_mask=A,labels=Y)
    out.loss.backward(); opt.step(); opt.zero_grad()
    if step%C0_EVAL==0:
        e=1-eval_digit_acc(model,TEST); model.train(); print(f"[C0] step {step} err={e:.3f}",flush=True)
        if e<=TARGET: break
del opt
if DEV=="cuda": torch.cuda.empty_cache()
C0={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
c0_acc=eval_digit_acc(model,TEST); c0_err=1-c0_acc
print(f"[C0] final digit acc={c0_acc:.3f} (err={c0_err:.3f})",flush=True)

# ---- inner probe ----
prows=collect_digits(model,gen(PROBE_N,np.random.default_rng(7)))
PX=np.array([r[1] for r in prows]); Py=np.array([r[3] for r in prows]).astype(float)
mu=PX.mean(0); sdv=PX.std(0)+1e-6; W=fit_logreg((PX-mu)/sdv,Py)
idx=np.arange(len(Py)); np.random.default_rng(0).shuffle(idx); folds=np.array_split(idx,5); oof=np.zeros(len(Py))
for f in range(5):
    te=folds[f]; tr=np.concatenate([folds[g] for g in range(5) if g!=f])
    m2=PX[tr].mean(0); s2=PX[tr].std(0)+1e-6; wf=fit_logreg((PX[tr]-m2)/s2,Py[tr]); oof[te]=pred_logreg(wf,(PX[te]-m2)/s2)
probe_auc=auc_np(Py,oof); print(f"[probe] oof AUC={probe_auc:.3f}",flush=True)

# ---- arms: same budget K, different COMPOSITION ----
ARMS=[("broad",None,0.0),("inner0.3","inner",0.3),("inner0.6","inner",0.6),("output0.3","output",0.3)]
per_seed={n:[] for n,_,_ in ARMS}
for s in range(SEEDS):
    model.load_state_dict(C0)
    pool=gen(POOL_N,np.random.default_rng(100+s)); rows=collect_digits(model,pool)
    ent=np.array([r[2] for r in rows]); pinner=pred_logreg(W,(np.array([r[1] for r in rows])-mu)/sdv)
    sig={"inner":per_problem(rows,pinner,POOL_N),"output":per_problem(rows,ent,POOL_N)}
    for ai,(name,key,rho) in enumerate(ARMS):
        pick=build_set(pool,sig.get(key),rho,K,np.random.default_rng(700+s*10+ai))
        model.load_state_dict(C0); train_on(model,pick,TEACH,seed=200+s)
        per_seed[name].append(eval_digit_acc(model,TEST))
        if DEV=="cuda": torch.cuda.empty_cache()
    print(f"[seed {s}] "+" ".join(f"{n}={per_seed[n][-1]*100:.2f}" for n,_,_ in ARMS),flush=True)

mean={n:float(np.mean(per_seed[n])) for n,_,_ in ARMS}; std={n:float(np.std(per_seed[n])) for n,_,_ in ARMS}
prim=[per_seed["inner0.3"][i]-per_seed["broad"][i] for i in range(SEEDS)]
ivo =[per_seed["inner0.3"][i]-per_seed["output0.3"][i] for i in range(SEEDS)]
primary=float(np.mean(prim))*100; n_pos=sum(1 for d in prim if d>0)
teach_worked=(mean["broad"]-c0_acc)*100; need=int(np.ceil(0.75*SEEDS))
if not(0.30<=c0_err<=0.70) or probe_auc<0.60 or teach_worked<0.5: verdict="BROKEN_MEASUREMENT"
elif primary>=1.0 and n_pos>=need and mean["inner0.3"]>=mean["output0.3"]: verdict="SUPPORTED"
else: verdict="REFUTED"

out=dict(task="addition 0-99, INTERLEAVE teach at equal budget K",device=DEV,
         C0_start_acc=c0_acc,C0_digit_err=c0_err,probe_oof_AUC=probe_auc,K=K,TEACH_steps=TEACH,SEEDS=SEEDS,
         mean_acc=mean,std_acc=std,improvement_over_C0_pts={n:(mean[n]-c0_acc)*100 for n,_,_ in ARMS},
         teaching_moved_needle_pts=teach_worked,PRIMARY_inner0_3_minus_broad_pts=primary,
         inner0_3_minus_broad_per_seed=[d*100 for d in prim],seeds_inner_beats_broad=f"{n_pos}/{SEEDS}",
         inner0_3_minus_output0_3_per_seed=[d*100 for d in ivo],verdict=verdict)
print("\n=== NUMBERS ==="); print(json.dumps(out,indent=1))
print("\n=== VERDICT ===",verdict,f"| PRIMARY inner0.3-broad={primary:+.2f}pt ({n_pos}/{SEEDS}) · "
      f"broad+{teach_worked:+.2f} vs C0 · probe AUC={probe_auc:.2f} · C0 err={c0_err:.2f}")
if not SMOKE:
    d=os.path.join(os.path.dirname(__file__),"..","campaigns","teach-curriculum"); os.makedirs(d,exist_ok=True)
    dst=os.path.join(d,"interleave_result.json"); json.dump(out,open(dst,"w"),indent=1); print("[saved]",os.path.abspath(dst),flush=True)
