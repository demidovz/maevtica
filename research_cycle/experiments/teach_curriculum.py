#!/usr/bin/env python3
"""'Учитель-провидец' — does teaching guided by the INNER signal beat blind /
output-guided teaching, at EQUAL budget? Causal/developmental test of Brick 1:
we can SEE where the child is confidently confused; does teaching THOSE spots grow
it faster than teaching random spots (or the spots where its own voice hesitates)?

Setup: gpt2, addition of two numbers 0-99. Train a partial-competence child C0
(uneven: good at easy sums, a 'confident fool' on carries). Then, at EQUAL budget
(same K teaching problems, same steps), teach 4 fresh copies of C0, differing ONLY
in WHICH problems are chosen:
  inner  : top-K by an internal error-probe (resid -> p(wrong)), trained on a
           separate labeled set (Brick-1 machinery). Where the child is inwardly-wrong.
  output : top-K by the child's own output entropy (where its voice hesitates).
  random : K random problems (blind teaching).
  oracle : top-K by the child's ACTUAL errors (perfect diagnostician -> ceiling/canary).
Measure: digit-accuracy on a FIXED held-out test after teaching. Repeat SEEDS times
(fresh pool + training order); test set fixed for comparability -> paired arms.

PREREGISTERED RULE (frozen before running):
  * BROKEN_MEASUREMENT if C0 digit-error not in [0.30,0.70] (wrong regime), OR
    probe out-of-fold AUC < 0.60 (inner signal invalid), OR random-arm did not beat C0
    by >=0.5pt (teaching budget too small to move the needle -> can't compare curricula).
    (NOTE: we do NOT require oracle>random. "Teach the hardest" is a HYPOTHESIS about
    curriculum, not an instrument check -- hardest-first can hurt; the child may learn best
    at the boundary of its ability (ZPD), which is where the *voice* hesitates, not where it
    is confidently-confused. oracle is kept as an informative reference arm, not a gate.)
  * PRIMARY = inner_acc - output_acc (Brick-1's prediction: looking inside beats
    listening, because in the confident-fool spots the voice is blind).
  * SUPPORTED iff mean(inner-output) >= +1.0 pt AND inner beats output in >= ceil(0.75*SEEDS)
    seeds AND inner_acc >= random_acc. REFUTED otherwise.
"""
import os, copy, json, numpy as np, torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# --- self-contained probe (no sklearn: gpu-venv lacks it) ---
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
POOL_N   = 200 if SMOKE else 700
K        = 64  if SMOKE else 384
TEACH    = 20  if SMOKE else 50
SEEDS    = 1   if SMOKE else 3
TEST_N   = 100 if SMOKE else 250
PROBE_N  = 100 if SMOKE else 400
HS=8; BATCH=16; LR=5e-4; LR_TEACH=1e-4         # LR: build C0 fast; LR_TEACH: gentle so teaching REFINES not forgets
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")  # small BATCH: RTX 3050 has 3.68 GiB
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
    opt=torch.optim.AdamW(model.parameters(),lr=LR_TEACH); model.train()   # gentle: refine, don't forget
    r=np.random.default_rng(seed)
    for _ in range(steps):
        batch=[probs[i] for i in r.integers(0,len(probs),BATCH)]
        X,Y,A=batch_tensors(batch)
        out=model(input_ids=X,attention_mask=A,labels=Y)
        out.loss.backward(); opt.step(); opt.zero_grad()
    return model

@torch.no_grad()
def eval_digit_acc(model,probs):
    model.eval(); ok=tot=0
    for a,b in probs:
        pj=tok(f"{sd(a)} + {sd(b)} =",add_special_tokens=False).input_ids
        ans=[int(c) for c in str(a+b)]; aj=[DIG[d] for d in ans]
        ids=torch.tensor(pj+aj,device=DEV).unsqueeze(0); lg=model(ids).logits[0]; start=len(pj)
        for k,d in enumerate(ans):
            ok+= 1 if int(torch.argmax(lg[start+k-1,DIG]))==d else 0; tot+=1
    return ok/tot

@torch.no_grad()
def collect_digits(model,probs):
    """per answer-digit: (prob_idx, resid[HS], entropy, err)."""
    model.eval(); rows=[]
    for pi,(a,b) in enumerate(probs):
        pj=tok(f"{sd(a)} + {sd(b)} =",add_special_tokens=False).input_ids
        ans=[int(c) for c in str(a+b)]; aj=[DIG[d] for d in ans]
        ids=torch.tensor(pj+aj,device=DEV).unsqueeze(0)
        out=model(ids,output_hidden_states=True); lg=out.logits[0]; hs=out.hidden_states[HS][0]; start=len(pj)
        for k,d in enumerate(ans):
            pos=start+k; z=lg[pos-1,DIG].float(); p=torch.softmax(z,-1)
            rows.append((pi, hs[pos-1].float().cpu().numpy(),
                         float(-(p*torch.log(p+1e-12)).sum()), 1 if int(torch.argmax(z))!=d else 0))
    return rows

def per_problem(rows,vals,nprob):
    """mean of a per-digit array over each problem's digits."""
    acc=np.zeros(nprob); cnt=np.zeros(nprob)
    for (pi,_,_,_),v in zip(rows,vals): acc[pi]+=v; cnt[pi]+=1
    return acc/np.maximum(cnt,1)

# ---- build the child C0 (adaptive: stop at partial competence; ONE model on GPU) ----
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
C0={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}   # start state, held on CPU
c0_acc=eval_digit_acc(model,TEST); c0_err=1-c0_acc
print(f"[C0] final digit acc={c0_acc:.3f} (err={c0_err:.3f}; want 0.30-0.70)",flush=True)

# ---- inner probe (Brick-1): train on a SEPARATE labeled set, measure oof AUC ----
prows=collect_digits(model,gen(PROBE_N,np.random.default_rng(7)))   # model still holds C0
PX=np.array([r[1] for r in prows]); Py=np.array([r[3] for r in prows]).astype(float)
mu=PX.mean(0); sdv=PX.std(0)+1e-6; W=fit_logreg((PX-mu)/sdv,Py)          # full-fit probe for scoring pools
idx=np.arange(len(Py)); np.random.default_rng(0).shuffle(idx); folds=np.array_split(idx,5); oof=np.zeros(len(Py))
for f in range(5):
    te=folds[f]; tr=np.concatenate([folds[g] for g in range(5) if g!=f])
    m2=PX[tr].mean(0); s2=PX[tr].std(0)+1e-6; wf=fit_logreg((PX[tr]-m2)/s2,Py[tr])
    oof[te]=pred_logreg(wf,(PX[te]-m2)/s2)
probe_auc=auc_np(Py,oof)
print(f"[probe] out-of-fold error-detection AUC={probe_auc:.3f} (need >=0.60)",flush=True)

# ---- run the arms ----
ARMS=["oracle","inner","output","random"]
per_seed={a:[] for a in ARMS}
for s in range(SEEDS):
    model.load_state_dict(C0)                                   # score pool on the C0 child
    pool=gen(POOL_N,np.random.default_rng(100+s))
    rows=collect_digits(model,pool)
    err=np.array([r[3] for r in rows]); ent=np.array([r[2] for r in rows])
    pinner=pred_logreg(W,(np.array([r[1] for r in rows])-mu)/sdv)
    sig={"oracle":per_problem(rows,err,POOL_N),
         "inner": per_problem(rows,pinner,POOL_N),
         "output":per_problem(rows,ent,POOL_N),
         "random":np.random.default_rng(500+s).random(POOL_N)}
    for a in ARMS:
        pick=[pool[i] for i in np.argsort(-sig[a])[:K]]
        model.load_state_dict(C0)                               # every arm starts from the SAME C0
        train_on(model,pick,TEACH,seed=200+s)
        per_seed[a].append(eval_digit_acc(model,TEST))
        if DEV=="cuda": torch.cuda.empty_cache()
    print(f"[seed {s}] "+" ".join(f"{a}={per_seed[a][-1]*100:.2f}" for a in ARMS),flush=True)

mean={a:float(np.mean(per_seed[a])) for a in ARMS}; std={a:float(np.std(per_seed[a])) for a in ARMS}
inner_minus_out=[per_seed["inner"][i]-per_seed["output"][i] for i in range(SEEDS)]
inner_minus_rand=[per_seed["inner"][i]-per_seed["random"][i] for i in range(SEEDS)]
primary=float(np.mean(inner_minus_out))*100
n_pos=sum(1 for d in inner_minus_out if d>0)
oracle_lead=(mean["oracle"]-mean["random"])*100

need=int(np.ceil(0.75*SEEDS))
teach_worked=(mean["random"]-c0_acc)*100          # did teaching at this budget move the needle at all?
if not(0.30<=c0_err<=0.70) or (probe_auc<0.60) or (teach_worked<0.5):
    verdict="BROKEN_MEASUREMENT"
elif primary>=1.0 and n_pos>=need and mean["inner"]>=mean["random"]:
    verdict="SUPPORTED"
else: verdict="REFUTED"

out=dict(task="addition 0-99, teach at equal budget K",device=DEV,
         C0_start_acc=c0_acc,C0_digit_err=c0_err,probe_oof_AUC=probe_auc,K=K,TEACH_steps=TEACH,SEEDS=SEEDS,
         mean_acc={a:mean[a] for a in ARMS},std_acc={a:std[a] for a in ARMS},
         improvement_over_C0_pts={a:(mean[a]-c0_acc)*100 for a in ARMS},teaching_moved_needle_pts=teach_worked,
         PRIMARY_inner_minus_output_pts=primary,inner_minus_output_per_seed=[d*100 for d in inner_minus_out],
         inner_minus_random_per_seed=[d*100 for d in inner_minus_rand],seeds_inner_beats_output=f"{n_pos}/{SEEDS}",
         oracle_minus_random_pts=oracle_lead,verdict=verdict)
print("\n=== NUMBERS ==="); print(json.dumps(out,indent=1))
print("\n=== VERDICT ===",verdict,f"| PRIMARY inner-output={primary:+.2f}pt ({n_pos}/{SEEDS} seeds) · "
      f"oracle-random={oracle_lead:+.2f}pt · probe AUC={probe_auc:.2f} · C0 err={c0_err:.2f}")
if not SMOKE:
    d=os.path.join(os.path.dirname(__file__),"..","campaigns","teach-curriculum"); os.makedirs(d,exist_ok=True)
    dst=os.path.join(d,"result.json"); json.dump(out,open(dst,"w"),indent=1); print("[saved]",os.path.abspath(dst),flush=True)
