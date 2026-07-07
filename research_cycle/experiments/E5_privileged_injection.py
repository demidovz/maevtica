#!/usr/bin/env python3
"""E5 — 'Privileged injection': is the verbalizable WORKSPACE causally special, or just readable?
Phase 0b builds the real J-lens: the workspace W at layer L, tens-predicting position ("=" token),
= top singular directions of the Jacobian of DIGIT logits w.r.t. the residual (the directions in
activation space that actually push the digit output = 'verbalizable'). Complement ⊥ = the silent rest.

Substrate: our gpt2 adder, 2-digit + 2-digit with sum<100 (tens digit predicted at the "=" token;
its unspoken intermediate = the units CARRY). Two probes:

  E5a (READ, non-tautological): is the CARRY (the reasoning intermediate) decodable from the W
     projection MORE than from a matched-dim random ⊥ subspace? -> the intermediate lives in the
     verbalizable workspace.

  E5b (CAUSAL, tautology-guarded): take the carry steering direction d (mean h|carry - mean h|nocarry).
     On NO-CARRY problems (correct tens digit t), inject a matched-norm vector at L,q and read the new
     tens digit. Compare 4 directions: d projected into W ; a RANDOM unit inside W ; d projected into ⊥;
     a random unit in full space. THE non-tautological signal is SPECIFICITY: does injecting the CARRY
     direction in W flip the tens digit specifically to (t+1) — the carry-consistent answer, i.e. the
     model REASONS from the injected concept — while random-in-W / ⊥ / random just scatter or do nothing?
     (Anthropic's spider->8->6 is exactly a specific-direction concept swap in the workspace.)

PREREG: BROKEN if adder tens-acc<0.80. Workspace CAUSALLY PRIVILEGED (SUPPORTED) iff
  (E5a) carry AUC from W >= 0.65 AND >= ⊥ AUC + 0.10, AND
  (E5b) at matched norm P(tens->t+1 | inject d_W) >= 0.30 AND >= 2x max(P|random-in-W, P|d_⊥, P|random-full)
        AND specificity P(->t+1)/P(changed) for d_W >= 0.5.
  Else REFUTED (no privilege beyond generic output-effect) — reinforces the detector-not-lever through-line.
"""
import os, json, numpy as np, torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
DEV="cpu"; torch.set_num_threads(max(1,os.cpu_count() or 4))
tok=GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token=tok.eos_token
model=GPT2LMHeadModel.from_pretrained("gpt2").to(DEV); model.eval()
rng=np.random.default_rng(0); L=8; K=8; BATCH=16
DIG=[tok(" "+str(d),add_special_tokens=False).input_ids[0] for d in range(10)]
def sd(n): return " ".join(list(str(n)))
def gen(n):
    out=[]
    while len(out)<n:
        a=int(rng.integers(10,100)); b=int(rng.integers(10,100))
        if a+b<100: out.append((a,b))
    return out

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
    y=np.asarray(y); p=(y==1); a=int(p.sum()); bb=int((~p).sum())
    if a==0 or bb==0: return float('nan')
    r=_rk(s); return float((r[p].sum()-a*(a+1)/2)/(a*bb))
def oof(X,y):
    idx=np.arange(len(y)); np.random.default_rng(0).shuffle(idx); fo=np.array_split(idx,5); o=np.zeros(len(y))
    for f in range(5):
        te=fo[f]; tr=np.concatenate([fo[g] for g in range(5) if g!=f])
        mu=X[tr].mean(0); sdv=X[tr].std(0)+1e-6; w=fit_logreg((X[tr]-mu)/sdv,y[tr]); o[te]=prd(w,(X[te]-mu)/sdv)
    return o

# ---- model internals: recompute logits from a residual injected at layer L ----
T=model.transformer
def _b(blk,h):
    o=blk(h); return o[0] if isinstance(o,tuple) else o
def embed(ids):
    pos=torch.arange(ids.shape[1],device=DEV); return T.drop(T.wte(ids)+T.wpe(pos))
def lower(h):
    for blk in T.h[:L]: h=_b(blk,h)
    return h
def upper(h):
    for blk in T.h[L:]: h=_b(blk,h)
    return model.lm_head(T.ln_f(h))

def bt(probs):
    seqs=[];masks=[]
    for a,b in probs:
        pj=tok(f"{sd(a)} + {sd(b)} =",add_special_tokens=False).input_ids; aj=tok(" "+sd(a+b),add_special_tokens=False).input_ids
        seqs.append(pj+aj+[tok.eos_token_id]); masks.append([-100]*len(pj)+aj+[tok.eos_token_id])
    Lm=max(len(s) for s in seqs); X=torch.full((len(seqs),Lm),tok.eos_token_id,dtype=torch.long)
    Y=torch.full((len(seqs),Lm),-100,dtype=torch.long); A=torch.zeros((len(seqs),Lm),dtype=torch.long)
    for i,(s,m) in enumerate(zip(seqs,masks)): X[i,:len(s)]=torch.tensor(s);Y[i,:len(m)]=torch.tensor(m);A[i,:len(s)]=1
    return X.to(DEV),Y.to(DEV),A.to(DEV)

@torch.no_grad()
def tens_acc(n=200):
    ok=0; probs=gen(n)
    for a,b in probs:
        pj=tok(f"{sd(a)} + {sd(b)} =",add_special_tokens=False).input_ids; q=len(pj)-1  # "=" position
        ids=torch.tensor(pj+[DIG[d] for d in [int(c) for c in str(a+b)]],device=DEV).unsqueeze(0)
        lg=model(ids).logits[0,q,DIG]; ok+= 1 if int(torch.argmax(lg))==(a+b)//10 else 0
    return ok/len(probs)

print(f"[train] adder (sum<100) on {DEV} ...",flush=True)
model.train(); opt=torch.optim.AdamW(model.parameters(),lr=5e-4)
for step in range(1,2201):
    X,Y,A=bt(gen(BATCH)); out=model(input_ids=X,attention_mask=A,labels=Y); out.loss.backward(); opt.step(); opt.zero_grad()
    if step%150==0:
        ta=tens_acc(); print(f"[train] step {step} tens_acc={ta:.3f}",flush=True)
        if ta>=0.88: break
del opt; model.eval()

# ---- collect residual at q ("=" position) + carry + correct tens digit t ----
@torch.no_grad()
def collect(n):
    rows=[]; probs=gen(n)
    for a,b in probs:
        pj=tok(f"{sd(a)} + {sd(b)} =",add_special_tokens=False).input_ids; q=len(pj)-1
        ids=torch.tensor(pj,device=DEV).unsqueeze(0); hL=lower(embed(ids))
        rows.append(dict(hL=hL[0,q].cpu().numpy(),q=q,carry=int((a%10+b%10)>=10),t=(a+b)//10,a=a,b=b,pj=pj))
    return rows
R=collect(900)
H=np.array([r["hL"] for r in R]); carry=np.array([r["carry"] for r in R])
print(f"[data] N={len(R)} carry-rate={carry.mean():.2f}",flush=True)

# ---- Phase 0b: build workspace W = top singular dirs of Jacobian of digit-logits wrt hL at q ----
print("[jlens] building workspace via Jacobian ...",flush=True)
Jrows=[]
for r in R[:60]:
    ids=torch.tensor(r["pj"],device=DEV).unsqueeze(0); hL=lower(embed(ids)).detach().requires_grad_(True)
    logits=upper(hL)[0,r["q"]]
    for tid_ in DIG:
        g=torch.autograd.grad(logits[tid_],hL,retain_graph=True)[0][0,r["q"]]
        Jrows.append(g.detach().cpu().numpy())
M=np.array(Jrows)                                  # [10*60, d]
U,S,Vt=np.linalg.svd(M,full_matrices=False); Wk=Vt[:K]     # [K,d] workspace basis (orthonormal)
evr=float((S[:K]**2).sum()/(S**2).sum())
print(f"[jlens] workspace K={K} captures {evr*100:.1f}% of Jacobian energy",flush=True)

# ---- E5a: carry decodable from W-projection vs matched-dim random ⊥ subspace ----
def proj_feats(H,B): return H@B.T                  # [.,k]
# random orthonormal K-dim basis inside the complement of W
G=np.random.default_rng(1).standard_normal((K,H.shape[1])); G=G-(G@Wk.T)@Wk    # remove W component
Qc,_=np.linalg.qr(G.T); Bperp=Qc.T[:K]             # [K,d] orthonormal in ⊥
aucW=auc(carry,oof(proj_feats(H,Wk),carry)); aucP=auc(carry,oof(proj_feats(H,Bperp),carry))
print(f"[E5a read] carry AUC from workspace={aucW:.3f} vs random-⊥={aucP:.3f} (Δ={aucW-aucP:+.3f})",flush=True)

# ---- E5b: inject carry direction, matched norm, 4 directions, on NO-CARRY problems ----
d=H[carry==1].mean(0)-H[carry==0].mean(0)          # carry steering direction (full space)
dW=(d@Wk.T)@Wk; dperp=d-dW                          # components
def unit(v):
    n=np.linalg.norm(v); return v/n if n>1e-9 else v
dirs={"carry_in_W":unit(dW),"random_in_W":unit(np.random.default_rng(2).standard_normal(K)@Wk),
      "carry_in_perp":unit(dperp),"random_full":unit(np.random.default_rng(3).standard_normal(H.shape[1]))}
NORM=float(np.linalg.norm(d))                       # matched injection norm = ||carry direction||
nocarry=[r for r in R if r["carry"]==0]
@torch.no_grad()
def inject_eval(vec,scale):
    v=torch.tensor(vec*NORM*scale,dtype=torch.float32,device=DEV); flip=chg=0
    for r in nocarry:
        ids=torch.tensor(r["pj"],device=DEV).unsqueeze(0); hL=lower(embed(ids)).clone(); hL[0,r["q"]]+=v
        new=int(torch.argmax(upper(hL)[0,r["q"],DIG])); t=r["t"]
        chg+= 1 if new!=t else 0; flip+= 1 if new==(t+1)%10 else 0
    return flip/len(nocarry), chg/len(nocarry)
print(f"[E5b inject] on {len(nocarry)} no-carry problems, matched norm={NORM:.2f}",flush=True)
scales=[1.0,2.0,3.0,4.0]; e5b={}
for name,vec in dirs.items():
    best=None
    for s in scales:
        pf,pc=inject_eval(vec,s)
        if best is None or pf>best[1]: best=(s,pf,pc)
    e5b[name]=dict(scale=best[0],P_to_t1=best[1],P_changed=best[2],specificity=(best[1]/best[2] if best[2]>0 else 0.0))
    print(f"   {name:14s} bestΔ@s={best[0]}: P(->t+1)={best[1]:.2f} P(changed)={best[2]:.2f} spec={e5b[name]['specificity']:.2f}",flush=True)

pW=e5b["carry_in_W"]["P_to_t1"]; others=max(e5b["random_in_W"]["P_to_t1"],e5b["carry_in_perp"]["P_to_t1"],e5b["random_full"]["P_to_t1"])
if tens_acc(150)<0.80: verdict="BROKEN_MEASUREMENT"
elif aucW>=0.65 and (aucW-aucP)>=0.10 and pW>=0.30 and pW>=2*others and e5b["carry_in_W"]["specificity"]>=0.5: verdict="SUPPORTED"
else: verdict="REFUTED"
out=dict(exp="E5 privileged injection (workspace causal privilege)",L=L,K=K,jac_energy=evr,
         carry_AUC_workspace=aucW,carry_AUC_perp=aucP,inject_norm=NORM,e5b=e5b,
         P_t1_W=pW,P_t1_best_control=others,verdict=verdict)
print("\n=== NUMBERS ==="); print(json.dumps(out,indent=1))
print("\n=== VERDICT ===",verdict,f"| E5a W={aucW:.2f} vs ⊥={aucP:.2f} · E5b P(->t+1) carry-in-W={pW:.2f} vs best-control={others:.2f} spec_W={e5b['carry_in_W']['specificity']:.2f}")
dd=os.path.join(os.path.dirname(__file__),"..","campaigns","workspace"); os.makedirs(dd,exist_ok=True)
json.dump(out,open(os.path.join(dd,"E5_result.json"),"w"),indent=1); print("[saved]",flush=True)
