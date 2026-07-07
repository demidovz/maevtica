#!/usr/bin/env python3
"""'малыш-крошка' — structure-guided vs blind compression. Does READING the concept-structure let
us squeeze the model MORE than a structure-blind baseline? Substrate: our gpt2 adder. At layer L,
"=" position, we replace the residual with its projection onto a k-dimensional subspace (keep k
directions, drop the rest) and re-run the upper layers -> a k-number bottleneck. Three ways to
choose the k directions:
  SMART (Jacobian): the directions that actually drive the digit output (our workspace read).
  BLIND (PCA):      the highest-variance directions of the activations (structure-blind).
  RANDOM:           k random directions.
We sweep k and measure how much addition survives. The one keeping the skill at the SMALLEST k wins.

PREREG: BROKEN if full adder tens-acc<0.85. STRUCTURE HELPS iff k needed to reach 95% of full acc is
>=2x smaller for SMART than for BLIND. If SMART≈BLIND -> variance is enough (reading adds nothing).
If RANDOM needs far more -> any structure helps but the fancy read isn't required.
"""
import os, json, numpy as np, torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
DEV="cpu"; torch.set_num_threads(max(1,os.cpu_count() or 4))
tok=GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token=tok.eos_token
model=GPT2LMHeadModel.from_pretrained("gpt2").to(DEV); model.eval()
rng=np.random.default_rng(0); L=8; BATCH=16
DIG=[tok(" "+str(d),add_special_tokens=False).input_ids[0] for d in range(10)]
def sd(n): return " ".join(list(str(n)))
def gen(n):
    out=[]
    while len(out)<n:
        a=int(rng.integers(10,100)); b=int(rng.integers(10,100))
        if a+b<100: out.append((a,b))
    return out
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
        pj=tok(f"{sd(a)} + {sd(b)} =",add_special_tokens=False).input_ids; q=len(pj)-1
        ids=torch.tensor(pj+[DIG[d] for d in [int(c) for c in str(a+b)]],device=DEV).unsqueeze(0)
        ok+= 1 if int(torch.argmax(model(ids).logits[0,q,DIG]))==(a+b)//10 else 0
    return ok/len(probs)

print(f"[train] adder (sum<100) ...",flush=True)
model.train(); opt=torch.optim.AdamW(model.parameters(),lr=5e-4)
for step in range(1,2201):
    X,Y,A=bt(gen(BATCH)); out=model(input_ids=X,attention_mask=A,labels=Y); out.loss.backward(); opt.step(); opt.zero_grad()
    if step%150==0:
        ta=tens_acc(); print(f"[train] step {step} tens_acc={ta:.3f}",flush=True)
        if ta>=0.90: break
del opt; model.eval()

# collect: store full lower-half residual per problem + q + correct tens t
@torch.no_grad()
def collect(n):
    rows=[]
    for a,b in gen(n):
        pj=tok(f"{sd(a)} + {sd(b)} =",add_special_tokens=False).input_ids; q=len(pj)-1
        hL=lower(embed(torch.tensor(pj,device=DEV).unsqueeze(0)))
        rows.append(dict(hL=hL, q=q, t=(a+b)//10))
    return rows
R=collect(600)
Hq=np.array([r["hL"][0,r["q"]].numpy() for r in R]); mean=Hq.mean(0)
full=np.mean([int(torch.argmax(upper(r["hL"])[0,r["q"],DIG]))==r["t"] for r in R])
print(f"[data] N={len(R)} full tens-acc (no compression)={full:.3f}",flush=True)

# SMART basis: Jacobian of digit-logits wrt hL at q
Jrows=[]
for r in R[:60]:
    hL=r["hL"].detach().requires_grad_(True); logits=upper(hL)[0,r["q"]]
    for tid_ in DIG:
        g=torch.autograd.grad(logits[tid_],hL,retain_graph=True)[0][0,r["q"]]; Jrows.append(g.detach().numpy())
_,_,VtJ=np.linalg.svd(np.array(Jrows),full_matrices=False)
# BLIND basis: PCA of centered activations
_,_,VtP=np.linalg.svd(Hq-mean,full_matrices=False)
# RANDOM basis: fixed random orthonormal
Qr,_=np.linalg.qr(np.random.default_rng(7).standard_normal((Hq.shape[1],Hq.shape[1])))
bases={"SMART_jacobian":VtJ,"BLIND_pca":VtP,"RANDOM":Qr.T}
mean_t=torch.tensor(mean,dtype=torch.float32)

@torch.no_grad()
def acc_at(Vt,k):
    B=torch.tensor(Vt[:k],dtype=torch.float32)                 # [k,d] orthonormal
    ok=0
    for r in R:
        h=r["hL"].clone(); v=h[0,r["q"]]-mean_t
        h[0,r["q"]]=mean_t+ (v@B.T)@B                          # keep only k directions
        ok+= 1 if int(torch.argmax(upper(h)[0,r["q"],DIG]))==r["t"] else 0
    return ok/len(R)

ks=[1,2,4,8,16,32,64,128]
print(f"[sweep] tens-acc after keeping only k directions (full={full:.3f}):",flush=True)
print("   k   | SMART  BLIND  RANDOM",flush=True)
tab={n:{} for n in bases}
for k in ks:
    line=f"  {k:4d} |"
    for name,Vt in bases.items():
        a=acc_at(Vt,k); tab[name][k]=a; line+=f"  {a:.3f}"
    print(line,flush=True)

def k95(d):
    tgt=0.95*full
    for k in ks:
        if d[k]>=tgt: return k
    return None
k_smart=k95(tab["SMART_jacobian"]); k_blind=k95(tab["BLIND_pca"]); k_rand=k95(tab["RANDOM"])
print(f"[k95] dims needed to reach 95% of full: SMART={k_smart} BLIND={k_blind} RANDOM={k_rand}",flush=True)
if full<0.85: verdict="BROKEN_MEASUREMENT"
elif k_smart is not None and k_blind is not None and k_smart*2<=k_blind: verdict="STRUCTURE_HELPS (reading beats blind compression)"
elif k_smart is not None and k_blind is not None and abs(k_smart-k_blind)<=max(1,k_blind//4): verdict="TIE (variance is enough; reading adds little over blind)"
else: verdict="MIXED"
out=dict(exp="малыш-крошка: structure-guided vs blind compression",full_acc=float(full),
         table={n:{int(k):float(v) for k,v in d.items()} for n,d in tab.items()},
         k95_smart=k_smart,k95_blind=k_blind,k95_random=k_rand,verdict=verdict)
print("\n=== VERDICT ===",verdict,f"| full {full:.2f} · k95 SMART {k_smart} BLIND {k_blind} RANDOM {k_rand}")
dd=os.path.join(os.path.dirname(__file__),"..","campaigns","workspace"); os.makedirs(dd,exist_ok=True)
json.dump(out,open(os.path.join(dd,"E_shrink_result.json"),"w"),indent=1); print("[saved]",flush=True)
