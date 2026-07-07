#!/usr/bin/env python3
"""E5-compose — does the carry LEVER compose? A dial or a switch?
E5 showed injecting the carry direction (in the workspace) flips the tens digit specifically to t+1
(carry applied), 99% specific. Now: inject it HARDER. If the landing digit marches t+1 -> t+2 -> t+3
as we push, the direction is a genuine linear 'add-to-the-tens' DIAL (the model has a magnitude axis
and steering composes). If it saturates at t+1 and then scatters, it's a binary SWITCH (carry / no
carry) with no composition. Both are informative. Same substrate/tooling as E5.
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
        lg=model(ids).logits[0,q,DIG]; ok+= 1 if int(torch.argmax(lg))==(a+b)//10 else 0
    return ok/len(probs)

print(f"[train] adder (sum<100) ...",flush=True)
model.train(); opt=torch.optim.AdamW(model.parameters(),lr=5e-4)
for step in range(1,2201):
    X,Y,A=bt(gen(BATCH)); out=model(input_ids=X,attention_mask=A,labels=Y); out.loss.backward(); opt.step(); opt.zero_grad()
    if step%150==0:
        ta=tens_acc(); print(f"[train] step {step} tens_acc={ta:.3f}",flush=True)
        if ta>=0.88: break
del opt; model.eval()

@torch.no_grad()
def collect(n):
    rows=[]
    for a,b in gen(n):
        pj=tok(f"{sd(a)} + {sd(b)} =",add_special_tokens=False).input_ids; q=len(pj)-1
        hL=lower(embed(torch.tensor(pj,device=DEV).unsqueeze(0)))
        rows.append(dict(hL=hL[0,q].cpu().numpy(),q=q,carry=int((a%10+b%10)>=10),t=(a+b)//10,pj=pj))
    return rows
R=collect(900); H=np.array([r["hL"] for r in R]); carry=np.array([r["carry"] for r in R])
print(f"[data] N={len(R)} carry-rate={carry.mean():.2f}",flush=True)

# workspace via Jacobian (same as E5)
Jrows=[]
for r in R[:60]:
    hL=lower(embed(torch.tensor(r["pj"],device=DEV).unsqueeze(0))).detach().requires_grad_(True)
    logits=upper(hL)[0,r["q"]]
    for tid_ in DIG:
        g=torch.autograd.grad(logits[tid_],hL,retain_graph=True)[0][0,r["q"]]; Jrows.append(g.detach().cpu().numpy())
U,S,Vt=np.linalg.svd(np.array(Jrows),full_matrices=False); Wk=Vt[:K]
d=H[carry==1].mean(0)-H[carry==0].mean(0); dW=(d@Wk.T)@Wk
dirv=dW/np.linalg.norm(dW); NORM=float(np.linalg.norm(d))
print(f"[setup] workspace built, carry-in-W dir ready, base norm={NORM:.1f}",flush=True)

# COMPOSE sweep: push the carry-in-W direction harder; where does the tens digit land?
nocarry=[r for r in R if r["carry"]==0]
@torch.no_grad()
def offsets_at(scale):
    v=torch.tensor(dirv*NORM*scale,dtype=torch.float32,device=DEV); off=np.zeros(10,dtype=int)
    for r in nocarry:
        hL=lower(embed(torch.tensor(r["pj"],device=DEV).unsqueeze(0))).clone(); hL[0,r["q"]]+=v
        new=int(torch.argmax(upper(hL)[0,r["q"],DIG])); off[(new-r["t"])%10]+=1
    return off/off.sum()
scales=[0.0,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,6.0,7.0]
print(f"[compose] pushing carry-in-W on {len(nocarry)} no-carry problems; P(landing offset from t):",flush=True)
print("  scale |  +0   +1   +2   +3   +4  +5.. | modal",flush=True)
table=[]
for s in scales:
    p=offsets_at(s); modal=int(np.argmax(p))
    p5=float(p[5:].sum())
    print(f"  {s:4.1f}  | {p[0]:.2f} {p[1]:.2f} {p[2]:.2f} {p[3]:.2f} {p[4]:.2f} {p5:.2f} | +{modal}",flush=True)
    table.append(dict(scale=s,p=[round(float(x),3) for x in p],modal=modal))

# read the march: modal offset as scale grows
modals=[t["modal"] for t in table]
reached2=any(t["modal"]==2 and t["p"][2]>=0.30 for t in table)
reached3=any(t["modal"]==3 and t["p"][3]>=0.25 for t in table)
march=modals==sorted(modals) and reached2   # non-decreasing and reaches a clean +2
verdict="DIAL (composes: lever is a linear add-to-tens knob)" if (reached2 and march) else \
        ("PARTIAL (reaches +2 but not a clean monotone march)" if reached2 else "SWITCH (saturates at +1 / scatters — binary carry, no composition)")
out=dict(exp="E5-compose: does the carry lever compose (dial vs switch)?",base_norm=NORM,
         table=table,reached_plus2=reached2,reached_plus3=reached3,verdict=verdict)
print("\n=== VERDICT ===",verdict)
print("   modal offsets by scale:",modals,"| reached +2:",reached2,"+3:",reached3)
dd=os.path.join(os.path.dirname(__file__),"..","campaigns","workspace"); os.makedirs(dd,exist_ok=True)
json.dump(out,open(os.path.join(dd,"E5_compose_result.json"),"w"),indent=1); print("[saved]",flush=True)
