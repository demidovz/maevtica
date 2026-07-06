#!/usr/bin/env python3
"""Step 1 of DEEP reflect-route: fine-tune gpt2 on addition to PARTIAL competence
(~50% digit error) on the RTX 3050, so a downstream reflect-route run can measure
the practical accuracy GAIN in an UN-saturated regime (base gpt2/410m are ~95% wrong
on all bands — calibration proved no un-saturated regime exists without training).
Saves the checkpoint closest to 50% error to CKPT_DIR for the CPU analysis step."""
import os, numpy as np, torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

CKPT=os.environ.get("CKPT_DIR","/tmp/ckpt_add")
DEV="cuda" if torch.cuda.is_available() else "cpu"
print(f"device={DEV} ckpt={CKPT}", flush=True)
tok=GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token=tok.eos_token
model=GPT2LMHeadModel.from_pretrained("gpt2").to(DEV); model.train()
opt=torch.optim.AdamW(model.parameters(),lr=5e-4)
rng=np.random.default_rng(0)
DIG=[tok(" "+str(d),add_special_tokens=False).input_ids[0] for d in range(10)]

def sd(n): return " ".join(list(str(n)))
def sample(n):
    probs=[]
    for _ in range(n):
        a=int(rng.integers(0,100)); b=int(rng.integers(0,100)); probs.append((a,b))
    return probs

def batch_tensors(probs):
    """full = 'a + b = sum'<eos>; labels mask everything up to and incl '=' (loss on answer digits only)."""
    seqs=[]; masks=[]
    for a,b in probs:
        prompt=f"{sd(a)} + {sd(b)} ="
        pj=tok(prompt,add_special_tokens=False).input_ids
        aj=tok(" "+sd(a+b),add_special_tokens=False).input_ids
        ids=pj+aj+[tok.eos_token_id]
        lab=[-100]*len(pj)+aj+[tok.eos_token_id]
        seqs.append(ids); masks.append(lab)
    L=max(len(s) for s in seqs)
    X=torch.full((len(seqs),L),tok.eos_token_id,dtype=torch.long)
    Y=torch.full((len(seqs),L),-100,dtype=torch.long)
    A=torch.zeros((len(seqs),L),dtype=torch.long)
    for i,(s,m) in enumerate(zip(seqs,masks)):
        X[i,:len(s)]=torch.tensor(s); Y[i,:len(m)]=torch.tensor(m); A[i,:len(s)]=1
    return X.to(DEV),Y.to(DEV),A.to(DEV)

@torch.no_grad()
def eval_err(nprob=300):
    model.eval(); errs=tot=0
    for a,b in sample(nprob):
        prompt=f"{sd(a)} + {sd(b)} ="
        pj=tok(prompt,add_special_tokens=False).input_ids
        ans=[int(c) for c in str(a+b)]
        aj=[DIG[d] for d in ans]
        ids=torch.tensor(pj+aj,device=DEV).unsqueeze(0)
        lg=model(ids).logits[0]
        start=len(pj)
        for k,d in enumerate(ans):
            pred=int(torch.argmax(lg[start+k-1,DIG]))
            errs+= 1 if pred!=d else 0; tot+=1
    model.train(); return errs/tot

BATCH=48; MAX_STEPS=4000; EVAL_EVERY=100
best_gap=9.9; hist=[]
for step in range(1,MAX_STEPS+1):
    X,Y,A=batch_tensors(sample(BATCH))
    out=model(input_ids=X,attention_mask=A,labels=Y)
    out.loss.backward(); opt.step(); opt.zero_grad()
    if step%EVAL_EVERY==0:
        er=eval_err()
        hist.append((step,round(er,3)))
        gap=abs(er-0.5)
        mark=""
        if gap<best_gap:
            best_gap=gap; os.makedirs(CKPT,exist_ok=True)
            model.save_pretrained(CKPT); tok.save_pretrained(CKPT)
            with open(os.path.join(CKPT,"meta.txt"),"w") as f: f.write(f"step={step} err={er:.3f}\n")
            mark=f"  <== saved (closest to 50%, gap={gap:.3f})"
        print(f"step {step:4} loss={out.loss.item():.3f} err={er:.1%}{mark}", flush=True)
        if er<0.22:   # well past 50%, no better ~50% point ahead
            print("[stop] model competent enough; best ~50% checkpoint already saved", flush=True); break
print("history:",hist, flush=True)
print(f"[done] best checkpoint saved to {CKPT} (gap-to-50%={best_gap:.3f})", flush=True)
