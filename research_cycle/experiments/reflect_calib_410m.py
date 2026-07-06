#!/usr/bin/env python3
"""Calibration for the DEEPER reflect-route run on pythia-410m: find a difficulty
band where the model errs ~40-60% of digits (un-saturated), so the practical
accuracy-GAIN metric has room — not the 89% saturation that blurred rr2-rr4 on gpt2.
Cheap: error-rate only, no probe."""
import os, numpy as np, torch
torch.set_num_threads(max(1,os.cpu_count() or 4))
from transformer_lens import HookedTransformer
m=HookedTransformer.from_pretrained("pythia-410m",device="cpu"); m.eval()
assert m.cfg.n_layers>=24, f"not 410m: {m.cfg.n_layers}"
DIG=[m.to_single_token(" "+str(d)) for d in range(10)]
assert all(isinstance(x,int) for x in DIG), "digits not single-token in pythia"
print(f"loaded pythia-410m n_layers={m.cfg.n_layers}", flush=True)

def sd(n): return " ".join(list(str(n)))

@torch.no_grad()
def err_rate(lo,hi,nprob,seed):
    rng=np.random.default_rng(seed); errs=tot=carry_err=carry_tot=0
    for _ in range(nprob):
        a=int(rng.integers(lo,hi)); b=int(rng.integers(lo,hi)); s=a+b
        prompt=f"{sd(a)} + {sd(b)} ="
        ans=[int(c) for c in str(s)]
        pids=m.to_tokens(prompt)[0]
        aids=torch.tensor([DIG[d] for d in ans])
        full=torch.cat([pids,aids]).unsqueeze(0)
        lg=m(full)[0]
        start=len(pids)
        is_carry=(a%10+b%10)>=10
        for k,d in enumerate(ans):
            p=start+k
            pred=int(torch.argmax(lg[p-1,DIG]))
            e=1 if pred!=d else 0
            errs+=e; tot+=1
            if is_carry: carry_err+=e; carry_tot+=1
    return errs/tot, tot, (carry_err/carry_tot if carry_tot else float('nan'))

print("band                 err_rate   n   (carry-case err)", flush=True)
for name,(lo,hi) in [("1-digit 0-9",(0,10)),("0-19",(0,20)),("0-49",(0,50)),("2-digit 0-99",(0,100))]:
    er,n,ce=err_rate(lo,hi,200,0)
    tag=" <== ~50% GOOD" if 0.35<=er<=0.65 else ""
    print(f"{name:18} {er:6.1%}   {n:4}   {ce:5.1%}{tag}", flush=True)
