#!/usr/bin/env python3
"""Gate check: can gpt2 TRACK a simple in-context fact? (batched, plain transformers, CPU)
Template: '{name} put the {item} in the {container}. The {item} is in the ___' -> does the
model rank the CORRECT container top among candidates? honest acc >> chance => the model
'knows' the fact and can later be forced to contradict it (a real conflict)."""
import os, numpy as np, torch
torch.set_num_threads(max(1,os.cpu_count() or 4))
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
tok=GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token=tok.eos_token
m=GPT2LMHeadModel.from_pretrained("gpt2"); m.eval()
rng=np.random.default_rng(0)
NAMES=["Anna","Tom","Sara","Ben","Mia","Jack","Lucy","Sam"]
ITEMS=["key","coin","ring","ball","card","stone","pen","cup"]
CONTS=["box","basket","jar","bag","drawer","bowl"]
def tid(w):
    ids=tok(" "+w,add_special_tokens=False).input_ids; return ids[0] if len(ids)==1 else None
CT={c:tid(c) for c in CONTS}; CONTS=[c for c in CONTS if CT[c] is not None]; CID=[CT[c] for c in CONTS]
print("containers (single-token):",CONTS,flush=True)

probs=[]
for _ in range(200):
    probs.append((rng.choice(NAMES),rng.choice(ITEMS),rng.choice(CONTS)))
@torch.no_grad()
def run(bs=50):
    ok=tot=0; margins=[]
    for i in range(0,len(probs),bs):
        chunk=probs[i:i+bs]
        texts=[f"{n} put the {it} in the {c}. The {it} is in the" for n,it,c in chunk]
        enc=tok(texts,return_tensors="pt",padding=True); lg=m(**enc).logits
        L=enc["attention_mask"].sum(1)-1
        for j,(n,it,tc) in enumerate(chunk):
            row=lg[j,L[j],CID].float(); pred=CONTS[int(torch.argmax(row))]
            ok+= 1 if pred==tc else 0; tot+=1
            s=torch.sort(row,descending=True).values; margins.append(float(s[0]-s[1]))
    return ok/tot, float(np.mean(margins))
a,mg=run()
print(f"[gate] honest in-context accuracy = {a:.2f} (chance {1/len(CONTS):.2f}) · mean top-2 logit margin={mg:.2f}",flush=True)
print("VERDICT:", "OK" if a>=0.55 else ("WEAK" if a>=0.30 else "TOO_WEAK"),
      "- gpt2 tracks the fact" if a>=0.55 else "- need bigger model / simpler task",flush=True)
