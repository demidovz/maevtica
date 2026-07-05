#!/usr/bin/env python3
"""Calibration probe for pythia-410m: find (layer, steer-mult) where the steering
ORACLE passes (>=1.0 logit shift pushing Cp-direction into GENUINE Cp members),
and report which category/candidate words are single-token in pythia's tokenizer.
Cheap: ~a few minutes. Output tells us how to patch sep_boundary_410m.py."""
import os, json, numpy as np, torch
torch.set_num_threads(max(1, os.cpu_count() or 4))
from transformer_lens import HookedTransformer
m = HookedTransformer.from_pretrained("pythia-410m", device="cpu"); m.eval()
print(f"n_layers={m.cfg.n_layers} d_model={m.cfg.d_model}", flush=True)

def tid(w):
    t = m.to_tokens(" " + w, prepend_bos=False)
    return int(t[0,0]) if t.shape[1]==1 else None

# --- single-token audit ---
CAT_WORDS=["fruit","vegetable","bird","mammal","fish","country","city","tree","flower","animal","plant","insect","reptile"]
CANDS=["red","green","feathers","fur","scales","Europe","Asia","Africa","legs","fins","sweet","sour"]
print("=== single-token audit (pythia) ===")
for w in CAT_WORDS+CANDS:
    print(f"  {'OK ' if tid(w) is not None else 'MULTI'} {w}", flush=True)

FAM={
 "fruit":{"C":"fruit","Cp":"vegetable","ex_C":["mango","kiwi","melon","lemon","cherry","plum"],
          "ex_Cp":["carrot","potato","onion","celery","spinach","cabbage"],
          "frame":"I bought a fresh {}","oracle":["carrot","potato","onion"]},
 "country":{"C":"country","Cp":"city","ex_C":["France","Spain","Japan","China","Egypt","Brazil"],
            "ex_Cp":["Paris","London","Tokyo","Rome","Berlin","Madrid"],
            "frame":"I traveled to {}","oracle":["Paris","London","Tokyo"]},
}
NEUTRAL=["The weather today is","She opened the door and","In the morning I like to","The meeting will start at","He walked down the"]

@torch.no_grad()
def resid_last(prompts, L):
    h=f"blocks.{L}.hook_resid_post"; out=[]
    for p in prompts:
        _,c=m.run_with_cache(m.to_tokens(p), names_filter=h); out.append(c[h][0,-1].float())
    return torch.stack(out)

@torch.no_grad()
def shift(prompt, Cpt, Ct, vec, L):
    h=f"blocks.{L}.hook_resid_post"
    base=m(m.to_tokens(prompt))[0,-1].float()
    def fn(r,hook,v=vec): r[:,-1,:]=r[:,-1,:]+v.to(r.dtype); return r
    st=m.run_with_hooks(m.to_tokens(prompt), fwd_hooks=[(h,fn)])[0,-1].float()
    return float((st[Cpt]-st[Ct])-(base[Cpt]-base[Ct]))

print("\n=== oracle sweep: layer x mult (vec = mult * ||neutral|| * unit_dir) ===")
print("fam      L   mult  oracle")
best={}
for fname,f in FAM.items():
    Ct,Cpt=tid(f["C"]),tid(f["Cp"])
    if Ct is None or Cpt is None:
        print(f"{fname}: category not single-token -> skip"); continue
    for L in [6,9,12,15,18]:
        c=resid_last([f["frame"].format(w) for w in f["ex_C"]],L).mean(0)
        cp=resid_last([f["frame"].format(w) for w in f["ex_Cp"]],L).mean(0)
        u=(cp-c); u=u/u.norm()
        nn=resid_last(NEUTRAL,L).norm(dim=-1).mean().item()
        for mult in [0.5,1.0,2.0,4.0,8.0]:
            vec=u*(mult*nn)
            orc=float(np.mean([shift(f"A {w} is a type of",Cpt,Ct,vec,L) for w in f["oracle"]]))
            tag=" <=PASS" if orc>=1.0 else ""
            print(f"{fname:8} {L:3} {mult:5.1f} {orc:+.3f}{tag}", flush=True)
            if orc>=1.0 and (fname not in best or orc>best[fname][2]):
                best[fname]=(L,mult,orc)
print("\n=== best passing (fam -> L,mult,oracle) ===")
print(json.dumps({k:list(v) for k,v in best.items()}, indent=1))
