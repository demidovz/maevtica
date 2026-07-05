#!/usr/bin/env python3
"""Window-calibration probe: does a CLEAN window (FLIP & PRESERVE & COHERENT) ever
open on pythia-410m, and at what steering strength? Sweeps a WIDE alpha grid on a
few real target entities. If windows open with variance -> fix main script & rerun.
If windows stay ~absent across all strengths -> the phenomenon is rare at this scale
(honest dead-end)."""
import os, numpy as np, torch
torch.set_num_threads(max(1, os.cpu_count() or 4))
from transformer_lens import HookedTransformer
m = HookedTransformer.from_pretrained("pythia-410m", device="cpu"); m.eval()
def tid(w):
    t=m.to_tokens(" "+w,prepend_bos=False); return int(t[0,0]) if t.shape[1]==1 else None

FAM={
 "fruit":{"C":"fruit","Cp":"vegetable","ex_C":["mango","kiwi","melon","lemon","cherry","plum"],
   "ex_Cp":["carrot","potato","onion","celery","spinach","cabbage"],"frame":"I bought a fresh {}",
   "cat_probes":["A {e} is a type of","At the market {e} is sold as a","A {e} is a kind of"],
   "attr_probe":"The color of a ripe {e} is","cands":["red","green","yellow","orange","purple"],
   "targets":["apple","banana","orange","grape","pear","peach"]},
 "country":{"C":"country","Cp":"city","ex_C":["France","Spain","Japan","China","Egypt","Brazil"],
   "ex_Cp":["Paris","London","Tokyo","Rome","Berlin","Madrid"],"frame":"I traveled to {}",
   "cat_probes":["{e} is a type of","Geographically {e} is a","{e} is officially a"],
   "attr_probe":"The continent of {e} is","cands":["Europe","Asia","Africa","America","Australia"],
   "targets":["France","Spain","Japan","India","Egypt","Brazil"]},
}
NEUTRAL=["The weather today is","She opened the door and","In the morning I like to","The meeting will start at","He walked down the"]
def ent(lg): p=torch.softmax(lg,-1); return float(-(p*torch.log(p+1e-12)).sum())
L=9
@torch.no_grad()
def resid_last(prompts):
    h=f"blocks.{L}.hook_resid_post"; out=[]
    for p in prompts:
        _,c=m.run_with_cache(m.to_tokens(p),names_filter=h); out.append(c[h][0,-1].float())
    return torch.stack(out)
@torch.no_grad()
def logits(prompt,vec=None):
    t=m.to_tokens(prompt)
    if vec is None: return m(t)[0,-1].float()
    h=f"blocks.{L}.hook_resid_post"
    def fn(r,hook,v=vec): r[:,-1,:]=r[:,-1,:]+v.to(r.dtype); return r
    return m.run_with_hooks(t,fwd_hooks=[(h,fn)])[0,-1].float()

neutH=np.mean([ent(logits(p)) for p in NEUTRAL])
MULTS=[0.5,1.0,1.5,2.0,3.0,4.0,6.0]
print(f"L={L}  neutH={neutH:.2f}  (FLIP&PRES&COH per mult)\n")
for fname,f in FAM.items():
    Ct,Cpt=tid(f["C"]),tid(f["Cp"])
    c=resid_last([f["frame"].format(w) for w in f["ex_C"]]).mean(0)
    cp=resid_last([f["frame"].format(w) for w in f["ex_Cp"]]).mean(0)
    u=(cp-c); u=u/u.norm()
    nn=resid_last(NEUTRAL).norm(dim=-1).mean().item()
    cid=[tid(w) for w in f["cands"]]
    print(f"--- {fname} ---")
    win_by_target={}
    for e in f["targets"]:
        cps=[p.format(e=e) for p in f["cat_probes"]]
        base=[float(logits(p)[Cpt]-logits(p)[Ct]) for p in cps]
        ap=f["attr_probe"].format(e=e); clean=logits(ap)
        clean_arg=int(np.argmax([float(clean[i]) for i in cid]))
        row=[]
        window=0
        for mlt in MULTS:
            vec=u*(mlt*nn)
            stc=[float(logits(p,vec)[Cpt]-logits(p,vec)[Ct]) for p in cps]
            FLIP=sum(1 for b,s in zip(base,stc) if b<0 and s>0)/len(cps)>0.5
            stl=logits(ap,vec); PRES=int(np.argmax([float(stl[i]) for i in cid]))==clean_arg
            COH=0.8<=ent(logits(ap,vec))/neutH<=1.2 if neutH>0 else False
            ok=FLIP and PRES and COH
            window+= 1 if ok else 0
            row.append(("F" if FLIP else ".")+("P" if PRES else ".")+("C" if COH else "."))
        win_by_target[e]=window
        print(f"  {e:8} win={window}  " + " ".join(f"{mlt}:{r}" for mlt,r in zip(MULTS,row)))
    ws=list(win_by_target.values())
    print(f"  => windows: {ws}  variance={'YES' if len(set(ws))>1 else 'NO'}\n")
