#!/usr/bin/env python3
"""
sep-boundary v2 FAITHFUL run on pythia-410m (author: Илья, NOT delegated).
410m IS cached (872MB). No gpt2 fallback permitted; if a small model loads, ABORT.

CALIBRATED (2026-07-06, sb_window_probe / sb_calib on 410m):
  * layer L=9 (oracle strong here: +1.7; windows open here with cross-config variance).
  * steering strength = mult * ||neutral||, mult in [1,1.5,2,3,4]  (the sb3 bug was
    mult ~0.1-0.5 -> too weak to flip -> ALL windows 0 -> degenerate outcome).
  * "mammal"/"reptile" are MULTI-token in pythia -> avoided; families use only
    single-token category words (fruit/vegetable/bird/fish/country/city/tree/flower/insect).

QUESTION: does a purely-internal predictor say IN ADVANCE where a clean concept-swap
(category flips, attribute preserved, text coherent) holds vs breaks -- and beat the
Park orthogonality baseline ABOVE CHANCE, judged WITHIN each family (fix for sb2's
Simpson's-paradox pooling)?

PREREGISTERED DECISION RULE (frozen before running; keyed on WITHIN-FAMILY):
  Per qualifying family (oracle>=1.0, n_fam>=8, window has variance):
    rO_f=|Spearman(OURS,window)|, rP_f=|Spearman(PARK,window)|, sign_f=sign(rho_OURS).
  meanO=mean_f rO_f, meanP=mean_f rP_f, neg_frac=frac(sign_f<0).
  within_floor = 95th pctile of meanO under 2000 within-family label shuffles.
  * BROKEN_MEASUREMENT if <2 qualifying families OR mean family oracle < 1.0.
  * SUPPORTED_V2  if meanO>=0.50 AND meanO>meanP AND meanO>=within_floor
                     AND delta_CI excludes 0 AND neg_frac>=0.6.
  * PROMISING     if meanO>=within_floor AND meanO>meanP AND 0.30<=meanO<0.50.
  * REFUTED       otherwise.
Prior sb1 (gpt2, n=32): rho_OURS=-0.324 (<floor 0.358). This run: real 410m, calibrated,
5 diverse families, within-family judging.
"""
import os, json, numpy as np, torch
from scipy.stats import spearmanr
torch.set_num_threads(max(1, os.cpu_count() or 4))
from transformer_lens import HookedTransformer

MODEL="pythia-410m"; L=9
print(f"[load] {MODEL} L={L} (offline)...", flush=True)
m = HookedTransformer.from_pretrained(MODEL, device="cpu"); m.eval()
assert m.cfg.n_layers >= 24, f"REFUSE: {m.cfg.n_layers} layers, not 410m"
print(f"[load] ok n_layers={m.cfg.n_layers} d_model={m.cfg.d_model}", flush=True)
rng = np.random.default_rng(0)
W_U = m.W_U.detach().float(); d_model=W_U.shape[0]
Wc = W_U - W_U.mean(dim=1, keepdim=True)
Sigma = (Wc @ Wc.T)/(W_U.shape[1]-1)
Sig_inv = torch.linalg.pinv(Sigma + 1e-4*torch.eye(d_model))
HOOK=f"blocks.{L}.hook_resid_post"

def tid(w):
    t=m.to_tokens(" "+w,prepend_bos=False); return int(t[0,0]) if t.shape[1]==1 else None
@torch.no_grad()
def logits_last(prompt, vec=None):
    toks=m.to_tokens(prompt)
    if vec is None: return m(toks)[0,-1].float()
    def fn(r,hook,v=vec): r[:,-1,:]=r[:,-1,:]+v.to(r.dtype); return r
    return m.run_with_hooks(toks, fwd_hooks=[(HOOK,fn)])[0,-1].float()
@torch.no_grad()
def resid_last(prompts):
    out=[]
    for p in prompts:
        _,c=m.run_with_cache(m.to_tokens(p),names_filter=HOOK); out.append(c[HOOK][0,-1].float())
    return torch.stack(out)
def grad_wrt_resid(prompt,pos_tok,neg_tok):
    saved={}
    def fn(r,hook): r.requires_grad_(True); r.retain_grad(); saved['r']=r; return r
    with torch.enable_grad():
        lg=m.run_with_hooks(m.to_tokens(prompt),fwd_hooks=[(HOOK,fn)])
        margin=lg[0,-1,pos_tok]-lg[0,-1,neg_tok]
        g=torch.autograd.grad(margin,saved['r'])[0][0,-1].detach().float()
    return g
def whit_cos(u,v):
    a=float(u@Sig_inv@v); du=float(u@Sig_inv@u); dv=float(v@Sig_inv@v)
    return abs(a/np.sqrt(du*dv)) if du>0 and dv>0 else 0.0
def entropy(lg): p=torch.softmax(lg,-1); return float(-(p*torch.log(p+1e-12)).sum())

FAMILIES={
 "fruit":{"C":"fruit","Cp":"vegetable",
   "ex_C":["mango","kiwi","melon","lemon","cherry","plum","peach","grape"],
   "ex_Cp":["carrot","potato","onion","celery","spinach","cabbage","pepper","radish"],
   "frame":"I bought a fresh {}",
   "cat_probes":["A {e} is a type of","At the market {e} is sold as a","A {e} is a kind of"],
   "attrs":{"color":["red","green","yellow","orange","purple"],
            "taste":["sweet","sour","bitter","salty","spicy"],
            "grows":["tree","vine","bush","ground","root"]},
   "attr_probe":{"color":"The color of a ripe {e} is","taste":"A {e} usually tastes",
                 "grows":"A {e} grows on a"},
   "targets":["apple","banana","orange","grape","pear","peach","lemon","mango","cherry","melon","plum","lime"],
   "oracle_members":["carrot","potato","onion"]},
 "bird":{"C":"bird","Cp":"fish",
   "ex_C":["robin","sparrow","hawk","finch","crow","pigeon","eagle","owl"],
   "ex_Cp":["salmon","tuna","shark","trout","carp","cod","bass","perch"],
   "frame":"I saw a {}",
   "cat_probes":["A {e} is a type of","In biology a {e} is a kind of","A {e} is classified as a"],
   "attrs":{"cover":["feathers","scales","fur","skin","hair"],
            "move":["flies","swims","runs","walks","jumps"],
            "size":["tiny","small","big","large","huge"]},
   "attr_probe":{"cover":"A {e} is covered in","move":"To move around a {e}","size":"In size a {e} is usually"},
   "targets":["eagle","owl","duck","goose","parrot","penguin","swan","turkey","hawk","crow","robin","pigeon"],
   "oracle_members":["salmon","tuna","shark"]},
 "country":{"C":"country","Cp":"city",
   "ex_C":["France","Spain","Japan","China","Egypt","Brazil","Italy","India"],
   "ex_Cp":["Paris","London","Tokyo","Rome","Berlin","Madrid","Cairo","Moscow"],
   "frame":"I traveled to {}",
   "cat_probes":["{e} is a type of","Geographically {e} is a","{e} is officially a"],
   "attrs":{"continent":["Europe","Asia","Africa","America","Australia"]},
   "attr_probe":{"continent":"The continent of {e} is"},
   "targets":["France","Spain","Japan","China","India","Egypt","Brazil","Germany","Italy","Canada","Russia","Mexico","Kenya","Chile","Norway"],
   "oracle_members":["Paris","London","Tokyo"]},
 "tree":{"C":"tree","Cp":"flower",
   "ex_C":["oak","pine","birch","maple","cedar","elm","ash","beech"],
   "ex_Cp":["rose","tulip","daisy","lily","poppy","iris","orchid","violet"],
   "frame":"In the garden grew a {}",
   "cat_probes":["A {e} is a type of","A {e} is a kind of","Botanically a {e} is a"],
   "attrs":{"color":["green","brown","red","yellow","white"]},
   "attr_probe":{"color":"The color of a {e} is usually"},
   "targets":["oak","pine","maple","birch","willow","cedar","palm","ash","elm","spruce"],
   "oracle_members":["rose","tulip","lily"]},
 "insect":{"C":"insect","Cp":"bird",
   "ex_C":["ant","bee","wasp","moth","fly","beetle","aphid","gnat"],
   "ex_Cp":["robin","sparrow","hawk","finch","crow","pigeon","eagle","owl"],
   "frame":"I saw a {}",
   "cat_probes":["A {e} is a type of","In biology a {e} is a kind of","A {e} is classified as a"],
   "attrs":{"size":["tiny","small","big","large","huge"],
            "color":["black","brown","red","green","yellow"]},
   "attr_probe":{"size":"In size a {e} is","color":"The color of a {e} is"},
   "targets":["ant","bee","wasp","fly","moth","beetle","spider","cricket","worm","snail"],
   "oracle_members":["robin","sparrow","hawk"]},
}
NEUTRAL=["The weather today is","She opened the door and","In the morning I like to",
         "The meeting will start at","He walked down the"]
neut_clean_H = np.mean([entropy(logits_last(p)) for p in NEUTRAL])
MULTS=[1.0,1.5,2.0,3.0,4.0]           # calibrated productive band
ORACLE_MULT=2.0

def build_dir(fam):
    c=resid_last([fam["frame"].format(w) for w in fam["ex_C"]]).mean(0)
    cp=resid_last([fam["frame"].format(w) for w in fam["ex_Cp"]]).mean(0)
    d=cp-c; return d/d.norm()

total_planned=sum(len(f["targets"])*len(f["attrs"]) for f in FAMILIES.values())
print(f"[plan] up to {total_planned} configs across {len(FAMILIES)} families, MULTS={MULTS}", flush=True)

records=[]; fam_oracle={}; fam_rnd={}; k=0
for fname,fam in FAMILIES.items():
    Ct,Cpt=tid(fam["C"]),tid(fam["Cp"])
    if Ct is None or Cpt is None:
        print(f"[skip {fname}] category not single-token", flush=True); continue
    u=build_dir(fam)
    nn=resid_last(NEUTRAL).norm(dim=-1).mean().item()
    steer_H={mlt: np.mean([entropy(logits_last(p,u*(mlt*nn))) for p in NEUTRAL]) for mlt in MULTS}
    ov=[]
    for w in fam["oracle_members"]:
        p=f"A {w} is a type of"; vec=u*(ORACLE_MULT*nn)
        ov.append(float(logits_last(p,vec)[Cpt]-logits_last(p,vec)[Ct])-float(logits_last(p)[Cpt]-logits_last(p)[Ct]))
    fam_oracle[fname]=float(np.mean(ov))
    rflip=rtot=0
    for e in fam["targets"][:2]:
        cps=[cp.format(e=e) for cp in fam["cat_probes"]]
        base=[float(logits_last(p)[Cpt]-logits_last(p)[Ct]) for p in cps]
        r=torch.tensor(rng.standard_normal(d_model),dtype=torch.float32); r=r/r.norm()*(ORACLE_MULT*nn)
        st=[float(logits_last(p,r)[Cpt]-logits_last(p,r)[Ct]) for p in cps]; rtot+=1
        if sum(1 for b,s in zip(base,st) if b<0 and s>0)/len(cps)>0.5: rflip+=1
    fam_rnd[fname]=f"{rflip}/{rtot}"
    print(f"[fam {fname}] oracle={fam_oracle[fname]:+.2f} random_flip={fam_rnd[fname]}", flush=True)
    for e in fam["targets"]:
        if tid(e) is None: continue
        cps=[cp.format(e=e) for cp in fam["cat_probes"]]
        base_cat=[float(logits_last(p)[Cpt]-logits_last(p)[Ct]) for p in cps]
        g_cat=grad_wrt_resid(cps[0],Cpt,Ct)
        for aname,cands in fam["attrs"].items():
            ap=fam["attr_probe"][aname].format(e=e)
            cid=[tid(w) for w in cands]
            if any(c is None for c in cid): continue
            clean=logits_last(ap); clean_arg=int(np.argmax([float(clean[i]) for i in cid]))
            order=np.argsort([float(clean[i]) for i in cid])[::-1]
            top,second=cid[order[0]],cid[order[1]]
            g_attr=grad_wrt_resid(ap,top,second)
            ours=float(abs(torch.nn.functional.cosine_similarity(g_cat,g_attr,dim=0)))
            swap_dir=W_U[:,Cpt]-W_U[:,Ct]
            others=[i for i in cid if i!=top]
            attr_dir=W_U[:,top]-W_U[:,others].mean(dim=1)
            park=whit_cos(swap_dir,attr_dir)
            window=0
            for mlt in MULTS:
                vec=u*(mlt*nn)
                stc=[float(logits_last(p,vec)[Cpt]-logits_last(p,vec)[Ct]) for p in cps]
                FLIP=sum(1 for b,s in zip(base_cat,stc) if b<0 and s>0)/len(cps)>0.5
                stl=logits_last(ap,vec); PRES=int(np.argmax([float(stl[i]) for i in cid]))==clean_arg
                coh=steer_H[mlt]/neut_clean_H
                if FLIP and PRES and 0.8<=coh<=1.2: window+=1
            records.append(dict(fam=fname,ent=e,attr=aname,window=window,ours=ours,park=park))
            k+=1
            print(f"[{k}/{total_planned}] {fname:8} {e:9} {aname:9} win={window} ours={ours:.3f} park={park:.3f}", flush=True)

# ---------- WITHIN-FAMILY STATS ----------
def fam_arr(recs):
    return (np.array([r["window"] for r in recs],float),np.array([r["ours"] for r in recs],float),
            np.array([r["park"] for r in recs],float))
by_fam={}
for fn2 in FAMILIES:
    recs=[r for r in records if r["fam"]==fn2]
    if len(recs)<8 or fam_oracle.get(fn2,0)<1.0: continue
    W,O,P=fam_arr(recs)
    if len(set(W.tolist()))<2: continue          # no window variance -> can't correlate
    ro=spearmanr(O,W).correlation; rp=spearmanr(P,W).correlation
    if np.isnan(ro) or np.isnan(rp): continue
    by_fam[fn2]=dict(n=len(recs),rho_OURS=float(ro),rho_PARK=float(rp),
                     window_mean=float(W.mean()),oracle=fam_oracle[fn2],random_flip=fam_rnd[fn2])
qual=list(by_fam.keys())
meanO=float(np.mean([abs(by_fam[f]["rho_OURS"]) for f in qual])) if qual else float('nan')
meanP=float(np.mean([abs(by_fam[f]["rho_PARK"]) for f in qual])) if qual else float('nan')
neg_frac=float(np.mean([by_fam[f]["rho_OURS"]<0 for f in qual])) if qual else 0.0
mean_oracle=float(np.mean([by_fam[f]["oracle"] for f in qual])) if qual else 0.0
fam_arrays={f:fam_arr([r for r in records if r["fam"]==f]) for f in qual}
perm=[]
for _ in range(2000):
    vals=[]
    for f in qual:
        W,O,P=fam_arrays[f]; Wp=rng.permutation(W); rr=spearmanr(O,Wp).correlation
        if not np.isnan(rr): vals.append(abs(rr))
    if vals: perm.append(np.mean(vals))
within_floor=float(np.nanpercentile(perm,95)) if perm else float('nan')
boot=[]
for _ in range(2000):
    dO=[]; dP=[]
    for f in qual:
        W,O,P=fam_arrays[f]; n=len(W); b=rng.choice(np.arange(n),n,replace=True)
        ro=abs(spearmanr(O[b],W[b]).correlation); rp=abs(spearmanr(P[b],W[b]).correlation)
        if not(np.isnan(ro) or np.isnan(rp)): dO.append(ro); dP.append(rp)
    if dO: boot.append(np.mean(dO)-np.mean(dP))
ci=(float(np.nanpercentile(boot,2.5)),float(np.nanpercentile(boot,97.5))) if boot else (float('nan'),)*2

if len(qual)<2 or mean_oracle<1.0: verdict="BROKEN_MEASUREMENT"
elif meanO>=0.50 and meanO>meanP and meanO>=within_floor and ci[0]>0 and neg_frac>=0.6: verdict="SUPPORTED_V2"
elif meanO>=within_floor and meanO>meanP and 0.30<=meanO<0.50: verdict="PROMISING"
else: verdict="REFUTED"

out=dict(model=MODEL,layer=L,mults=MULTS,n_total=len(records),families_qualified=qual,
         by_family=by_fam,mean_absrho_OURS=meanO,mean_absrho_PARK=meanP,neg_frac_OURS=neg_frac,
         within_family_floor95=within_floor,delta_CI=ci,mean_oracle=mean_oracle,
         all_oracles=fam_oracle,verdict=verdict)
print("\n=== NUMBERS (within-family primary) ==="); print(json.dumps(out,indent=1))
print("\n=== VERDICT ===",verdict,f"| meanO={meanO:.3f} meanP={meanP:.3f} floor={within_floor:.3f} negfrac={neg_frac:.2f} oracle={mean_oracle:+.2f} qual={qual}")
dst=os.path.join(os.path.dirname(__file__),"..","campaigns","sep-boundary","sb3_410m_result.json")
json.dump(out,open(dst,"w"),indent=1); print("[saved]",os.path.abspath(dst),flush=True)
