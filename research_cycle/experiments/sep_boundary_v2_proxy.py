#!/usr/bin/env python3
"""
sep-boundary v2 PROXY test (TESTER / teeth).

FAITHFUL v2 = pythia-410m + ~250 diverse configs -> NOT computable in tight budget
(410m not cached; 250-config alpha-sweep is multi-hour on CPU). Instead this runs
a decisive CHEAP PROXY on gpt2 that stresses the ONE load-bearing empirical
assumption behind v2: that adding configs AND swap-family DIVERSITY makes the
OURS grad-overlap predictor lift past the random-permutation floor. If OURS
collapses when we pool a 2nd family, more power (410m/250) will not rescue a
predictor that is not cross-family stable.

PREREGISTERED DECISION RULE (frozen before running):
  Let rO=|Spearman(OURS,window)|, rP=|Spearman(Park,window)|, sign_O=sign(rho_OURS),
  floor=95th pctile of |Spearman| under 5000 label permutations at this n.
  Prior sb1 (n=32, fruit only): rho_OURS=-0.324 (right sign, negative), floor 0.358.
  * BROKEN_MEASUREMENT  if oracle < 1.0  -> refuse any verdict.
  * PROXY_SUPPORTS_V2   if rO >= floor  AND rO > rP  AND rO >= 0.30 AND sign_O < 0.
        (signal survives diversity+n, clears the now-lower floor, still beats Park,
         same sign/magnitude as prior -> v2's n+diversity lever is real; the full
         |rho|>=0.5 bar still needs the big 410m/250 run = designed_not_run.)
  * PROXY_REFUTES_V2    if rO < floor OR sign_O >= 0 OR rO < 0.15.
        (diversity/pooling kills the signal -> 410m unlikely to rescue it.)
  * PROXY_INCONCLUSIVE  otherwise.
This proxy CANNOT by itself confirm v2 (needs 410m/250); it can only de-risk or kill the direction.
"""
import os, json, numpy as np, torch
from scipy.stats import spearmanr
torch.set_num_threads(max(1, os.cpu_count() or 4))
from transformer_lens import HookedTransformer

DEV="cpu"; L=8
m = HookedTransformer.from_pretrained("gpt2", device=DEV); m.eval()
rng = np.random.default_rng(0)
W_U = m.W_U.detach().float()              # [d_model, vocab]
d_model = W_U.shape[0]
# Park causal whitening: Sigma = cov of unembedding rows (over vocab)
Wc = W_U - W_U.mean(dim=1, keepdim=True)
Sigma = (Wc @ Wc.T) / (W_U.shape[1]-1)    # [d,d]
Sig_inv = torch.linalg.pinv(Sigma + 1e-4*torch.eye(d_model))

def tid(w):
    t = m.to_tokens(" " + w, prepend_bos=False)
    return int(t[0,0]) if t.shape[1]==1 else None

@torch.no_grad()
def logits_last(prompt, vec=None):
    toks = m.to_tokens(prompt)
    if vec is None:
        return m(toks)[0,-1].float()
    h=f"blocks.{L}.hook_resid_post"
    def fn(r,hook,v=vec): r[:,-1,:]=r[:,-1,:]+v.to(r.dtype); return r
    return m.run_with_hooks(toks, fwd_hooks=[(h,fn)])[0,-1].float()

@torch.no_grad()
def resid_last(prompts):
    h=f"blocks.{L}.hook_resid_post"; out=[]
    for p in prompts:
        _,c=m.run_with_cache(m.to_tokens(p), names_filter=h)
        out.append(c[h][0,-1].float())
    return torch.stack(out)

def grad_wrt_resid(prompt, pos_tok, neg_tok):
    """grad of (logit[pos]-logit[neg]) w.r.t. L8 resid_post last-token vector."""
    h=f"blocks.{L}.hook_resid_post"; saved={}
    def fn(r,hook):
        r.requires_grad_(True); r.retain_grad(); saved['r']=r; return r
    with torch.enable_grad():
        logits = m.run_with_hooks(m.to_tokens(prompt), fwd_hooks=[(h,fn)])
        margin = logits[0,-1,pos_tok]-logits[0,-1,neg_tok]
        g = torch.autograd.grad(margin, saved['r'])[0][0,-1].detach().float()
    return g

def whit_cos(u,v):
    a = float(u @ Sig_inv @ v)
    du = float(u @ Sig_inv @ u); dv = float(v @ Sig_inv @ v)
    if du<=0 or dv<=0: return 0.0
    return abs(a/np.sqrt(du*dv))

def entropy(lg):
    p=torch.softmax(lg,-1); return float(-(p*torch.log(p+1e-12)).sum())

# ---------------- FAMILIES ----------------
# Family A: fruit -> vegetable (validated machinery). Direction from DECOUPLED exemplars.
# Family B: bird -> mammal (a genuinely different swap; injects diversity).
FAMILIES = {
 "fruit": {
   "C":"fruit","Cp":"vegetable",
   "ex_C":["mango","kiwi","melon","apricot","cherry","plum"],
   "ex_Cp":["carrot","potato","broccoli","onion","celery","spinach"],
   "frame":"I bought a fresh {}",
   "cat_probes":["A {e} is a type of","At the store {e}s are sold as a","A {e} is a kind of"],
   "attrs":{
     "color":["red","green","yellow","orange","purple"],
     "taste":["sweet","sour","bitter","salty","spicy"],
     "texture":["juicy","dry","hard","soft","crunchy"],
     "grows":["tree","vine","bush","ground","root"],
   },
   "attr_probe":{"color":"The color of a ripe {e} is usually",
                 "taste":"A {e} tastes","texture":"When you bite a {e} it is",
                 "grows":"A {e} grows on a"},
   "targets":["apple","banana","orange","grape","pear","peach","lemon"],
   "oracle_members":["carrot","potato","broccoli"],
 },
 "bird": {
   "C":"bird","Cp":"mammal",
   "ex_C":["robin","sparrow","hawk","finch","crow","pigeon"],
   "ex_Cp":["wolf","tiger","horse","rabbit","deer","fox"],
   "frame":"I saw a wild {}",
   "cat_probes":["A {e} is a type of","In biology a {e} is a kind of","A {e} is classified as a"],
   "attrs":{
     "cover":["feathers","fur","scales","skin","shell"],
     "move":["flies","swims","runs","crawls","jumps"],
     "young":["eggs","milk","babies","seeds","pups"],
   },
   "attr_probe":{"cover":"A {e} is covered in","move":"To get around a {e}",
                 "young":"A {e} reproduces by laying"},
   "targets":["eagle","owl","duck","goose","parrot","penguin"],
   "oracle_members":["wolf","tiger","horse"],
 },
}
NEUTRAL=["The weather today is","She opened the door and","In the morning I like to",
         "The meeting will start at","He walked down the"]
neut_clean_H = np.mean([entropy(logits_last(p)) for p in NEUTRAL])

def build_dir(fam):
    ex=[fam["frame"].format(w) for w in fam["ex_C"]]
    exp=[fam["frame"].format(w) for w in fam["ex_Cp"]]
    c=resid_last(ex).mean(0); cp=resid_last(exp).mean(0)
    d=cp-c; return d/d.norm()

ALPHAS=[1,2,3,4,5]
records=[]; oracle_vals=[]; rnd_flip_total=0; rnd_flip_ent=0
for fname,fam in FAMILIES.items():
    Ct,Cpt = tid(fam["C"]),tid(fam["Cp"])
    u = build_dir(fam)
    nn = resid_last(NEUTRAL).norm(dim=-1).mean().item()
    # precompute steered neutral entropy per alpha (shared in family)
    steer_H={}
    for a in ALPHAS:
        vec=u*(a*nn/10.0)
        steer_H[a]=np.mean([entropy(logits_last(p,vec)) for p in NEUTRAL])
    # oracle: add Cp direction to GENUINE Cp members' category probe
    for w in fam["oracle_members"]:
        p=f"A {w} is a type of"
        base=float(logits_last(p)[Cpt]-logits_last(p)[Ct])
        vec=u*(5*nn/10.0)
        st=float(logits_last(p,vec)[Cpt]-logits_last(p,vec)[Ct])
        oracle_vals.append(st-base)
    # random control on one target
    for e in fam["targets"][:2]:
        cps=[cp.format(e=e) for cp in fam["cat_probes"]]
        base_ld=[float(logits_last(p)[Cpt]-logits_last(p)[Ct]) for p in cps]
        r=torch.tensor(rng.standard_normal(d_model),dtype=torch.float32); r=r/r.norm()*(4*nn/10.0)
        rnd_flip_ent+=1
        st_ld=[float(logits_last(p,r)[Cpt]-logits_last(p,r)[Ct]) for p in cps]
        if sum(1 for b,s in zip(base_ld,st_ld) if b<0 and s>0)/len(cps) > 0.5: rnd_flip_total+=1

    for e in fam["targets"]:
        et=tid(e)
        if et is None: continue
        cps=[cp.format(e=e) for cp in fam["cat_probes"]]
        base_cat_ld=[float(logits_last(p)[Cpt]-logits_last(p)[Ct]) for p in cps]
        # OURS grad_cat: category-margin (Cp - C) at first cat probe
        g_cat=grad_wrt_resid(cps[0],Cpt,Ct)
        for aname,cands in fam["attrs"].items():
            ap=fam["attr_probe"][aname].format(e=e)
            cand_ids=[tid(w) for w in cands];
            if any(c is None for c in cand_ids): continue
            clean_lg=logits_last(ap)
            clean_arg=int(np.argmax([float(clean_lg[i]) for i in cand_ids]))
            order=np.argsort([float(clean_lg[i]) for i in cand_ids])[::-1]
            top,second=cand_ids[order[0]],cand_ids[order[1]]
            # OURS grad_attr: attribute-margin (top - second) at attr probe
            g_attr=grad_wrt_resid(ap,top,second)
            ours=float(abs(torch.nn.functional.cosine_similarity(g_cat,g_attr,dim=0)))
            # PARK: whitened |cos| swap-dir vs attr-dir in unembedding
            swap_dir=W_U[:,Cpt]-W_U[:,Ct]
            others=[i for i in cand_ids if i!=top]
            attr_dir=W_U[:,top]-W_U[:,others].mean(dim=1)
            park=whit_cos(swap_dir,attr_dir)
            # OUTCOME window: count alphas with FLIP & PRESERVE & COHERENT
            window=0
            for a in ALPHAS:
                vec=u*(a*nn/10.0)
                st_cat_ld=[float(logits_last(p,vec)[Cpt]-logits_last(p,vec)[Ct]) for p in cps]
                FLIP=sum(1 for b,s in zip(base_cat_ld,st_cat_ld) if b<0 and s>0)/len(cps) > 0.5
                st_lg=logits_last(ap,vec)
                PRES=int(np.argmax([float(st_lg[i]) for i in cand_ids]))==clean_arg
                cohR=steer_H[a]/neut_clean_H
                if FLIP and PRES and 0.8<=cohR<=1.2: window+=1
            records.append(dict(fam=fname,ent=e,attr=aname,window=window,ours=ours,park=park))
            print(f"{fname:6} {e:8} {aname:8} win={window} ours={ours:.3f} park={park:.3f}")

# ---------------- STATS ----------------
oracle=float(np.mean(oracle_vals))
W=np.array([r["window"] for r in records],float)
O=np.array([r["ours"] for r in records],float)
P=np.array([r["park"] for r in records],float)
n=len(records)
rho_O=spearmanr(O,W).correlation
rho_P=spearmanr(P,W).correlation
# permutation floor (95th pct |spearman| under label shuffle)
perm=[]
for _ in range(5000):
    Wp=rng.permutation(W); perm.append(abs(spearmanr(O,Wp).correlation))
floor=float(np.nanpercentile(perm,95))
# bootstrap delta CI
boot=[]
idx=np.arange(n)
for _ in range(3000):
    b=rng.choice(idx,n,replace=True)
    ro=abs(spearmanr(O[b],W[b]).correlation); rp=abs(spearmanr(P[b],W[b]).correlation)
    if not(np.isnan(ro) or np.isnan(rp)): boot.append(ro-rp)
ci=(float(np.nanpercentile(boot,2.5)),float(np.nanpercentile(boot,97.5)))

out=dict(n=n, window_mean=float(W.mean()), window_sd=float(W.std()),
         window_dist={int(k):int((W==k).sum()) for k in sorted(set(W))},
         rho_OURS=float(rho_O), rho_PARK=float(rho_P),
         abs_rho_OURS=float(abs(rho_O)), abs_rho_PARK=float(abs(rho_P)),
         delta_abs=float(abs(rho_O)-abs(rho_P)), delta_CI=ci,
         perm_floor95=floor, oracle=oracle,
         ours_range=[float(O.min()),float(O.max())],
         random_flip=f"{rnd_flip_total}/{rnd_flip_ent}")
print("\n=== NUMBERS ==="); print(json.dumps(out,indent=1))

rO,rP=abs(rho_O),abs(rho_P); signO=np.sign(rho_O)
print("\n=== VERDICT ===")
if oracle<1.0: v="BROKEN_MEASUREMENT"
elif rO>=floor and rO>rP and rO>=0.30 and signO<0: v="PROXY_SUPPORTS_V2"
elif rO<floor or signO>=0 or rO<0.15: v="PROXY_REFUTES_V2"
else: v="PROXY_INCONCLUSIVE"
print(v, f"| rO={rO:.3f} rP={rP:.3f} floor={floor:.3f} sign={signO:+.0f} oracle={oracle:+.2f}")
out["verdict"]=v
json.dump(out,open(os.path.join(os.path.dirname(__file__),"sep_boundary_v2_proxy_result.json"),"w"),indent=1)
