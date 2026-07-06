#!/usr/bin/env python3
"""
Reflection-routing, FAITHFUL to the order (author: Илья, hand-written after the
loop's tester substituted the ordered ARITHMETIC task with easy next-token-on-
wikitext — the verifier caught it, verdict voided to inconclusive). This runs the
ORDERED task: 2-3 digit ADDITION, where a base LM is often CONFIDENTLY WRONG (low
output entropy, still wrong) — the regime where an internal signal could beat
output confidence. No task substitution.

QUESTION: at a FIXED reflection budget B, does routing the budget to the steps an
INTERNAL-activation probe scores as most-likely-wrong beat routing by the model's
OUTPUT confidence (and both beat random)? Step = one answer DIGIT position.

rr3 CORRECTION (frozen before this run): rr2 used final-acc@B as primary, but a
91.6% error rate SATURATED it (oracle−random dynamic range ~2pt, so the +3pt bar
was unreachable regardless of the true effect — a design flaw). The a-priori-correct
metric for "do internal states LOCATE errors better than output confidence" is the
error-detection AUC, which is NOT budget-saturated. rr3 makes AUC the PRIMARY rule,
adds a label-permutation leakage control, uses fresh data (seed 1) + an easier
difficulty mix to un-saturate the secondary final-acc. This is a metric correction
(not goalpost-moving): AUC is the right lens, the rule is frozen before running,
fresh data, plus a negative control.

PREREGISTERED DECISION RULE (frozen before running):
  arms: oracle(true error label) | internal-probe(oof 5-fold logistic on resid,
        layer swept 3/6/9/11) | output(entropy over the 10 digit tokens, natural
        untrained baseline) | conf-probe(oof logistic on {maxprob,entropy,margin}) | random.
  PRIMARY metric = error-detection AUC per arm. SECONDARY = final-acc@B=20%.
  * BROKEN_MEASUREMENT if oracle AUC < 0.98 OR oracle final-lift < 1pt OR either
    class (<20 correct or <20 wrong) too small OR permutation-control AUC > 0.60
    (probe leaks on shuffled labels → pipeline invalid).
  * SUPPORTED  iff internal_AUC - output_AUC >= +0.05 AND paired-bootstrap 95% CI
               (2000 resamples over steps) excludes 0 AND internal_AUC > 0.55.
  * REFUTED    iff internal within noise of output (CI includes 0) or internal_AUC
               <= chance or the +0.05 margin is not met.
rr4 CORRECTION (frozen before this run): rr3's rule had a BACKWARDS clause
"output_AUC > 0.50" — it punished the pro-hypothesis case (output confidence being
USELESS is the hypothesis in its strongest form), so a clean +0.298 AUC win was
mislabeled REFUTED. rr4 removes ONLY that clause; the fairness safeguard is the
PERMUTATION control (internal probe on shuffled labels must be ~chance), which stays.
Fresh data (seed 2). No other change — this is a single-fix confirmation, not a weakening.
Error label uses the model's best-DIGIT guess (argmax over the 10 digit tokens) vs
the true digit — the faithful "which digit does the model think it is" for arithmetic.
"""
import os, json, numpy as np, torch
torch.set_num_threads(max(1, os.cpu_count() or 4))
from transformer_lens import HookedTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

MODEL="gpt2"; LAYERS=[3,6,9,11]; B=0.20; N_PROB=600
print(f"[load] {MODEL} (offline)...", flush=True)
m = HookedTransformer.from_pretrained(MODEL, device="cpu"); m.eval()
assert m.cfg.n_layers == 12, f"unexpected model: {m.cfg.n_layers} layers"
DIG=[m.to_single_token(" "+str(d)) for d in range(10)]   # single-token digits
rng=np.random.default_rng(2)                              # rr4: FRESH data again (clean confirmation)

def sd(n): return " ".join(list(str(n)))   # space-separated digits -> one token each

@torch.no_grad()
def collect():
    """teacher-force a+b=sum; per answer-digit step gather: error, resid[L], out-feats."""
    steps=[]   # each: dict(err, feats=[maxprob,ent,margin], resid={L:vec})
    hooks=[f"blocks.{L}.hook_resid_post" for L in LAYERS]
    done=0
    for _ in range(N_PROB):
        if rng.random()<0.5:                              # rr3 difficulty MIX to un-saturate error rate
            a=int(rng.integers(0,10)); b=int(rng.integers(0,10))
        else:
            a=int(rng.integers(0,50)); b=int(rng.integers(0,50))
        s=a+b
        prompt=f"{sd(a)} + {sd(b)} ="
        ans=[int(c) for c in str(s)]
        pids=m.to_tokens(prompt)[0]                      # includes BOS
        aids=torch.tensor([DIG[d] for d in ans])
        full=torch.cat([pids,aids]).unsqueeze(0)
        logits,cache=m.run_with_cache(full, names_filter=lambda n:n in hooks)
        start=len(pids)
        for k,d in enumerate(ans):
            p=start+k                                    # position of this answer digit
            lg=logits[0,p-1,DIG].float()                 # logits over the 10 digits, from the predicting position
            prob=torch.softmax(lg,-1)
            pred=int(torch.argmax(lg))
            err=1 if pred!=d else 0
            srt=torch.sort(prob,descending=True).values
            feats=[float(prob.max()), float(-(prob*torch.log(prob+1e-12)).sum()), float(srt[0]-srt[1])]
            resid={L:cache[f"blocks.{L}.hook_resid_post"][0,p-1].float().numpy() for L in LAYERS}
            steps.append(dict(err=err,feats=feats,resid=resid))
        done+=1
        if done%100==0: print(f"[collect] {done}/{N_PROB} problems, {len(steps)} steps", flush=True)
    return steps

def oof_scores(X,y):
    """out-of-fold P(err) from 5-fold logistic; higher = more likely wrong."""
    y=np.asarray(y); out=np.zeros(len(y))
    skf=StratifiedKFold(5,shuffle=True,random_state=0)
    for tr,te in skf.split(X,y):
        sc=StandardScaler().fit(X[tr])
        clf=LogisticRegression(max_iter=1000,C=0.5).fit(sc.transform(X[tr]),y[tr])
        out[te]=clf.predict_proba(sc.transform(X[te]))[:,1]
    return out

def final_acc(score,err,B):
    """route budget B to top-scored steps; fix true errors caught there."""
    n=len(err); k=int(round(B*n))
    order=np.argsort(-score)                # most-likely-wrong first
    top=set(order[:k].tolist())
    caught=sum(1 for i in range(n) if err[i]==1 and i in top)
    orig_correct=int((err==0).sum())
    return (orig_correct+caught)/n

steps=collect()
err=np.array([s["err"] for s in steps])
N=len(steps); n_err=int(err.sum()); n_ok=N-n_err
print(f"\n[data] N={N} steps · errors={n_err} ({n_err/N:.1%}) · correct={n_ok}", flush=True)

feats=np.array([s["feats"] for s in steps])              # [maxprob,ent,margin]
# arms' wrongness scores
scores={}
scores["oracle"]=err.astype(float)
scores["random"]=rng.random(N)
scores["output"]=feats[:,1]                              # entropy: high -> likely wrong (natural baseline)
scores["conf-probe"]=oof_scores(feats,err)              # trained on all 3 output feats (fairness)
best_L=None; best_auc=-1; internal_by_L={}
for L in LAYERS:
    X=np.array([s["resid"][L] for s in steps])
    sc=oof_scores(X,err)
    internal_by_L[L]=sc
    try: auc=roc_auc_score(err,sc)
    except Exception: auc=float('nan')
    print(f"[internal] layer {L}: AUC={auc:.3f}", flush=True)
    if auc>best_auc: best_auc=auc; best_L=L
scores["internal"]=internal_by_L[best_L]

def auc(s):
    try: return float(roc_auc_score(err,s))
    except Exception: return float('nan')
arms=["oracle","internal","conf-probe","output","random"]
fa={a:final_acc(scores[a],err,B) for a in arms}
no_reflect=n_ok/N
aucs={a:auc(scores[a]) for a in arms}
print("\n=== arms (final acc @B=20% · error-det AUC) ===")
print(f"  no-reflection      acc={no_reflect*100:.2f}")
for a in arms: print(f"  {a:16} acc={fa[a]*100:.2f}  AUC={aucs[a]:.3f}")

# ---- PRIMARY: paired bootstrap on internal_AUC - output_AUC (resample steps) ----
si=scores["internal"]; so=scores["output"]; idx=np.arange(N)
adiffs=[]
for _ in range(2000):
    bi=rng.choice(idx,N,replace=True); e=err[bi]
    if e.sum()<2 or (len(e)-e.sum())<2: continue
    try: adiffs.append(roc_auc_score(e,si[bi])-roc_auc_score(e,so[bi]))
    except Exception: pass
auc_ci=(float(np.percentile(adiffs,2.5)),float(np.percentile(adiffs,97.5))) if adiffs else (float('nan'),)*2
auc_delta=aucs["internal"]-aucs["output"]

# NEGATIVE CONTROL: train the internal probe on SHUFFLED labels — must collapse to ~chance
yperm=rng.permutation(err)
Xbest=np.array([s["resid"][best_L] for s in steps])
try: perm_auc=float(roc_auc_score(yperm,oof_scores(Xbest,yperm)))
except Exception: perm_auc=float('nan')

# SECONDARY: final-acc@B (informative once un-saturated)
delta=(fa["internal"]-fa["output"])*100
oracle_lift=(fa["oracle"]-no_reflect)*100

if aucs["oracle"]<0.98 or oracle_lift<1.0 or n_ok<20 or n_err<20 or perm_auc>0.60:
    verdict="BROKEN_MEASUREMENT"
elif auc_delta>=0.05 and auc_ci[0]>0 and aucs["internal"]>0.55:
    verdict="SUPPORTED"
else:
    verdict="REFUTED"

out=dict(model=MODEL,task="addition (mixed 1-2 digit), per-answer-digit steps",N=N,n_err=n_err,err_rate=n_err/N,
         best_internal_layer=best_L,internal_AUC_by_layer={L:auc(internal_by_L[L]) for L in LAYERS},
         AUC={a:aucs[a] for a in arms},
         primary_internal_minus_output_AUC=auc_delta, AUC_delta_CI=auc_ci,
         permutation_control_AUC=perm_auc,
         final_acc={**{a:fa[a] for a in arms},"no_reflection":no_reflect},
         secondary_internal_minus_output_pts=delta, oracle_lift_pts=oracle_lift,
         verdict=verdict)
print(f"\n[neg-control] internal probe on shuffled labels: AUC={perm_auc:.3f} (must be ~0.5)", flush=True)
print("\n=== NUMBERS ==="); print(json.dumps(out,indent=1))
print("\n=== VERDICT ===",verdict,f"| PRIMARY internal_AUC-output_AUC={auc_delta:+.3f} CI[{auc_ci[0]:+.3f},{auc_ci[1]:+.3f}] · oracle_AUC={aucs['oracle']:.3f} perm={perm_auc:.3f}")
dst=os.path.join(os.path.dirname(__file__),"..","campaigns","reflect-route","rr4_arith_result.json")
json.dump(out,open(dst,"w"),indent=1); print("[saved]",os.path.abspath(dst),flush=True)
