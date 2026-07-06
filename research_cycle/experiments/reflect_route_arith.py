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

PREREGISTERED DECISION RULE (frozen before running):
  arms: oracle(true error label) | internal-probe(oof 5-fold logistic on resid,
        layer swept 3/6/9/11) | output(entropy over the 10 digit tokens, the
        natural untrained baseline) | conf-probe(oof logistic on {maxprob,entropy,
        margin}, fairness control) | random.
  per arm: rank steps by wrongness, fix the true errors among the top-B (B=20%);
  final_acc = (orig_correct + true_errors_caught_in_topB)/N ; also error-det AUC.
  * BROKEN_MEASUREMENT if oracle does not top the ranking OR oracle lift < 1pt
    OR either class (<20 correct or <20 wrong) too small.
  * SUPPORTED  iff internal_final - output_final >= +3.0 pts AND paired-bootstrap
               95% CI (2000 resamples over steps) excludes 0 AND both beat random.
  * REFUTED    iff |internal-output| < 3 or CI includes 0, or nothing beats random.
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
rng=np.random.default_rng(0)

def sd(n): return " ".join(list(str(n)))   # space-separated digits -> one token each

@torch.no_grad()
def collect():
    """teacher-force a+b=sum; per answer-digit step gather: error, resid[L], out-feats."""
    steps=[]   # each: dict(err, feats=[maxprob,ent,margin], resid={L:vec})
    hooks=[f"blocks.{L}.hook_resid_post" for L in LAYERS]
    done=0
    for _ in range(N_PROB):
        a=int(rng.integers(0,100)); b=int(rng.integers(0,100)); s=a+b
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

# paired bootstrap on internal_final - output_final (resample steps, re-rank)
diffs=[]
idx=np.arange(N)
for _ in range(2000):
    bi=rng.choice(idx,N,replace=True)
    e=err[bi]
    if e.sum()==0 or e.sum()==len(e): continue
    fi=final_acc(scores["internal"][bi],e,B); fo=final_acc(scores["output"][bi],e,B)
    diffs.append((fi-fo)*100)
ci=(float(np.percentile(diffs,2.5)),float(np.percentile(diffs,97.5))) if diffs else (float('nan'),)*2
delta=(fa["internal"]-fa["output"])*100
oracle_lift=(fa["oracle"]-no_reflect)*100
internal_beats_rand=fa["internal"]>fa["random"]; output_beats_rand=fa["output"]>fa["random"]

if fa["oracle"]<max(fa["internal"],fa["output"],fa["random"]) or oracle_lift<1.0 or n_ok<20 or n_err<20:
    verdict="BROKEN_MEASUREMENT"
elif delta>=3.0 and ci[0]>0 and internal_beats_rand and output_beats_rand:
    verdict="SUPPORTED"
else:
    verdict="REFUTED"

out=dict(model=MODEL,task="2-3 digit addition, per-answer-digit steps",N=N,n_err=n_err,
         best_internal_layer=best_L,internal_AUC_by_layer={L:auc(internal_by_L[L]) for L in LAYERS},
         final_acc={**{a:fa[a] for a in arms},"no_reflection":no_reflect},
         AUC={a:aucs[a] for a in arms},
         internal_minus_output_pts=delta,delta_CI=ci,oracle_lift_pts=oracle_lift,
         internal_beats_random=bool(internal_beats_rand),output_beats_random=bool(output_beats_rand),
         verdict=verdict)
print("\n=== NUMBERS ==="); print(json.dumps(out,indent=1))
print("\n=== VERDICT ===",verdict,f"| internal-output={delta:+.2f}pts CI[{ci[0]:+.2f},{ci[1]:+.2f}] oracle_lift={oracle_lift:+.2f}")
dst=os.path.join(os.path.dirname(__file__),"..","campaigns","reflect-route","rr2_arith_result.json")
json.dump(out,open(dst,"w"),indent=1); print("[saved]",os.path.abspath(dst),flush=True)
