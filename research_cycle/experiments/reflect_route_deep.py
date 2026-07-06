#!/usr/bin/env python3
"""Step 2 of DEEP reflect-route: run reflection-routing on the FINE-TUNED gpt2
(partial competence ~48% digit error — an UN-saturated regime), so we can measure
not just DETECTION (does internal beat output at finding errors) but the practical
GAIN (does internal routing actually raise final accuracy more than output routing).

Loads the fine-tuned checkpoint into transformer_lens; a sanity gate aborts if the
loaded model's error rate isn't ~the training band (else the weights didn't load).

PREREGISTERED DECISION RULE (frozen before running):
  arms: oracle | internal-probe(oof 5-fold logistic on resid, layer swept 3/6/9/11)
        | output(entropy over the 10 digit tokens) | conf-probe | random.
  PRIMARY = practical GAIN: final_acc(internal) - final_acc(output) at B=20% (now
    un-saturated). SECONDARY/replication = error-detection AUC (internal vs output).
  * BROKEN_MEASUREMENT if loaded err_rate not in [0.30,0.70] (bad load/regime) OR
    oracle final-lift < 1pt OR oracle AUC < 0.98 OR class too small (<40 either)
    OR permutation-control AUC > 0.60.
  * SUPPORTED iff GAIN >= +1.5 pts AND paired-bootstrap 95% CI (2000 resamples)
    excludes 0 AND final_acc(internal) > final_acc(random).
  * REFUTED otherwise.
(Detection AUC reported alongside as replication of rr4's confirmed finding.)
"""
import os, json, numpy as np, torch
torch.set_num_threads(max(1,os.cpu_count() or 4))
from transformer_lens import HookedTransformer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

CKPT=os.environ["CKPT_DIR"]; LAYERS=[3,6,9,11]; B=0.20; N_PROB=700
print(f"[load] fine-tuned gpt2 from {CKPT}", flush=True)
hf=GPT2LMHeadModel.from_pretrained(CKPT)
tok=GPT2TokenizerFast.from_pretrained(CKPT)
m=HookedTransformer.from_pretrained("gpt2", hf_model=hf, tokenizer=tok, device="cpu"); m.eval()
DIG=[m.to_single_token(" "+str(d)) for d in range(10)]
rng=np.random.default_rng(3)

def sd(n): return " ".join(list(str(n)))

@torch.no_grad()
def collect():
    steps=[]; hooks=[f"blocks.{L}.hook_resid_post" for L in LAYERS]; done=0
    for _ in range(N_PROB):
        a=int(rng.integers(0,100)); b=int(rng.integers(0,100)); s=a+b
        prompt=f"{sd(a)} + {sd(b)} ="
        ans=[int(c) for c in str(s)]
        pids=m.to_tokens(prompt,prepend_bos=False)[0]   # match fine-tuning: NO start token (with BOS the model degrades 48%->74%)
        aids=torch.tensor([DIG[d] for d in ans])
        full=torch.cat([pids,aids]).unsqueeze(0)
        logits,cache=m.run_with_cache(full, names_filter=lambda n:n in hooks)
        start=len(pids)
        for k,d in enumerate(ans):
            p=start+k
            lg=logits[0,p-1,DIG].float(); prob=torch.softmax(lg,-1)
            pred=int(torch.argmax(lg)); err=1 if pred!=d else 0
            srt=torch.sort(prob,descending=True).values
            feats=[float(prob.max()), float(-(prob*torch.log(prob+1e-12)).sum()), float(srt[0]-srt[1])]
            resid={L:cache[f"blocks.{L}.hook_resid_post"][0,p-1].float().numpy() for L in LAYERS}
            steps.append(dict(err=err,feats=feats,resid=resid))
        done+=1
        if done%150==0: print(f"[collect] {done}/{N_PROB} · {len(steps)} steps", flush=True)
    return steps

def oof(X,y):
    y=np.asarray(y); out=np.zeros(len(y))
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(X,y):
        sc=StandardScaler().fit(X[tr]); clf=LogisticRegression(max_iter=1000,C=0.5).fit(sc.transform(X[tr]),y[tr])
        out[te]=clf.predict_proba(sc.transform(X[te]))[:,1]
    return out
def final_acc(score,err,B):
    n=len(err); k=int(round(B*n)); top=set(np.argsort(-score)[:k].tolist())
    caught=sum(1 for i in range(n) if err[i]==1 and i in top)
    return (int((err==0).sum())+caught)/n
def auc(s):
    try: return float(roc_auc_score(err,s))
    except Exception: return float('nan')

steps=collect()
err=np.array([s["err"] for s in steps]); N=len(steps); n_err=int(err.sum()); n_ok=N-n_err
er=n_err/N
print(f"\n[data] N={N} · errors={n_err} ({er:.1%}) · correct={n_ok}  [sanity: expect ~0.48]", flush=True)
feats=np.array([s["feats"] for s in steps])

scores={"oracle":err.astype(float),"random":rng.random(N),"output":feats[:,1],"conf-probe":oof(feats,err)}
best_L=None; best_auc=-1; internal_by_L={}
for L in LAYERS:
    X=np.array([s["resid"][L] for s in steps]); sc=oof(X,err); internal_by_L[L]=sc
    a=roc_auc_score(err,sc); print(f"[internal] layer {L}: AUC={a:.3f}",flush=True)
    if a>best_auc: best_auc=a; best_L=L
scores["internal"]=internal_by_L[best_L]

arms=["oracle","internal","conf-probe","output","random"]
fa={a:final_acc(scores[a],err,B) for a in arms}; no_reflect=n_ok/N
aucs={a:auc(scores[a]) for a in arms}
print("\n=== arms (final acc @B=20% · detection AUC) ===")
print(f"  no-reflection      acc={no_reflect*100:.2f}")
for a in arms: print(f"  {a:16} acc={fa[a]*100:.2f}  AUC={aucs[a]:.3f}",flush=True)

# PRIMARY: gain bootstrap (internal_final - output_final)
idx=np.arange(N); gdiffs=[]; adiffs=[]
for _ in range(2000):
    bi=rng.choice(idx,N,replace=True); e=err[bi]
    if e.sum()<2 or (len(e)-e.sum())<2: continue
    gdiffs.append((final_acc(scores["internal"][bi],e,B)-final_acc(scores["output"][bi],e,B))*100)
    try: adiffs.append(roc_auc_score(e,scores["internal"][bi])-roc_auc_score(e,scores["output"][bi]))
    except Exception: pass
gain=(fa["internal"]-fa["output"])*100
gain_ci=(float(np.percentile(gdiffs,2.5)),float(np.percentile(gdiffs,97.5))) if gdiffs else (float('nan'),)*2
auc_delta=aucs["internal"]-aucs["output"]
auc_ci=(float(np.percentile(adiffs,2.5)),float(np.percentile(adiffs,97.5))) if adiffs else (float('nan'),)*2
# negative control
yperm=rng.permutation(err); Xb=np.array([s["resid"][best_L] for s in steps])
try: perm_auc=float(roc_auc_score(yperm,oof(Xb,yperm)))
except Exception: perm_auc=float('nan')
oracle_lift=(fa["oracle"]-no_reflect)*100
print(f"\n[neg-control] internal probe on shuffled labels AUC={perm_auc:.3f} (must be ~0.5)",flush=True)

if not(0.30<=er<=0.70) or oracle_lift<1.0 or aucs["oracle"]<0.98 or n_ok<40 or n_err<40 or perm_auc>0.60:
    verdict="BROKEN_MEASUREMENT"
elif gain>=1.5 and gain_ci[0]>0 and fa["internal"]>fa["random"]:
    verdict="SUPPORTED"
else:
    verdict="REFUTED"

out=dict(model="gpt2 fine-tuned on addition (partial competence)",task="addition 0-99, per-answer-digit steps",
         N=N,n_err=n_err,err_rate=er,best_internal_layer=best_L,
         final_acc={**{a:fa[a] for a in arms},"no_reflection":no_reflect},
         AUC={a:aucs[a] for a in arms},
         PRIMARY_gain_internal_minus_output_pts=gain, gain_CI=gain_ci,
         detection_internal_minus_output_AUC=auc_delta, AUC_delta_CI=auc_ci,
         permutation_control_AUC=perm_auc, oracle_lift_pts=oracle_lift, verdict=verdict)
print("\n=== NUMBERS ==="); print(json.dumps(out,indent=1))
print("\n=== VERDICT ===",verdict,f"| PRIMARY gain(internal-output)={gain:+.2f}pts CI[{gain_ci[0]:+.2f},{gain_ci[1]:+.2f}] · detection ΔAUC={auc_delta:+.3f} · oracle_lift={oracle_lift:+.2f} perm={perm_auc:.3f}")
dst=os.path.join(os.path.dirname(__file__),"..","campaigns","reflect-route","rr5_deep_result.json")
json.dump(out,open(dst,"w"),indent=1); print("[saved]",os.path.abspath(dst),flush=True)
