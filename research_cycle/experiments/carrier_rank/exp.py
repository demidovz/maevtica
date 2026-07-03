#!/usr/bin/env python3
"""Carrier rank rho(P) vs SAE feature-splitting. Prereg: ./PREREG.md"""
from __future__ import annotations
import glob, json, os
import numpy as np, torch
CENTER = os.environ.get("CENTER") == "1"
OUT = "result2.json" if CENTER else "result.json"
torch.set_grad_enabled(False); torch.set_num_threads(8)
from transformer_lens import HookedTransformer
from safetensors.torch import load_file

L = 7
HOOK = f"blocks.{L}.hook_resid_pre"
rng = np.random.default_rng(0)

print("loading gpt2 ...", flush=True)
m = HookedTransformer.from_pretrained("gpt2", device="cpu"); m.eval()
d = m.cfg.d_model

NEUTRAL = ["The","It was","I think that","We went to","There is a","People often say"]
CONCEPT = {
 "dog":  ["The dog barked at the","My dog loves to","A puppy and its","The dog wagged its tail","Dogs are loyal","She walked her dog in the"],
 "war":  ["The war between the two","Soldiers fought in the","The battle raged on the","War brings destruction and","The army prepared for war","Generals planned the war"],
 "love": ["I love you so","She fell in love with","Love is a powerful","They shared a deep love","He wrote a love letter to","Love conquers all and"],
 "music":["The orchestra played a beautiful","She listened to music on her","The melody of the song was","He played the guitar and sang","Music filled the concert hall","The band performed their new song"],
 "snow": ["Snow fell softly on the","The children built a snowman in","The blizzard covered the roads with","White snow blanketed the quiet","He shoveled the snow from the","Snowflakes drifted past the frozen"],
}
CORPUS = [
 "The weather today is quite","I went to the store to buy","She looked at him and",
 "In the news this morning a","The scientist explained that the","History teaches us that every",
 "My favorite thing about summer is","The old house at the end of","He opened the letter and read",
 "Economists worry that the market","The recipe calls for two cups of","Children love to play in the",
 "The government announced a new","After the long journey they finally","Music has the power to",
 "The mountain rose sharply above","During the meeting the manager","A cold wind blew across the",
 "The book on the table was","Technology continues to change how","The garden was full of",
 "When the sun set over the","The team celebrated their victory","Doctors recommend that patients",
 "The river flowed gently past","At the airport travelers waited","The painting showed a quiet",
 "Farmers depend on the rain to","The city streets were crowded with","Under the ocean strange creatures",
]

def last_tok_acts(prompts):
    out=[]
    for p in prompts:
        _,c=m.run_with_cache(m.to_tokens(p), names_filter=HOOK)
        out.append(c[HOOK][0,-1])
    return torch.stack(out).float()

def all_tok_acts(prompts):
    out=[]
    for p in prompts:
        _,c=m.run_with_cache(m.to_tokens(p), names_filter=HOOK)
        out.append(c[HOOK][0])
    return torch.cat(out,0).float()

print("collecting activations ...", flush=True)
neutral_mean = last_tok_acts(NEUTRAL).mean(0)
concept_acts = {c: last_tok_acts(ps) for c,ps in CONCEPT.items()}
concept_diff = {c: (a.mean(0)-neutral_mean).numpy() for c,a in concept_acts.items()}
if CENTER:
    shared = np.mean(list(concept_diff.values()), axis=0)
    concept_diff = {c: v - shared for c,v in concept_diff.items()}
u = {c: v/np.linalg.norm(v) for c,v in concept_diff.items()}

# ---- SAE atoms (live only) ----
base = glob.glob("/home/friemann/.cache/huggingface/hub/models--jbloom--GPT2-Small-SAEs-Reformatted/snapshots/*/"+HOOK)[0]
sd = load_file(base+"/sae_weights.safetensors")
W_dec = sd["W_dec"].float(); W_enc = sd["W_enc"].float(); b_enc = sd["b_enc"].float(); b_dec = sd["b_dec"].float()
Xc = all_tok_acts(CORPUS)
feat = torch.relu((Xc - b_dec) @ W_enc + b_enc)
live = (feat.mean(0) > 1e-6).numpy()
A = W_dec.numpy()
A = A / (np.linalg.norm(A,axis=1,keepdims=True)+1e-12)
A_live = A[live]
print(f"live atoms: {live.sum()}/{A.shape[0]}", flush=True)

def metrics(v):
    v = v/ (np.linalg.norm(v)+1e-12)
    cos = np.abs(A_live @ v)
    return dict(maxcos=float(cos.max()),
                count03=int((cos>0.3).sum()), count05=int((cos>0.5).sum()),
                count08=int((cos>0.8).sum()),
                top5=[round(float(x),3) for x in np.sort(cos)[-5:][::-1]])

# ---- ORACLES ----
# O1: a known live atom through the pipeline
probe_atom = A_live[int(rng.integers(0, A_live.shape[0]))]
o1 = metrics(probe_atom)
# O2 null: 200 random unit dirs
nulls = rng.standard_normal((200,d)).astype(np.float32)
nulls /= np.linalg.norm(nulls,axis=1,keepdims=True)
null_max = np.abs(nulls @ A_live.T).max(1)
null_995 = float(np.quantile(null_max, 0.995))
o2 = {c: metrics(u[c])["maxcos"] for c in CONCEPT}
oracle_ok = o1["maxcos"] > 0.99 and all(v > null_995 for v in o2.values())

# ---- validity gates ----
names = list(CONCEPT)
pw = {}
v1_ok = True
for i in range(len(names)):
    for j in range(i+1,len(names)):
        cij = float(abs(np.dot(u[names[i]],u[names[j]])))
        pw[f"{names[i]}~{names[j]}"] = round(cij,3)
        if cij >= 0.5: v1_ok = False

def part_ratio(mat):
    s = np.linalg.svd(mat, compute_uv=False)
    return float((s**2).sum()**2/ (s**4).sum())

PROPS = [(1,["dog"]),(1,["war"]),(1,["love"]),(1,["music"]),(1,["snow"]),
         (2,["dog","war"]),(2,["love","music"]),(2,["war","snow"]),
         (3,["dog","war","love"]),(3,["music","snow","war"])]
pr3 = {"+".join(cs): part_ratio(np.stack([concept_diff[c] for c in cs]))
       for rho,cs in PROPS if rho==3}
v2_ok = all(v >= 2.0 for v in pr3.values())

# ---- property metrics ----
results=[]
for rho, cs in PROPS:
    if CENTER:
        pooled = np.mean([concept_diff[c] for c in cs], axis=0)
    else:
        pooled = np.stack([concept_acts[c].numpy() for c in cs]).reshape(-1,d).mean(0) - neutral_mean.numpy()
    mt = metrics(pooled)
    # secondary: subspace projection
    B = np.stack([u[c] for c in cs]).T           # [d,rho]
    Q,_ = np.linalg.qr(B)
    proj = np.linalg.norm(A_live @ Q, axis=1)
    mt["subspace_maxproj"] = float(proj.max())
    mt["subspace_count03"] = int((proj>0.3).sum())
    results.append(dict(rho=rho, prop="+".join(cs), **mt))

def spearman(x,y):
    def rank(a):
        a=np.asarray(a,float); o=np.argsort(a); r=np.empty(len(a))
        i=0
        while i<len(a):
            j=i
            while j+1<len(a) and a[o[j+1]]==a[o[i]]: j+=1
            r[o[i:j+1]] = (i+j)/2.0; i=j+1
        return r
    rx,ry = rank(x),rank(y)
    rx-=rx.mean(); ry-=ry.mean()
    return float((rx*ry).sum()/(np.sqrt((rx**2).sum()*(ry**2).sum())+1e-12))

rhos=[r["rho"] for r in results]
r_count = spearman(rhos,[r["count03"] for r in results])
r_max   = spearman(rhos,[r["maxcos"] for r in results])
mean_c1 = float(np.mean([r["count03"] for r in results if r["rho"]==1]))
mean_c3 = float(np.mean([r["count03"] for r in results if r["rho"]==3]))
f2 = any(r["maxcos"]>=0.8 for r in results if r["rho"]==3)
rho1_no_dominant = sum(1 for r in results if r["rho"]==1 and r["maxcos"]<=0.5)

if not oracle_ok: verdict="BROKEN_MEASUREMENT"
elif not (v1_ok and v2_ok): verdict="INCONCLUSIVE_DESIGN"
elif (r_count<=0.3 or mean_c3<=mean_c1) or f2: verdict="REFUTED"
elif r_count>=0.6 and r_max<=-0.3 and not f2: verdict="SUPPORTED"
else: verdict="INCONCLUSIVE"

report=dict(oracle=dict(o1_atom_maxcos=round(o1["maxcos"],4), null_995=round(null_995,4),
                        concept_maxcos={k:round(v,3) for k,v in o2.items()}, oracle_ok=bool(oracle_ok)),
            validity=dict(pairwise_cos=pw, v1_ok=bool(v1_ok), pr_rho3=pr3, v2_ok=bool(v2_ok)),
            live_atoms=int(live.sum()),
            properties=results,
            stats=dict(spearman_rho_count03=round(r_count,3), spearman_rho_maxcos=round(r_max,3),
                       mean_count03_rho1=mean_c1, mean_count03_rho3=mean_c3,
                       any_rho3_maxcos_ge_08=bool(f2), rho1_props_without_atom_gt05=rho1_no_dominant),
            verdict=verdict)
print("=== REPORT ===")
print(json.dumps(report, indent=1))
json.dump(report, open(__file__.rsplit("/",1)[0]+"/"+OUT,"w"), indent=1)
