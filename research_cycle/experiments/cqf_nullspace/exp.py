#!/usr/bin/env python3
"""Causal Quotient Feature — null-space-shifted copies. Per PREREG.md.

venv: ~/.local/state/mst/crc-venv311/bin/python
"""
import os, json, numpy as np, torch

torch.set_num_threads(max(1, os.cpu_count() or 4))
torch.set_grad_enabled(False)
rng = np.random.default_rng(0)
torch.manual_seed(0)

HERE = os.path.dirname(os.path.abspath(__file__))
SAE_PATH = os.environ.get(
    "SAE32K",
    "/tmp/claude-1000/-home-friemann-workspace-maestratica/"
    "b9bd7464-4c05-40a5-ba2b-b36c0ae2836c/scratchpad/saes/sae32k_L6.pt")
LAYER, D, TOPK = 6, 768, 32
HOOK = f"blocks.{LAYER}.hook_resid_post"
NEUTRAL = [
    "The weather today is",
    "In the middle of the city there was a",
    "She looked at him and said",
    "The most important thing about science is",
]

def log(*a): print(*a, flush=True)

# ---------------- concept sentences (last token carries concept) ----------
FRAMES_ADJ = [
    "The movie we watched last night was truly {}",
    "Everyone agreed that the dinner was {}",
    "Critics said the new album sounds {}",
    "Honestly, the whole trip felt {}",
    "The service at that hotel was {}",
    "Her performance in the play was {}",
    "The ending of the book was {}",
    "People kept saying the party was {}",
]
POS_ADJ = ["wonderful", "fantastic", "amazing", "delightful", "excellent", "superb"]
NEG_ADJ = ["terrible", "awful", "horrible", "dreadful", "disappointing", "miserable"]

FRAMES_TENSE_PAST = [
    "Yesterday the children happily {}",
    "Last week my neighbors loudly {}",
    "During the storm the sailors {}",
    "Long ago the villagers often {}",
    "That evening the students quietly {}",
    "Back then the workers usually {}",
    "After dinner the guests {}",
    "In those days the farmers {}",
]
PAST_V = ["played", "danced", "worked", "sang", "walked", "talked"]
PRES_V = ["play", "dance", "work", "sing", "walk", "talk"]
FRAMES_TENSE_PRES = [f.replace("Yesterday", "Today").replace("Last week", "This week")
                     .replace("During the storm", "During storms")
                     .replace("Long ago", "Nowadays").replace("That evening", "Every evening")
                     .replace("Back then", "These days").replace("After dinner", "Every day")
                     .replace("In those days", "These days") for f in FRAMES_TENSE_PAST]

FRAMES_NOUN = [
    "She wrote a long story about the old {}",
    "Down the road we suddenly saw a {}",
    "The photograph showed a very large {}",
    "My uncle spent all his money on a {}",
    "Behind the barn there was a small {}",
    "The children drew a picture of a {}",
    "In the museum they displayed an ancient {}",
    "He could not stop talking about the {}",
]
ANIMALS = ["cat", "dog", "horse", "rabbit", "wolf", "sheep"]
VEHICLES = ["truck", "car", "tractor", "bus", "wagon", "bicycle"]

def build(frames, words):
    out = []
    for i in range(24):
        out.append(frames[i % len(frames)].format(words[i % len(words)]))
    return out

CONCEPTS = {
    "sentiment": (build(FRAMES_ADJ, POS_ADJ), build(FRAMES_ADJ, NEG_ADJ)),
    "tense": (build(FRAMES_TENSE_PAST, PAST_V), build(FRAMES_TENSE_PRES, PRES_V)),
    "animal_vehicle": (build(FRAMES_NOUN, ANIMALS), build(FRAMES_NOUN, VEHICLES)),
}

# ---------------- model + activations ----------------
from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained("gpt2", device="cpu"); model.eval()
V = model.cfg.d_vocab

def resid_last(prompts):
    out = []
    for p in prompts:
        _, c = model.run_with_cache(model.to_tokens(p), names_filter=HOOK)
        out.append(c[HOOK][0, -1].float())
    return torch.stack(out)

# ---------------- SAE ----------------
sd = torch.load(SAE_PATH, map_location="cpu")
keys = {k.lower(): k for k in sd}
def get(*names):
    for n in names:
        if n in sd: return sd[n].float()
        if n.lower() in keys: return sd[keys[n.lower()]].float()
    return None
ENC = get("encoder.weight", "W_enc")   # [L,768] or [768,L]
DEC = get("decoder.weight", "W_dec")
PRE = get("pre_bias", "b_dec")
LAT = get("latent_bias", "b_enc")
if ENC.shape[0] == D and ENC.shape[1] != D: ENC = ENC.T   # -> [L,768]
if DEC.shape[0] != D and DEC.shape[1] == D: DEC = DEC.T   # -> [768,L]
log("SAE enc", tuple(ENC.shape), "dec", tuple(DEC.shape))

def sae_acts(X):
    H = (X - (PRE if PRE is not None else 0)) @ ENC.T
    if LAT is not None: H = H + LAT
    A = torch.zeros_like(H)
    tv, ti = H.topk(TOPK, dim=-1)
    A.scatter_(-1, ti, tv.clamp(min=0))
    return A

# ---------------- objects per concept ----------------
def probe_fit(Xtr, ytr):
    w = torch.zeros(D, requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    Xs = Xtr / Xtr.norm(dim=-1, keepdim=True).mean()
    with torch.enable_grad():
        for _ in range(600):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                Xs @ w + b, ytr) + 1e-2 * w.pow(2).sum()
            loss.backward(); opt.step()
    return w.detach(), b.detach(), Xs

objects, o3 = {}, {}
for name, (pos, neg) in CONCEPTS.items():
    Xp, Xn = resid_last(pos), resid_last(neg)
    idx = rng.permutation(24)
    tr, te = idx[:18], idx[18:]
    Xtr = torch.cat([Xp[tr], Xn[tr]]); ytr = torch.cat([torch.ones(18), torch.zeros(18)])
    Xte = torch.cat([Xp[te], Xn[te]]); yte = torch.cat([torch.ones(6), torch.zeros(6)])
    m = Xp[tr].mean(0) - Xn[tr].mean(0)
    w, b, Xs = probe_fit(Xtr, ytr)
    scale = Xtr.norm(dim=-1, keepdim=True).mean()
    acc = (((Xte / scale) @ w + b > 0).float() == yte).float().mean().item()
    Ap, An = sae_acts(Xp[tr]), sae_acts(Xn[tr])
    diff = (Ap.mean(0) - An.mean(0))
    li = int(diff.argmax())
    s = DEC[:, li].clone()
    u = {"m": m, "p": w, "s": s}
    for k in u: u[k] = u[k] / u[k].norm()
    for k in ("p", "s"):
        if torch.dot(u[k], u["m"]) < 0: u[k] = -u[k]
    objects[name] = u
    o3[name] = {"probe_test_acc": acc, "sae_latent": li,
                "sae_act_diff": float(diff[li]),
                "cos_mp": float(torch.dot(u["m"], u["p"])),
                "cos_ms": float(torch.dot(u["m"], u["s"])),
                "cos_ps": float(torch.dot(u["p"], u["s"]))}
    log(name, o3[name])

# ---------------- Jacobian via batched finite differences ----------------
clean_norms = resid_last(NEUTRAL).norm(dim=-1)
EPS = 0.02 * clean_norms.mean().item()
ALPHA = 0.5 * clean_norms.mean().item()
log(f"eps={EPS:.3f} alpha={ALPHA:.3f}")

def perturbed_logits(toks, Dirs, scale):
    B = Dirs.shape[0]
    def hook(resid, hook):
        resid[:, -1, :] = resid[:, -1, :] + scale * Dirs
        return resid
    with model.hooks([(HOOK, hook)]):
        lg = model(toks.repeat(B, 1))
    return lg[:, -1, :].float()

G = torch.zeros(D, D)
CH = 48
eye = torch.eye(D)
for pi, p in enumerate(NEUTRAL):
    toks = model.to_tokens(p)
    l0 = model(toks)[0, -1, :].float()
    M = torch.zeros(D, V)
    for st in range(0, D, CH):
        Dirs = eye[st:st + CH]
        M[st:st + CH] = (perturbed_logits(toks, Dirs, EPS) - l0) / EPS
    G += M @ M.T
    if pi == 0:
        Rd = torch.randn(8, D); Rd = Rd / Rd.norm(dim=-1, keepdim=True)
        E1 = (perturbed_logits(toks, Rd, EPS) - l0) / 1.0
        E2 = (perturbed_logits(toks, Rd, 2 * EPS) - l0) / 1.0
        o1_cos = torch.nn.functional.cosine_similarity(E1, E2, dim=-1)
        o1_ratio = E2.norm(dim=-1) / E1.norm(dim=-1)
    log("jacobian probe", pi, "done")
del M

lam, Vecs = torch.linalg.eigh(G)          # ascending
lam = lam.flip(0); Vecs = Vecs.flip(1)     # descending
cum = torch.cumsum(lam, 0) / lam.sum()
k90 = int((cum < 0.90).sum().item()) + 1
Vk = Vecs[:, :k90]
log("k90 =", k90)

# ---------------- oracles ----------------
o1 = {"median_cos": float(o1_cos.median()), "median_ratio": float(o1_ratio.median())}
o1_ok = o1["median_cos"] >= 0.98 and 1.7 <= o1["median_ratio"] <= 2.3

top1, bot1 = Vecs[:, 0], Vecs[:, -1]
tops, bots = [], []
for p in NEUTRAL:
    toks = model.to_tokens(p)
    l0 = model(toks)[0, -1, :].float()
    lt = perturbed_logits(toks, top1[None], ALPHA)[0]
    lb = perturbed_logits(toks, bot1[None], ALPHA)[0]
    tops.append((lt - l0).norm().item()); bots.append((lb - l0).norm().item())
o2 = {"median_top": float(np.median(tops)), "median_bot": float(np.median(bots))}
o2["ratio"] = o2["median_top"] / max(o2["median_bot"], 1e-9)
o2_ok = o2["ratio"] >= 5
log("O1", o1, o1_ok, "| O2", o2, o2_ok)

def ce(v):
    v = v / v.norm()
    return float(torch.sqrt(v @ G @ v))

def nf(v):
    v = v / v.norm()
    pt = Vk.T @ v
    return float(1 - (pt @ pt))

Rc = torch.randn(200, D); Rc = Rc / Rc.norm(dim=-1, keepdim=True)
rand_nf = np.median([nf(r) for r in Rc])
rand_ce = np.median([ce(r) for r in Rc])
nf_vacuous = rand_nf >= 0.9

# ---------------- pair metrics ----------------
pairs, dropped = [], []
for name, u in objects.items():
    if o3[name]["probe_test_acc"] < 0.8:
        dropped.append(name); continue
    for a, b in (("m", "p"), ("m", "s"), ("p", "s")):
        c = float(torch.dot(u[a], u[b]))
        if c >= 0.6: continue
        d = u[a] - u[b]
        pa, pb = Vk.T @ u[a], Vk.T @ u[b]
        cos_topk = float(torch.dot(pa, pb) / (pa.norm() * pb.norm()))
        rel = ce(d) / max(ce(u[a]), ce(u[b]))
        pairs.append({"concept": name, "pair": a + "-" + b, "cos_raw": c,
                      "nf": nf(d), "cos_topk": cos_topk, "rel_ce": rel,
                      "nf_a": nf(u[a]), "nf_b": nf(u[b])})
        log(pairs[-1])

res = {"k90": k90, "eps": EPS, "alpha": ALPHA,
       "oracle": {"o1": o1, "o1_ok": bool(o1_ok), "o2": o2, "o2_ok": bool(o2_ok)},
       "o3": o3, "dropped_concepts": dropped,
       "calibration": {"rand_nf_median": float(rand_nf),
                       "rand_ce_median": float(rand_ce),
                       "nf_clause_vacuous": bool(nf_vacuous)},
       "n_qualifying_pairs": len(pairs), "pairs": pairs}

if not (o1_ok and o2_ok):
    res["verdict"] = "BROKEN_MEASUREMENT"
elif len(pairs) < 3:
    res["verdict"] = "INCONCLUSIVE_NO_PAIRS"
else:
    mnf = float(np.median([q["nf"] for q in pairs]))
    mct = float(np.median([q["cos_topk"] for q in pairs]))
    mrc = float(np.median([q["rel_ce"] for q in pairs]))
    res["medians"] = {"nf": mnf, "cos_topk": mct, "rel_ce": mrc}
    nf_ok = mnf >= 0.90 or nf_vacuous
    if nf_ok and mct >= 0.90 and mrc < 0.5:
        res["verdict"] = "SUPPORTED"
    elif mct < 0.60 or mrc >= 0.5:
        res["verdict"] = "REFUTED"
    else:
        res["verdict"] = "INCONCLUSIVE"

log(json.dumps(res, indent=1)[:2000])
with open(os.path.join(HERE, "result.json"), "w") as f:
    json.dump(res, f, indent=1)
log("VERDICT:", res["verdict"])
