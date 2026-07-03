import os, numpy as np, torch
from scipy.stats import pearsonr
torch.set_num_threads(max(1, os.cpu_count() or 4))
rng = np.random.default_rng(0)

from transformer_lens import HookedTransformer
m = HookedTransformer.from_pretrained("gpt2", device="cpu"); m.eval()
d_in = m.cfg.d_model
LAYERS = [5, 8]

CATS = {
 "colors": "red blue green black white yellow brown pink gray gold",
 "animals": "dog cat horse cow pig sheep bird fish lion bear wolf fox",
 "fruits": "apple banana orange lemon grape peach pear",
 "countries": "France China Japan Spain India Italy Egypt",
 "drinks": "water wine beer juice milk coffee tea",
 "body": "head hand foot arm leg eye nose ear",
}
BASE = ["The", "It was", "I think that", "We saw a", "There is a", "Yesterday I found a"]

def stid(w):
    t = m.to_tokens(" " + w, prepend_bos=False)[0]
    return int(t[0]) if t.shape[0] == 1 else None

cat_ids = {}
for c, s in CATS.items():
    ids = [stid(w) for w in s.split()]
    cat_ids[c] = [i for i in ids if i is not None]
all_ids = {c: np.array(v) for c, v in cat_ids.items()}
union_ids = np.array(sorted({i for v in cat_ids.values() for i in v}))

def src_prompts(c):
    ws = CATS[c].split()[:5]
    return [f"{ws[i].capitalize()}, {ws[(i+1)%5]} and" for i in range(4)]

def heldout_prompts(c):
    ws = CATS[c].split()
    ps = [f"I really like {ws[i%len(ws)]} and {ws[(i+2)%len(ws)]}" for i in range(4)]
    ps += [f"She talked about the {ws[(i+1)%len(ws)]} yesterday" for i in range(4)]
    return ps

def metric_from_logits(lg, c):
    pos = lg[all_ids[c]].mean()
    return float(pos - lg[union_ids].mean())

def resid_last(prompt, L):
    hook = f"blocks.{L}.hook_resid_pre"
    with torch.no_grad():
        _, cache = m.run_with_cache(m.to_tokens(prompt), names_filter=hook)
    return cache[hook][0, -1].float()

def run_patched(prompt, L, newvec, c):
    hook = f"blocks.{L}.hook_resid_pre"
    def fn(r, hook):
        r[:, -1, :] = newvec.to(r.dtype); return r
    with torch.no_grad():
        lg = m.run_with_hooks(m.to_tokens(prompt), fwd_hooks=[(hook, fn)])[0, -1].float().numpy()
    return metric_from_logits(lg, c)

def grad_read_dir(prompt, L, c):
    hook = f"blocks.{L}.hook_resid_pre"
    store = {}
    def fn(r, hook):
        r.requires_grad_(True); r.retain_grad(); store["r"] = r; return r
    lg = m.run_with_hooks(m.to_tokens(prompt), fwd_hooks=[(hook, fn)])[0, -1]
    pos = lg[torch.tensor(all_ids[c])].mean()
    B = pos - lg[torch.tensor(union_ids)].mean()
    B.backward()
    g = store["r"].grad[0, -1].detach().float().numpy()
    m.zero_grad(set_to_none=True)
    return g

def unit(v):
    v = np.asarray(v, dtype=np.float64); return v / (np.linalg.norm(v) + 1e-12)

def auc_of(d, pos_acts, neg_acts):
    sp = pos_acts @ d; sn = neg_acts @ d
    lab = np.r_[np.ones(len(sp)), np.zeros(len(sn))]
    sc = np.r_[sp, sn]
    order = np.argsort(sc)
    ranks = np.empty(len(sc)); ranks[order] = np.arange(1, len(sc) + 1)
    n1, n0 = len(sp), len(sn)
    a = (ranks[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(max(a, 1 - a))

rows = []   # per (group, dir): m, cos2, auc, bilin, tag
groups = [] # per group summary
for L in LAYERS:
    ho_acts = {c: torch.stack([resid_last(p, L) for p in heldout_prompts(c)]).numpy() for c in CATS}
    base_clean = {}
    for c in CATS:
        # source mean / base acts / clean metric
        xs = torch.stack([resid_last(p, L) for p in src_prompts(c)]).mean(0)
        xb = [resid_last(p, L) for p in BASE]
        with torch.no_grad():
            clean = [metric_from_logits(m(m.to_tokens(p))[0, -1].float().numpy(), c) for p in BASE]
        E_full = float(np.mean([run_patched(p, L, xs, c) - clean[i] for i, p in enumerate(BASE)]))
        r = unit(np.mean([grad_read_dir(p, L, c) for p in BASE], axis=0))
        Delta_hats = [unit((xs - x).numpy()) for x in xb]
        Dh_mean = unit(np.mean([(xs - x).numpy() for x in xb], axis=0))
        p_dir = None
        # probe: diff-in-means of source acts vs grand mean over all cats' source acts
        base_clean[c] = (xs, xb, clean, E_full, r, Dh_mean)
    grand = torch.stack([base_clean[c][0] for c in CATS]).mean(0)
    for c in CATS:
        xs, xb, clean, E_full, r, Dh = base_clean[c]
        p_dir = unit((xs - grand).numpy())
        d_nm = unit(p_dir - (p_dir @ r) * r)
        def mix(a, u, v): return unit(a * np.asarray(u) + (1 - a) * np.asarray(v))
        rnds = [unit(rng.standard_normal(d_in)) for _ in range(3)]
        pool = [("probe", p_dir), ("read", r), ("delta", Dh), ("d_nm", d_nm),
                ("mix25", mix(.25, p_dir, r)), ("mix50", mix(.5, p_dir, r)),
                ("mix75", mix(.75, p_dir, r)),
                ("pr50", mix(.5, p_dir, rnds[0])), ("pr75", mix(.75, p_dir, rnds[1])),
                ("rnd1", rnds[0]), ("rnd2", rnds[1]), ("rnd3", rnds[2])]
        neg = np.concatenate([ho_acts[o] for o in CATS if o != c])
        posx = ho_acts[c]
        def E_of(d):
            dt = torch.tensor(np.asarray(d), dtype=torch.float32)
            es = []
            for i, pr in enumerate(BASE):
                x = xb[i]
                coef = float(dt @ (xs - x))
                es.append(run_patched(pr, L, x + coef * dt, c) - clean[i])
            return float(np.mean(es))
        # O2 code-correctness oracle: per-prompt delta-hat direction patch must
        # reconstruct the source exactly => effect == E_full
        o2_es = []
        for i, pr in enumerate(BASE):
            dhi = unit((xs - xb[i]).numpy())
            dt = torch.tensor(dhi, dtype=torch.float32)
            coef = float(dt @ (xs - xb[i]))
            o2_es.append(run_patched(pr, L, xb[i] + coef * dt, c) - clean[i])
        m_o2 = float(np.mean(o2_es)) / E_full if abs(E_full) > 1e-9 else np.nan
        grp = {"L": L, "c": c, "E_full": E_full, "m_o2": m_o2}
        for tag, d in pool:
            e = E_of(d)
            md = e / E_full if abs(E_full) > 1e-9 else np.nan
            cos2 = float((d @ r) ** 2)
            au = auc_of(np.asarray(d), posx, neg)
            bil = float((d @ Dh) * (d @ r))
            rows.append({"L": L, "c": c, "tag": tag, "m": md, "cos2": cos2,
                         "auc": au, "bil": bil})
            grp[tag] = md
            if tag == "read": grp["recovery"] = md
            if tag == "delta": grp["m_delta"] = md
            if tag == "probe": grp["auc_p"] = au
            if tag == "d_nm": grp["auc_nm"] = au; grp["m_nm"] = md
            if tag == "rnd1": grp["auc_rnd"] = au
        groups.append(grp)
        print(f"L{L} {c:10s} E_full {E_full:7.3f} m_o2 {grp['m_o2']:.3f} m(delta) {grp['m_delta']:.3f} "
              f"recovery {grp['recovery']:.3f} m(probe) {grp['probe']:.3f} "
              f"m(d_nm) {grp['m_nm']:.3f} auc_p {grp['auc_p']:.3f} auc_nm {grp['auc_nm']:.3f}")

M = np.array([w["m"] for w in rows]); C2 = np.array([w["cos2"] for w in rows])
AU = np.array([w["auc"] for w in rows]); BI = np.array([w["bil"] for w in rows])

def r2(X, y):
    X = np.column_stack([np.ones(len(y))] + list(X))
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    return 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))

R2_read = r2([C2], M); R2_auc = r2([AU], M); R2_both = r2([C2, AU], M)
dR2 = R2_both - R2_read
R2_bil = r2([BI], M)
rec = np.array([g["recovery"] for g in groups]); RECOVERY = float(np.median(rec))

# oracles
Efulls = np.array([g["E_full"] for g in groups])
O1 = float(Efulls.mean())
O2 = float(np.median(np.abs(np.array([g["m_o2"] for g in groups]) - 1)))
O3p = float(np.mean([g["auc_p"] for g in groups])); O3r = float(np.mean([g["auc_rnd"] for g in groups]))

qual = [(g["auc_nm"] >= 0.8 and abs(g["m_nm"]) < 0.3) for g in groups]

print(f"\nn={len(rows)} points, {len(groups)} groups")
print(f"O1 mean E_full {O1:.3f} | O2 median|m(delta)-1| {O2:.4f} | O3 auc_probe {O3p:.3f} auc_rand {O3r:.3f}")
print(f"R2_read {R2_read:.3f} | R2_auc {R2_auc:.3f} | R2_both {R2_both:.3f} | dR2_auc {dR2:.3f}")
print(f"[secondary] R2_bilinear (d.Delta)(d.r) = {R2_bil:.3f}")
print(f"RECOVERY median E(r)/E_full = {RECOVERY:.3f} (per-group: {np.round(rec,2).tolist()})")
print(f"d_nm qualifies decodable-non-mediating: {sum(qual)}/{len(qual)}")
pr_cm = pearsonr(C2, M); pr_am = pearsonr(AU, M)
print(f"pearson(cos2,m) {pr_cm[0]:.3f} (p={pr_cm[1]:.2g}) | pearson(auc,m) {pr_am[0]:.3f} (p={pr_am[1]:.2g})")

broken = (O1 < 0.5) or (O2 > 0.10) or (O3p < 0.85) or not (0.3 <= O3r <= 0.7)
if broken:
    V = "BROKEN_MEASUREMENT"
elif (R2_read <= R2_auc) or (R2_read < 0.25) or (RECOVERY < 0.35):
    V = "REFUTED"
elif (R2_read >= 0.50) and (dR2 < 0.05) and (RECOVERY >= 0.70):
    V = "SUPPORTED"
else:
    V = "INCONCLUSIVE"
print("\nVERDICT", V)
