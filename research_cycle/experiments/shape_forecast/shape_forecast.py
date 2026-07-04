#!/usr/bin/env python3
"""Process-Pullback Geometry — preregistered shape forecasting test.
See PREREG_shape_forecast.md (same dir). Phases: controls | run"""
import sys, math, numpy as np, torch, torch.nn as nn
import ripser

torch.set_num_threads(12)
rng = np.random.default_rng(0)
torch.manual_seed(0)

T0, T1 = 0.55, 0.25  # locked after control-only calibration (see prereg amendment)

# ---------------- domains ----------------
class Domain:
    def __init__(self, name, n_states, vocab, init, emis, step, pred):
        self.name, self.n_states, self.vocab, self.init = name, n_states, vocab, init
        self.emis, self.step, self.pred = emis, step, pred  # pred = (b0,b1)
        self.E = np.stack([emis(s) for s in range(n_states)])  # ground-truth config

def clock(mod):
    inc = [1, 2, 3]
    def emis(r):
        v = np.array([math.exp(0.7*math.cos(2*math.pi*r/mod - 2*math.pi*k/3)) for k in range(3)])
        return v / v.sum()
    return Domain(f"clock{mod}", mod, 3, 0, emis, lambda s, t: (s + inc[t]) % mod, (1, 1))

def flag():
    def emis(s): return np.array([.45, .45, .10]) if s == 0 else np.array([.8, .2, 0.])
    return Domain("flag", 2, 3, 0, emis, lambda s, t: 1 if (s == 1 or t == 2) else 0, (2, 0))

def counter():
    def emis(c): p = 0.15 + 0.7*c/8; return np.array([1-p, p])
    return Domain("counter", 9, 2, 4, emis,
                  lambda c, t: min(8, max(0, c + (1 if t == 1 else -1))), (1, 0))

def twoflag():
    def emis(s):
        f1, f2 = s & 1, s >> 1
        pt1, pt2 = (0. if f1 else .06), (0. if f2 else .06)
        m = 0.5*(1 - pt1 - pt2)
        a, b = (.8 if f1 else .2), (.8 if f2 else .2)
        return np.array([m*a, m*(1-a), m*b, m*(1-b), pt1, pt2])
    def step(s, t):
        if t == 4: s |= 1
        if t == 5: s |= 2
        return s
    return Domain("twoflag", 4, 6, 0, emis, step, (4, 0))

DOMAINS = [clock(12), clock(8), flag(), counter(), twoflag()]

# ---------------- geometry classifier ----------------
def classify(P, t0=None, t1=None):
    t0, t1 = t0 or T0, t1 or T1
    P = np.asarray(P, np.float64)
    P = P - P.mean(0)                       # v2: whiten significant PCA axes
    U, S, _ = np.linalg.svd(P, full_matrices=False)
    keep = (S**2) > 0.05 * (S[0]**2)
    P = U[:, keep] * math.sqrt(len(P))
    D = np.linalg.norm(P[:, None] - P[None, :], axis=-1)
    diam = D.max()
    if diam == 0: return (1, 0), 0.0
    # b0: single linkage components at cut t0*diam (union-find)
    par = list(range(len(P)))
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for i in range(len(P)):
        for j in range(i+1, len(P)):
            if D[i, j] < t0*diam: par[find(i)] = find(j)
    b0 = len({find(i) for i in range(len(P))})
    # b1: Rips persistence
    b1, ratio = 0, 0.0
    if len(P) >= 6:                         # v2: a 4-cycle is not loop evidence
        h1 = ripser.ripser(D, distance_matrix=True, maxdim=1)['dgms'][1]
        if len(h1):
            pers = h1[:, 1] - h1[:, 0]
            ratio = float(pers.max() / diam)
            b1 = int((pers > t1*diam).sum())
    return (b0, b1), ratio

# ---------------- sampling ----------------
def sample(dom, B, L=63):
    toks = np.zeros((B, L), np.int64); states = np.zeros((B, L), np.int64)
    logp = np.zeros(B)
    s = np.full(B, dom.init)
    for t in range(L):
        E = dom.E[s]                                   # [B, vocab]
        u = rng.random(B)
        x = (E.cumsum(1) < u[:, None]).sum(1).clip(0, dom.vocab-1)
        logp += np.log(E[np.arange(B), x] + 1e-12)
        s = np.array([dom.step(int(si), int(xi)) for si, xi in zip(s, x)])
        toks[:, t], states[:, t] = x, s
    return toks, states, -logp.mean()/L  # ground-truth CE per token

# ---------------- model ----------------
class GPT(nn.Module):
    def __init__(self, vocab, d=64, nl=2, nh=4, L=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, d); self.pos = nn.Embedding(L, d)
        blk = nn.TransformerEncoderLayer(d, nh, 4*d, dropout=0.0,
                                         batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(blk, nl)
        self.lnf = nn.LayerNorm(d); self.head = nn.Linear(d, vocab)
        self.register_buffer("mask", torch.triu(torch.full((L, L), float("-inf")), 1))
    def forward(self, x, want_resid=False):
        h = self.emb(x) + self.pos(torch.arange(x.shape[1]))
        h = self.blocks(h, mask=self.mask[:x.shape[1], :x.shape[1]])
        return (self.head(self.lnf(h)), h) if want_resid else self.head(self.lnf(h))

def train_domain(dom, steps):
    V = dom.vocab + 1; BOS = dom.vocab
    m = GPT(V)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.01)
    for i in range(steps):
        toks, _, _ = sample(dom, 64)
        x = torch.from_numpy(np.concatenate([np.full((64, 1), BOS), toks], 1))
        logits = m(x[:, :-1])
        loss = nn.functional.cross_entropy(logits.reshape(-1, V), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return m

@torch.no_grad()
def evaluate(dom, m, n=3000):
    V = dom.vocab + 1; BOS = dom.vocab
    toks, states, ce_true = sample(dom, n)
    x = torch.from_numpy(np.concatenate([np.full((n, 1), BOS), toks], 1))
    ce, cents, cnt = 0.0, np.zeros((dom.n_states, 64)), np.zeros(dom.n_states)
    for b in range(0, n, 250):
        xb = x[b:b+250]
        logits, resid = m(xb[:, :-1], want_resid=True)
        ce += nn.functional.cross_entropy(
            logits.reshape(-1, V), xb[:, 1:].reshape(-1), reduction="sum").item()
        R = resid[:, 20:63].reshape(-1, 64).numpy()      # position i holds token x_i
        S = states[b:b+250, 19:62].reshape(-1)           # s_i after token x_i
        for s in range(dom.n_states):
            sel = S == s
            cents[s] += R[sel].sum(0); cnt[s] += sel.sum()
    ce /= n * 63
    return ce, ce_true, cents / np.maximum(cnt, 1)[:, None]

# ---------------- phases ----------------
def controls():
    print("== ORACLE (ground-truth emission configs) ==")
    ok = True
    for d in DOMAINS:
        sig, r = classify(d.E)
        good = sig == d.pred
        ok &= good
        print(f"  {d.name:8s} pred={d.pred} got={sig} maxH1pers/diam={r:.3f} {'OK' if good else 'FAIL'}")
    print("== STRESS controls (noisy synthetic shapes in R^64) ==")
    B = rng.normal(size=(2, 64)); B /= np.linalg.norm(B, axis=1, keepdims=True)
    th12 = np.linspace(0, 2*np.pi, 12, endpoint=False)
    th8 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    ell = lambda th: np.outer(4*np.cos(th), B[0]) + np.outer(np.sin(th), B[1])
    line9 = np.outer(np.linspace(0, 1, 9), B[0])
    sq = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) @ B
    blobs = np.concatenate([np.outer(np.zeros(6), B[0]), np.outer(np.ones(6), B[0])]) \
        + rng.normal(size=(12, 64)) * 0.03
    for nm, pts, want in [("ellipse12", ell(th12), (1, 1)), ("ellipse8", ell(th8), (1, 1)),
                          ("line9", line9, (1, 0)), ("square4", sq, (4, 0)),
                          ("2blobs", blobs, (2, 0))]:
        noisy = pts + rng.normal(size=pts.shape) * 0.02 * np.linalg.norm(pts.std(0))
        sig, r = classify(noisy)
        good = sig == want; ok &= good
        print(f"  {nm:10s} want={want} got={sig} H1r={r:.3f} {'OK' if good else 'FAIL'}")
    print("== NEGATIVE control (Gaussian point sets in R^64) ==")
    for n in (8, 12):
        fp = sum(classify(rng.normal(size=(n, 64)))[0][1] >= 1 for _ in range(200))
        ok &= fp/200 < 0.05
        print(f"  n={n}: false-circle rate {fp/200:.3f} (need <0.05)")
    print("ORACLE PASS" if ok else "ORACLE BROKEN")

def run():
    results = []
    for d in DOMAINS:
        m = train_domain(d, 3000)
        ce, ce_true, C = evaluate(d, m)
        if ce - ce_true >= 0.05:
            m = train_domain(d, 8000)   # preregistered single extension
            ce, ce_true, C = evaluate(d, m)
        learned = ce - ce_true < 0.05
        sig, r = classify(C)
        match = learned and sig == d.pred
        results.append((d.name, learned, ce, ce_true, sig, d.pred, r, match))
        print(f"{d.name:8s} CE={ce:.4f} truth={ce_true:.4f} dCE={ce-ce_true:+.4f} "
              f"learned={learned} pred={d.pred} got={sig} H1r={r:.3f} match={match}")
    k = sum(r[1] for r in results); mm = sum(r[7] for r in results)
    from scipy.stats import binom
    p = float(binom.sf(mm - 1, k, 0.25)) if k else 1.0
    falsifier = any(r[0].startswith("clock") and r[1] and r[4][1] == 0 for r in results)
    print(f"\nlearned k={k}, matches m={mm}, binomial p (null 1/4) = {p:.5f}")
    print(f"falsifier (circle process, matched loss, no loop): {falsifier}")
    verdict = ("REFUTED" if falsifier or mm <= 1 else
               "SUPPORTED" if (k >= 4 and mm == k and p < 0.01) else "INCONCLUSIVE")
    print("VERDICT:", verdict)

if __name__ == "__main__":
    {"controls": controls, "run": run}[sys.argv[1]]()
