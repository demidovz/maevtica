import glob, os, sys, time
import numpy as np
import torch
from safetensors.torch import load_file
from scipy.stats import pearsonr, spearmanr

torch.set_grad_enabled(False)
torch.set_num_threads(max(1, os.cpu_count() or 4))
rng = np.random.default_rng(0)
t0 = time.time()

SMOKE = "--smoke" in sys.argv
N_ATOMS = 8 if SMOKE else 200
N_SEQ_STAT = 16 if SMOKE else 64
N_SEQ_CAUSAL = 4 if SMOKE else 8
SEQ_LEN = 128
K_LABEL = 10

L = 8
SNAP = glob.glob("/home/friemann/.cache/huggingface/hub/models--jbloom--GPT2-Small-SAEs-Reformatted/snapshots/*/")[0]
w = load_file(f"{SNAP}blocks.{L}.hook_resid_pre/sae_weights.safetensors")
W_enc = w["W_enc"].float()          # [768, 24576]
b_enc = w["b_enc"].float()          # [24576]
b_dec = w["b_dec"].float()          # [768]
W_dec = w["W_dec"].float()          # [24576, 768]
d_sae, d_in = W_dec.shape
dec_unit = (W_dec / (W_dec.norm(dim=1, keepdim=True) + 1e-9)).numpy()

from transformer_lens import HookedTransformer
m = HookedTransformer.from_pretrained("gpt2", device="cpu"); m.eval()
HOOK = f"blocks.{L}.hook_resid_pre"

# ---------- corpus: wikitext-103 test parquet, local ----------
import pyarrow.parquet as pq
PARQ = glob.glob("/home/friemann/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/*/wikitext-103-raw-v1/test-00000-of-00001.parquet")[0]
texts = [t for t in pq.read_table(PARQ).column("text").to_pylist() if len(t) >= 400]
stream = []
need = (N_SEQ_STAT + N_SEQ_CAUSAL) * SEQ_LEN
for t in texts:
    stream.extend(m.tokenizer(t)["input_ids"])
    if len(stream) >= need + 1:
        break
assert len(stream) >= need + 1, "not enough corpus tokens"
toks_all = torch.tensor(stream[:need], dtype=torch.long).view(-1, SEQ_LEN)
toks_stat = toks_all[:N_SEQ_STAT]                  # λ / κ stats
toks_causal = toks_all[N_SEQ_STAT:]                # C held-out
print(f"corpus: {len(texts)} texts, stat {tuple(toks_stat.shape)} causal {tuple(toks_causal.shape)}")

# ---------- resid cache + SAE firing counts (chunked) ----------
resid = []
for i in range(0, N_SEQ_STAT, 8):
    _, cache = m.run_with_cache(toks_stat[i:i+8], names_filter=HOOK)
    resid.append(cache[HOOK].float())
resid = torch.cat(resid, 0)                        # [S,128,768]
train_mask_seq = (torch.arange(N_SEQ_STAT) % 2 == 0)
X = resid.view(-1, d_in)                           # [P,768]
tok_flat = toks_stat.reshape(-1).numpy()
pos_train = train_mask_seq.repeat_interleave(SEQ_LEN).numpy()
P = X.shape[0]

fire_train = torch.zeros(d_sae); fire_test = torch.zeros(d_sae); fire_all = torch.zeros(d_sae)
CH = 1024
for i in range(0, P, CH):
    a = torch.relu((X[i:i+CH] - b_dec) @ W_enc + b_enc)   # [ch, d_sae]
    f = (a > 0).float()
    mtr = torch.tensor(pos_train[i:i+CH], dtype=torch.float32).unsqueeze(1)
    fire_train += (f * mtr).sum(0); fire_test += (f * (1 - mtr)).sum(0); fire_all += f.sum(0)
alive = ((fire_train >= 20) & (fire_test >= 20) & (fire_all / P < 0.5)).numpy()
alive_idx = np.where(alive)[0]
print(f"alive atoms: {len(alive_idx)} / {d_sae}")
sample = rng.choice(alive_idx, size=min(N_ATOMS, len(alive_idx)), replace=False)
NS = len(sample)

# acts for sampled atoms only
We_s = W_enc[:, sample]; be_s = b_enc[sample]
A = torch.relu((X - b_dec) @ We_s + be_s).numpy()  # [P, NS]
f_rate = (A > 0).mean(0)                           # κ axis (firing rate)

# ---------- λ: unigram-label simulator ----------
tr = pos_train.astype(bool); te = ~tr
tok_tr = tok_flat[tr]; tok_te = tok_flat[te]
V = 50257
cnt_tr = np.bincount(tok_tr, minlength=V)
eligible = cnt_tr >= 2

def legibility(act_vec, K=K_LABEL):
    a_tr = act_vec[tr]; a_te = act_vec[te]
    mean_per_tok = np.bincount(tok_tr, weights=a_tr, minlength=V) / np.maximum(cnt_tr, 1)
    mean_per_tok[~eligible] = -np.inf
    order = np.argsort(mean_per_tok)[::-1][:K]
    label = order[np.isfinite(mean_per_tok[order]) & (mean_per_tok[order] > 0)]
    if len(label) == 0:
        return 0.0
    lut = np.zeros(V); lut[label] = mean_per_tok[label]
    sim = lut[tok_te]
    if sim.std() == 0 or a_te.std() == 0:
        return 0.0
    return float(pearsonr(sim, a_te)[0])

lam = np.array([legibility(A[:, i]) for i in range(NS)])
lam50 = np.array([legibility(A[:, i], K=50) for i in range(NS)])

# ---------- ORACLES O1/O2 ----------
top_tok = int(np.argmax(np.bincount(tok_flat, minlength=V)))
synth_leg = (tok_flat == top_tok).astype(np.float64)
o1_pos = legibility(synth_leg)
synth_noise = np.abs(rng.standard_normal(P))
o1_neg = legibility(synth_noise)
o2_std = float(lam.std())
print(f"O1 synthetic-legible λ = {o1_pos:.3f} (need >0.9) | noise λ = {o1_neg:.3f} (need <0.15)")
print(f"O2 std(λ) = {o2_std:.3f} (need >0.03) | λ range [{lam.min():.3f}, {lam.max():.3f}]")

# ---------- R: read-fraction (reader Gram, as read_fraction/exp.py) ----------
G = np.zeros((d_in, d_in), dtype=np.float64)
for layer in range(L, m.cfg.n_layers):
    b = m.blocks[layer]; mats = []
    for W in [b.attn.W_Q, b.attn.W_K, b.attn.W_V]:
        Wn = W.detach().numpy(); mats.append(Wn.transpose(1, 0, 2).reshape(d_in, -1))
    mats.append(b.mlp.W_in.detach().numpy())
    R_ = np.concatenate(mats, axis=1); G += R_ @ R_.T
evals, Vg = np.linalg.eigh(G)
lam_max = evals[-1]
topv = Vg[:, -1]; botv = Vg[:, 0]
rho_top = np.sqrt(max(topv @ G @ topv, 0)); rho_bot = np.sqrt(max(botv @ G @ botv, 0))
rr = rng.standard_normal((2000, d_in)); rr /= np.linalg.norm(rr, axis=1, keepdims=True)
rho_rand = np.sqrt(np.clip(np.einsum('ij,ij->i', rr @ G, rr), 0, None))
OB_ratio = rho_top / (rho_bot + 1e-12)
print(f"O3 rho_top {rho_top:.3f} rho_bot {rho_bot:.4g} ratio {OB_ratio:.2f} rho_rand {rho_rand.mean():.3f} (expect {np.sqrt(np.trace(G)/d_in):.3f})")
Rf = np.einsum('ij,jk,ik->i', dec_unit[sample], G, dec_unit[sample]) / lam_max

# ---------- C: causal effect via per-atom ablation (secondary) ----------
loss_clean = m(toks_causal, return_type="loss", loss_per_token=True).float().numpy()
C = np.zeros(NS)
for i, j in enumerate(sample):
    wj_enc = W_enc[:, j]; bj = b_enc[j]; wj_dec = W_dec[j]
    def fn(r, hook):
        a = torch.relu((r - b_dec) @ wj_enc + bj)
        return r - a.unsqueeze(-1) * wj_dec
    la = m.run_with_hooks(toks_causal, return_type="loss", loss_per_token=True,
                          fwd_hooks=[(HOOK, fn)]).float().numpy()
    C[i] = float((la - loss_clean).mean())
    if i % 25 == 0:
        print(f"  causal {i}/{NS} elapsed {time.time()-t0:.0f}s", flush=True)
o4_mean = float(C.mean()); o4_frac = float((C >= 0).mean())
print(f"O4 mean C = {o4_mean:.5f} (need >0), frac C>=0 = {o4_frac:.2f} (need >=0.70)")

# ---------- PRIMARY ----------
logf = np.log10(f_rate)
r_R, p_R = pearsonr(lam, Rf)
r_k, p_k = pearsonr(lam, logf)
print(f"\nn={NS} atoms | λ mean {lam.mean():.3f} | f_rate range [{f_rate.min():.4f},{f_rate.max():.4f}] | R range [{Rf.min():.3f},{Rf.max():.3f}]")
print(f"PRIMARY Pearson(λ, R)       = {r_R:+.3f} (p={p_R:.3g})")
print(f"PRIMARY Pearson(λ, log10 f) = {r_k:+.3f} (p={p_k:.3g})")
print(f"  robustness: Spearman(λ,R) {spearmanr(lam,Rf).correlation:+.3f} | Spearman(λ,logf) {spearmanr(lam,logf).correlation:+.3f} | Pearson(λ,raw f) {pearsonr(lam,f_rate)[0]:+.3f} | K=50: r_R {pearsonr(lam50,Rf)[0]:+.3f}, r_κ {pearsonr(lam50,logf)[0]:+.3f}")

# corollaries (secondary)
o4_ok = (o4_mean > 0) and (o4_frac >= 0.70)
med_lam = np.median(lam); med_C = np.median(C)
topC = np.argsort(C)[::-1][:max(1, NS // 10)]
topL = np.argsort(lam)[::-1][:max(1, NS // 10)]
dark = float((lam[topC] < med_lam).mean())
inert = float((C[topL] < med_C).mean())
print(f"corollary (i) dark-computation share (top-decile C, λ<median) = {dark:.2f}" + ("" if o4_ok else " [O4 FAILED — not gating, reported raw]"))
print(f"corollary (ii) inert-legible share (top-decile λ, C<median)  = {inert:.2f}")

# ---------- VERDICT ----------
broken = not (o1_pos > 0.9 and o1_neg < 0.15 and o2_std > 0.03 and OB_ratio > 5)
if broken:
    V_ = "BROKEN_MEASUREMENT"
elif abs(r_R) < 0.20 and abs(r_k) < 0.20:
    V_ = "SUPPORTED"
elif abs(r_R) > 0.50 or abs(r_k) > 0.50:
    V_ = "REFUTED"
else:
    V_ = "INCONCLUSIVE"
print(f"\nVERDICT {V_}  ({'SMOKE — not binding' if SMOKE else 'full preregistered run'})  total {time.time()-t0:.0f}s")
