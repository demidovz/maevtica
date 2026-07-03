import json, glob, numpy as np, torch
from safetensors.torch import load_file
from scipy.stats import spearmanr, pearsonr
torch.set_grad_enabled(False)

SNAP=glob.glob("/home/friemann/.cache/huggingface/hub/models--jbloom--GPT2-Small-SAEs-Reformatted/snapshots/*/")[0]
L=8; base=f"{SNAP}blocks.{L}.hook_resid_pre"
w=load_file(f"{base}/sae_weights.safetensors")
W_dec=w["W_dec"].float().numpy()               # [d_sae, d_in]
sp=load_file(f"{base}/sparsity.safetensors")
spk=list(sp.keys())[0]; logfreq=sp[spk].float().numpy()   # log10 feature sparsity
freq=10.0**logfreq
d_sae,d_in=W_dec.shape
print("d_sae",d_sae,"d_in",d_in,"sparsity key",spk,"logfreq range",logfreq.min(),logfreq.max())

dec_norm=np.linalg.norm(W_dec,axis=1)
salience=freq*dec_norm
d_unit=W_dec/(dec_norm[:,None]+1e-9)

# GPT-2 small, LN folded (default). Build reader Gram G over layers >= L.
from transformer_lens import HookedTransformer
m=HookedTransformer.from_pretrained("gpt2", device="cpu"); m.eval()
G=np.zeros((d_in,d_in),dtype=np.float64)
ncols=0
for layer in range(L, m.cfg.n_layers):
    b=m.blocks[layer]
    mats=[]
    # attention readers: W_Q,W_K,W_V shape [n_heads,d_model,d_head] -> reshape [d_model, n_heads*d_head]
    for W in [b.attn.W_Q,b.attn.W_K,b.attn.W_V]:
        Wn=W.detach().numpy()            # [h,d_model,d_head]
        mats.append(Wn.transpose(1,0,2).reshape(d_in,-1))
    # mlp reader: W_in [d_model,d_mlp]
    mats.append(b.mlp.W_in.detach().numpy())
    R=np.concatenate(mats,axis=1)        # [d_in, cols]
    G+=R@R.T; ncols+=R.shape[1]
print("G built, reader cols total",ncols,"trace",np.trace(G))

# rho_i = sqrt(d_i^T G d_i)
Gd=d_unit@G                              # [d_sae,d_in]
quad=np.einsum('ij,ij->i',Gd,d_unit)     # d_i^T G d_i
rho=np.sqrt(np.clip(quad,0,None))

# ---- ORACLE O1: reader metric respects G spectrum ----
evals=np.linalg.eigvalsh(G)              # ascending
lam_max=evals[-1]; lam_min=evals[0]
w_,V=np.linalg.eigh(G)
top=V[:,-1]; bot=V[:,0]
rho_top=np.sqrt(max(top@G@top,0)); rho_bot=np.sqrt(max(bot@G@bot,0))
rng=np.random.default_rng(0)
rr=rng.standard_normal((2000,d_in)); rr/=np.linalg.norm(rr,axis=1,keepdims=True)
rho_rand=np.sqrt(np.clip(np.einsum('ij,ij->i',rr@G,rr),0,None))
o1_ratio=rho_top/(rho_bot+1e-12)
print(f"O1 lam_max {lam_max:.3f} lam_min {lam_min:.6g} rho_top {rho_top:.4f} rho_bot {rho_bot:.6g} ratio {o1_ratio:.2f}")
print(f"O1 rho_rand mean {rho_rand.mean():.4f} (expect ~sqrt(tr/d)= {np.sqrt(np.trace(G)/d_in):.4f})")

# ---- ORACLE O2: ranking pipeline ----
noise=salience+rng.standard_normal(d_sae)*1e-6*np.std(salience)
o2_self=spearmanr(salience,noise).correlation
o2_perm=spearmanr(salience,rng.permutation(salience)).correlation
print(f"O2 self {o2_self:.4f} perm {o2_perm:.4f}")

# ---- PRIMARY ----
sp_corr=spearmanr(salience,rho).correlation
pe_corr=pearsonr(salience,rho)[0]
# also freq-only vs rho (decoder norm nearly const?)
print("dec_norm mean/std",dec_norm.mean(),dec_norm.std())
sp_freq=spearmanr(freq,rho).correlation
print(f"PRIMARY spearman(salience,rho) {sp_corr:.4f}  pearson {pe_corr:.4f}  spearman(freq,rho) {sp_freq:.4f}")

# verdict
broken = o1_ratio < 5 or o2_self < 0.99 or abs(o2_perm) > 0.05
if broken:
    v="BROKEN_MEASUREMENT"
elif abs(pe_corr)>0.5 or abs(sp_corr)>0.5:
    v="REFUTED"
elif abs(sp_corr)<=0.2 and abs(pe_corr)<=0.5:
    v="SUPPORTED"
else:
    v="INCONCLUSIVE"
print("VERDICT",v)
