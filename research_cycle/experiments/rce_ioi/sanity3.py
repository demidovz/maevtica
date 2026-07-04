import os, json
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(8)
from transformer_lens import HookedTransformer
m = HookedTransformer.from_pretrained("gpt2", device="cpu"); m.eval()
m.set_use_attn_result(True)
nL, nH = m.cfg.n_layers, m.cfg.n_heads
pairs = [("John","Mary"),("Tom","Sarah"),("James","Anna"),("Paul","Kate"),
         ("Mark","Alice"),("Dan","Emma"),("Peter","Rose"),("Jack","Mary")]
prompts, io_ids, s_ids = [], [], []
for A, B in pairs:
    for a, b in ((A, B), (B, A)):
        prompts.append(f"When {a} and {b} went to the store, {b} gave a drink to")
        io_ids.append(int(m.to_tokens(" "+a, prepend_bos=False)[0,0]))
        s_ids.append(int(m.to_tokens(" "+b, prepend_bos=False)[0,0]))
toks = m.to_tokens(prompts)
io_ids = torch.tensor(io_ids); s_ids = torch.tensor(s_ids)
ar = torch.arange(toks.shape[0])
ld = lambda lg: float((lg[:, -1].float()[ar, io_ids] - lg[:, -1].float()[ar, s_ids]).mean())
udiff = (m.W_U[:, io_ids] - m.W_U[:, s_ids]).T.float()
rfilt = lambda n: n.endswith("hook_result") or n == "ln_final.hook_scale"
zfilt = lambda n: n.endswith("hook_z")
lg0, zc = m.run_with_cache(toks, names_filter=zfilt)
clean_ld = ld(lg0)
z_mean = {L: zc[f"blocks.{L}.attn.hook_z"].mean(0, keepdim=True) for L in range(nL)}

def dla99(c):
    scale = c["ln_final.hook_scale"][:, -1].float()
    r = c["blocks.9.attn.hook_result"][:, -1].float()
    r = (r - r.mean(-1, keepdim=True)) / scale.unsqueeze(1)
    return float((r[:, 9] * udiff).sum(-1).mean())

def zero99(z, hook):
    z[:, :, 9] = 0.0; return z
with m.hooks(fwd_hooks=[("blocks.9.attn.hook_z", zero99)]):
    lgA, cA = m.run_with_cache(toks, names_filter=rfilt)
a = dla99(cA)
print("A: DLA(9.9) under its own zero-ablation =", round(a, 4), "| LD =", round(ld(lgA),3))

NM = [(9,9),(9,6),(10,0)]
byL = {}
for L,H in NM: byL.setdefault(L, []).append(H)
hooks = []
for L, Hs in byL.items():
    def mk(L, Hs):
        def fn(z, hook):
            for H in Hs: z[:, :, H] = z_mean[L][:, :, H]
            return z
        return fn
    hooks.append((f"blocks.{L}.attn.hook_z", mk(L, Hs)))
ldB = ld(m.run_with_hooks(toks, fwd_hooks=hooks))
effB = clean_ld - ldB
print("B: joint mean-abl name movers: LD", round(ldB,3), "effect", round(effB,3))
okA = abs(a) <= 0.1; okB = effB >= 1.0
print("A", "PASS" if okA else "FAIL", "| B", "PASS" if okB else "FAIL")
json.dump({"A_dla99_under_abl": a, "B_mean_abl_nm_effect": effB, "clean_ld": clean_ld,
           "A_pass": bool(okA), "B_pass": bool(okB)},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sanity3.json"), "w"), indent=1)
