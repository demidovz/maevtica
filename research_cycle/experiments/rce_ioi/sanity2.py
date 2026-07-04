#!/usr/bin/env python3
"""Supplemental positive controls — see SANITY_ADDENDUM.md (written first)."""
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

U = m.W_U
udiff = (U[:, io_ids] - U[:, s_ids]).T.float()
rfilt = lambda n: n.endswith("hook_result") or n == "ln_final.hook_scale"
lg, c = m.run_with_cache(toks, names_filter=rfilt)
clean_ld = ld(lg)
scale = c["ln_final.hook_scale"][:, -1].float()
dla = np.zeros((nL, nH), np.float32)
for L in range(nL):
    r = c[f"blocks.{L}.attn.hook_result"][:, -1].float()
    r = (r - r.mean(-1, keepdim=True)) / scale.unsqueeze(1)
    dla[L] = (r * udiff.unsqueeze(1)).sum(-1).mean(0).numpy()

top = np.unravel_index(np.argmax(dla), dla.shape)
print("clean_ld", round(clean_ld, 3))
print("DLA 9.9 =", round(float(dla[9,9]), 3), "| max-DLA head =", f"{top[0]}.{top[1]}",
      "=", round(float(dla.max()), 3))
print("top-5 DLA:", sorted([(round(float(dla[L,H]),3), f"{L}.{H}") for L in range(nL) for H in range(nH)], reverse=True)[:5])

NM = [(9,9),(9,6),(10,0)]
def zero_hooks(S):
    byL = {}
    for L,H in S: byL.setdefault(L, []).append(H)
    def mk(Hs):
        def fn(z, hook):
            for H in Hs: z[:, :, H] = 0.0
            return z
        return fn
    return [(f"blocks.{L}.attn.hook_z", mk(Hs)) for L, Hs in byL.items()]

ld_nm = ld(m.run_with_hooks(toks, fwd_hooks=zero_hooks(NM)))
eff = clean_ld - ld_nm
print("joint zero-abl {9.9,9.6,10.0}: LD", round(ld_nm,3), "effect", round(eff,3))

ok = float(dla[9,9]) >= 1.0 and tuple(top) == (9,9) and eff >= 1.0
print("SANITY2:", "PASS" if ok else "FAIL")
json.dump({"clean_ld": clean_ld, "dla_9_9": float(dla[9,9]),
           "max_dla_head": f"{top[0]}.{top[1]}", "joint_nm_zero_effect": eff,
           "pass": bool(ok)},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sanity2.json"), "w"), indent=1)
