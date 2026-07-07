#!/usr/bin/env python3
"""Phase 0a: the simplest 'poised-to-say' lens — the logit-lens. Unembed each mid-layer
residual through the model's own output head => what the model would say if it stopped there.
Look at (1) a sanity where the answer should EMERGE across layers, (2) an UNSPOKEN intermediate
(does 'spider' show up before the number?). Raw logit-lens is known to be noisy in early layers
for gpt2 (that's exactly why the J-lens/tuned-lens exist) — this tells us if we need Phase 0b."""
import os, torch
torch.set_num_threads(max(1,os.cpu_count() or 4))
from transformer_lens import HookedTransformer
m=HookedTransformer.from_pretrained("gpt2",device="cpu"); m.eval()

@torch.no_grad()
def lens(prompt, topk=6):
    toks=m.to_tokens(prompt); _,cache=m.run_with_cache(toks)
    print(f"\nPROMPT {prompt!r}  -> model says next: "
          f"{[m.to_string(t).strip() for t in torch.topk(m(toks)[0,-1],3).indices]}")
    for L in range(m.cfg.n_layers):
        r=cache["resid_post",L][:,-1:,:]
        lg=m.unembed(m.ln_final(r))[0,-1]
        top=torch.topk(lg,topk).indices
        print(f"  L{L:2d}: {[m.to_string(t).strip() for t in top]}")

for p in ["The capital of France is",
          "The animal that spins webs has",
          "The opposite of hot is",
          "A dog is a kind of",
          "7 plus 5 equals"]:
    lens(p)
print("\n[done]",flush=True)
