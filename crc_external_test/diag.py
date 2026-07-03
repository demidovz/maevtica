import os, torch, numpy as np
torch.set_grad_enabled(False); torch.set_num_threads(os.cpu_count() or 4)
from transformer_lens import HookedTransformer
def load(n):
    m=HookedTransformer.from_pretrained(n, device="cpu"); m.eval(); return m
def stid(m,w):
    t=m.to_tokens(" "+w,prepend_bos=False)[0]; return int(t[0]) if t.shape[0]==1 else None
def resid_last(m,ps,L):
    h=f"blocks.{L}.hook_resid_post"; out=[]
    for p in ps:
        _,c=m.run_with_cache(m.to_tokens(p),names_filter=h); out.append(c[h][0,-1].float())
    return torch.stack(out)
tmpl=["For dessert she ate an","The basket was full of","He picked a ripe","On the tree grew a"]
neutral=["The","I think that","It was a","We went to"]
for name,L in [("gpt2",6),("pythia-160m",6)]:
    m=load(name)
    per=resid_last(m,tmpl,L).mean(0)
    grand=resid_last(m,tmpl+neutral,L).mean(0)  # rough grand
    v=per-grand
    n=resid_last(m,neutral,L).norm(-1).mean().item()
    print(f"\n[{name}] resid_norm≈{n:.2f}  raw diff-vec norm={v.norm().item():.4f}")
    v=v/v.norm()*(0.5*n)
    print(f"  scaled steer-vec norm={v.norm().item():.2f}")
    h=f"blocks.{L}.hook_resid_post"
    def fn(r,hook,v=v): r[:,-1,:]=r[:,-1,:]+v.to(r.dtype); return r
    t=m.to_tokens("The")
    clean=m(t)[0,-1]; steer=m.run_with_hooks(t,fwd_hooks=[(h,fn)])[0,-1]
    d=(steer-clean)
    aid=stid(m,"apple")
    print(f"  delta on ' apple' logit = {float(d[aid]):.4f}   |delta| max={float(d.abs().max()):.3f} mean|d|={float(d.abs().mean()):.4f}")
    top=torch.topk(d,5).indices.tolist()
    print("  top-boosted tokens:", [repr(m.to_string([i])) for i in top])
