#!/usr/bin/env python3
"""Gap #3 — reusable TEETH: primitives for real external tests of concepts.

Extracted from crc_external_test/crc_transfer_test_r{2,3,4}.py so a new test is
a short config, not a rewrite. Covers the common prediction shape:

  "matching internal mechanisms across models by <property X> transfers
   interventions better than matching by <baseline Y>."

The two hard-won lessons of 2026-07-03 are baked into code, not left to memory:
  • honest DECOUPLED signatures (measure role on side-effects, mask the answers);
  • an ORACLE SANITY GATE — verdict() refuses to conclude when the oracle/
    positive-control transfer is ~0 (that means the measurement is broken, which
    once faked a REJECTED via a .norm(-1) bug and once via zero-effect ablation).

venv: ~/.local/state/mst/crc-venv311/bin/python
"""
from __future__ import annotations
import numpy as np, torch, os
torch.set_grad_enabled(False); torch.set_num_threads(max(1, os.cpu_count() or 4))


def load(name):
    from transformer_lens import HookedTransformer
    m = HookedTransformer.from_pretrained(name, device="cpu"); m.eval(); return m


def single_tok_id(m, word):
    t = m.to_tokens(" " + word, prepend_bos=False)[0]
    return int(t[0]) if t.shape[0] == 1 else None


def resid_last(m, prompts, layer):
    h = f"blocks.{layer}.hook_resid_post"; out = []
    for p in prompts:
        _, c = m.run_with_cache(m.to_tokens(p), names_filter=h)
        out.append(c[h][0, -1].float())
    return torch.stack(out)


def steering_vectors(m, categories, layer, steer_c=0.5, neutral=None):
    """Diff-in-means direction per category, scaled to steer_c * mean resid norm.
    `categories`: {name: (words, eliciting_templates)}."""
    per = {k: resid_last(m, tmpl, layer) for k, (_, tmpl) in categories.items()}
    grand = torch.cat(list(per.values()), 0).mean(0)
    d = {k: per[k].mean(0) - grand for k in categories}
    ref = neutral or ["The", "It was", "I think that", "We went to"]
    n = resid_last(m, ref, layer).norm(dim=-1).mean().item()   # NB: dim=-1 (the .norm(-1) bug)
    return {k: d[k] / d[k].norm() * (steer_c * n) for k in d}


def per_probe_delta(m, vecs, layer, probes, mode="add"):
    """[K, n_probes, vocab] of (intervened - clean) last-token logits.
    mode: 'add' (steer) or 'ablate' (project the direction out)."""
    h = f"blocks.{layer}.hook_resid_post"; keys = list(vecs)
    tl = [m.to_tokens(p) for p in probes]
    clean = torch.stack([m(t)[0, -1].float() for t in tl])
    D = np.zeros((len(keys), len(probes), clean.shape[1]), np.float32)
    for ki, k in enumerate(keys):
        v = vecs[k]; u = v / v.norm()
        if mode == "add":
            def fn(r, hook, v=v): r[:, -1, :] = r[:, -1, :] + v.to(r.dtype); return r
        else:
            def fn(r, hook, u=u):
                r[:, -1, :] = r[:, -1, :] - (r[:, -1, :] @ u.to(r.dtype)).unsqueeze(-1) * u.to(r.dtype); return r
        for pi, t in enumerate(tl):
            D[ki, pi] = (m.run_with_hooks(t, fwd_hooks=[(h, fn)])[0, -1].float() - clean[pi]).numpy()
    return keys, D


def strong_geometry_align(ma, mb, la, lb, align_prompts):
    """Fair strong baseline: least-squares map A-space→B-space from PAIRED
    activations (same prompts to both). Requires equal d_model, or returns a
    rectangular map (works for cosine after projection)."""
    Xa = resid_last(ma, align_prompts, la); Xb = resid_last(mb, align_prompts, lb)
    return torch.linalg.lstsq(Xa, Xb).solution.numpy()


def cos_rows(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return A @ B.T


def top1(S):
    return float((S.argmax(1) == np.arange(S.shape[0])).mean())


def oracle_transfer(Db, answer_ids_b_by_cat, keys):
    """Positive control: applying the CORRECT mechanism must move the answer
    tokens. Near-zero ⇒ the measurement is broken."""
    return float(np.mean([Db[i][:, answer_ids_b_by_cat[keys[i]]].mean() for i in range(len(keys))]))


def verdict(crc_acc, geom_acc, chance, oracle, *, sup=0.15, rej=0.05, min_oracle=0.5):
    """Preregistered decision rule + ORACLE SANITY GATE.

    Returns one of: 'SUPPORTED', 'REJECTED', 'INCONCLUSIVE', or
    'BROKEN_MEASUREMENT' when |oracle| < min_oracle (refuse to conclude)."""
    if abs(oracle) < min_oracle:
        return "BROKEN_MEASUREMENT"          # the 2026-07-03 lesson, enforced
    gap = crc_acc - geom_acc
    if gap >= sup and crc_acc > chance + 1e-9:
        return "SUPPORTED"
    if gap <= rej:
        return "REJECTED"
    return "INCONCLUSIVE"
