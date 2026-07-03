#!/usr/bin/env python3
"""CRC external test — ROUND 3 (break it).

Hardest test yet:
 (1) CROSS-FAMILY: GPT-2 (124M) vs Pythia-160m — DIFFERENT tokenizers. The
     causal-role signature is measured over a shared *string* vocabulary (words
     that are single-token in BOTH), looking up each model's own token id. This
     is the real "across model instance / family" case CRC claims.
 (2) MANY CONFUSABLE categories (fine-grained: pets/farm/wild/sea animals,
     fruits/vegetables/meats/drinks, …) → chance ≈ 1/K, hard discrimination.
 (3) DECOUPLED metric: role fingerprint over a broad set of common NEUTRAL words
     (single-token in both), disjoint from every category answer word.
 + STRONG geometry baseline (paired-activation alignment; both d_model=768).

Run: crc-venv311/bin/python crc_transfer_test_r3.py
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np, torch

torch.set_grad_enabled(False); torch.set_num_threads(max(1, os.cpu_count() or 4))
OUT = Path(__file__).parent; FRAC_LAYER = 0.5; STEER_C = 0.5; SEEDS = list(range(8)); MIN_WORDS = 4

CATEGORIES = {
    "pets":       (["cat","dog","hamster","rabbit","parrot","kitten","puppy","goldfish"],
                   ["My pet is a","She adopted a little","The child wanted a","At home we keep a"]),
    "farm":       (["cow","pig","sheep","goat","horse","chicken","duck","hen"],
                   ["On the farm there was a","The farmer fed the","In the barn stood a","We milked the"]),
    "wild":       (["lion","tiger","bear","wolf","fox","deer","elephant","monkey"],
                   ["In the jungle lived a","The hunter tracked a","At the zoo we saw a","In the forest roamed a"]),
    "sea":        (["fish","shark","whale","crab","seal","dolphin","squid","octopus"],
                   ["In the ocean swam a","The diver spotted a","The net caught a","Near the reef was a"]),
    "birds":      (["eagle","owl","robin","crow","dove","hawk","sparrow","swan"],
                   ["High in the sky flew an","On the branch sat a","The nest belonged to a","We heard a"]),
    "fruits":     (["apple","banana","orange","grape","lemon","peach","cherry","mango"],
                   ["For dessert she ate an","The basket was full of","He picked a ripe","On the tree grew a"]),
    "vegetables": (["carrot","potato","onion","tomato","pepper","bean","cabbage","garlic"],
                   ["In the soup she added","The garden grew fresh","He chopped up a","The salad had some"]),
    "meats":      (["beef","pork","chicken","bacon","ham","lamb","turkey","steak"],
                   ["For dinner we grilled","The butcher sold fresh","The recipe needs some","He fried some"]),
    "drinks":     (["water","juice","milk","coffee","tea","wine","beer","soda"],
                   ["She poured a glass of","In the morning he drinks","The waiter brought some","I ordered a"]),
    "colors":     (["red","blue","green","yellow","black","white","brown","pink"],
                   ["Her favorite color is","The sky slowly turned","He painted the wall","My new shirt is"]),
    "metals":     (["gold","iron","silver","copper","steel","tin","lead","bronze"],
                   ["The ring was made of","The bridge is built from","They mined for","The coin was pure"]),
    "numbers":    (["one","two","three","four","five","six","seven","eight"],
                   ["The total came to","She counted up to","There were exactly","I bought just"]),
    "countries":  (["France","Japan","Brazil","Egypt","China","India","Spain","Italy"],
                   ["Last summer we flew to","She was born in","He is travelling to","The team came from"]),
    "cities":     (["London","Paris","Tokyo","Berlin","Moscow","Rome","Boston","Chicago"],
                   ["The conference is held in","She moved to","The train arrives in","He grew up in"]),
    "body":       (["hand","foot","head","arm","leg","eye","nose","finger"],
                   ["The doctor examined my","He gently touched her","She injured her left","She raised her"]),
    "clothing":   (["shirt","coat","hat","shoe","dress","sock","glove","jacket"],
                   ["In winter she wears a","He bought a new","On the hook hung a","He forgot his"]),
    "furniture":  (["chair","table","sofa","bed","desk","shelf","lamp","couch"],
                   ["In the corner stood a","She sat down on the","He built a wooden","They bought a new"]),
    "tools":      (["hammer","saw","drill","wrench","screwdriver","axe","knife","shovel"],
                   ["He reached for the","The carpenter used a","In the toolbox was a","She tightened it with a"]),
    "vehicles":   (["car","truck","bus","train","plane","boat","bike","ship"],
                   ["He drove a fast","They boarded the","At the station waited a","She rode her"]),
    "weather":    (["rain","snow","wind","storm","fog","sun","cloud","frost"],
                   ["Tomorrow expect heavy","The forecast calls for","Outside there was","The morning brought"]),
    "sports":     (["soccer","tennis","golf","boxing","hockey","cricket","rugby","chess"],
                   ["On weekends he plays","She is training for","The stadium hosted","His favorite sport is"]),
    "jobs":       (["doctor","teacher","lawyer","farmer","pilot","nurse","baker","judge"],
                   ["When she grows up she wants to be a","He works as a","The town needed a new","Ask the"]),
    "instruments":(["piano","guitar","violin","drum","flute","trumpet","harp","cello"],
                   ["She learned to play the","In the orchestra he plays","On stage stood a","He tuned his"]),
    "emotions":   (["joy","anger","fear","sadness","love","hope","pride","shame"],
                   ["Her heart was filled with","He could not hide his","The news brought great","Deep inside he felt"]),
}
NEUTRAL = ["The","I think that","Yesterday,","She said that","It was a","We went to","Here is the",
           "After a while,","In the morning","He looked at the","Later that day","They decided to"]
ALIGN_PROMPTS = NEUTRAL + ["The quick brown fox jumps","History shows that","In science we learn",
    "Once upon a time there","The economy grew because","When the sun rises","A good book will",
    "The river flowed past","Children love to play","Under the old bridge","Every year the town",
    "Scientists recently found","The ancient castle stood","On a clear night you can","People often say that"]
# broad common words for the DECOUPLED role fingerprint (disjoint from category answers)
BROAD = """time year people way day man thing woman life child world school state family student group
country problem hand part place case week company system program question work government number night point
home water room mother area money story fact month lot right study book eye job word business issue side kind
head house service friend father power hour game line end member law car city community name president team
minute idea body information back parent face others level office door health person art war history party
result change morning reason research girl guy moment air teacher force education foot boy age policy process
music market sense nation plan college interest death experience effort rule student""".split()

def log(*a): print(*a, file=sys.stderr, flush=True)
def load(n):
    from transformer_lens import HookedTransformer
    log(f"[load] {n} …"); m = HookedTransformer.from_pretrained(n, device="cpu"); m.eval(); return m
def stid(m, w):
    t = m.to_tokens(" "+w, prepend_bos=False)[0]
    return int(t[0]) if t.shape[0] == 1 else None
def resid_last(m, prompts, L):
    h = f"blocks.{L}.hook_resid_post"; out = []
    for p in prompts:
        _, c = m.run_with_cache(m.to_tokens(p), names_filter=h); out.append(c[h][0, -1].float())
    return torch.stack(out)
def steer_vecs(m, cats, L):
    per = {k: resid_last(m, t, L) for k, (_, t) in cats.items()}
    g = torch.cat(list(per.values()), 0).mean(0)
    return {k: per[k].mean(0)-g for k in cats}
def per_probe_delta(m, vecs, L, probes):
    h = f"blocks.{L}.hook_resid_post"; keys = list(vecs)
    tl = [m.to_tokens(p) for p in probes]
    clean = torch.stack([m(t)[0, -1].float() for t in tl])
    D = np.zeros((len(keys), len(probes), clean.shape[1]), np.float32)
    for ki, k in enumerate(keys):
        v = vecs[k]
        def fn(r, hook, v=v): r[:, -1, :] = r[:, -1, :]+v.to(r.dtype); return r
        for pi, t in enumerate(tl):
            D[ki, pi] = (m.run_with_hooks(t, fwd_hooks=[(h, fn)])[0, -1].float()-clean[pi]).numpy()
    return keys, D
def cos_rows(A, B):
    A = A/(np.linalg.norm(A, axis=1, keepdims=True)+1e-9); B = B/(np.linalg.norm(B, axis=1, keepdims=True)+1e-9)
    return A @ B.T
def top1(S): return float((S.argmax(1) == np.arange(S.shape[0])).mean())

def main():
    t0 = time.time()
    ma, mb = load("gpt2"), load("pythia-160m")        # DIFFERENT families/tokenizers
    La, Lb = round(ma.cfg.n_layers*FRAC_LAYER), round(mb.cfg.n_layers*FRAC_LAYER)

    # keep categories whose words are single-token in BOTH; drop thin ones
    cats, ans_ids_a, ans_ids_b = {}, {}, {}
    for k, (words, tmpl) in CATEGORIES.items():
        good = [(w, stid(ma, w), stid(mb, w)) for w in words]
        good = [(w, a, b) for (w, a, b) in good if a is not None and b is not None]
        if len(good) >= MIN_WORDS:
            cats[k] = (words, tmpl)
            ans_ids_a[k] = [a for _, a, _ in good]; ans_ids_b[k] = [b for _, _, b in good]
    keys = list(cats); K = len(keys)
    ans_a = sorted({i for v in ans_ids_a.values() for i in v})
    ans_b = sorted({i for v in ans_ids_b.values() for i in v})
    ans_words = {w for k in keys for w in cats[k][0]}
    # broad shared-string probe (single-token in both, not a category answer)
    broad = [(w, stid(ma, w), stid(mb, w)) for w in dict.fromkeys(BROAD)]
    broad = [(w, a, b) for (w, a, b) in broad if a is not None and b is not None and w not in ans_words]
    bA = np.array([a for _, a, _ in broad]); bB = np.array([b for _, _, b in broad])
    log(f"[cfg] K={K}/{len(CATEGORIES)} kept, layers A{La}/B{Lb}, broad probe={len(broad)} words")

    va, vb = steer_vecs(ma, cats, La), steer_vecs(mb, cats, Lb)
    na = resid_last(ma, NEUTRAL, La).norm(dim=-1).mean().item(); nb = resid_last(mb, NEUTRAL, Lb).norm(dim=-1).mean().item()
    for d, n in ((va, na), (vb, nb)):
        for k in d: d[k] = d[k]/d[k].norm()*(STEER_C*n)

    log("[deltas] gpt2 …"); _, Da = per_probe_delta(ma, va, La, NEUTRAL)
    log("[deltas] pythia …"); _, Db = per_probe_delta(mb, vb, Lb, NEUTRAL)

    # DECOUPLED role signature over broad shared strings (each model's own ids)
    sigA = Da.mean(1)[:, bA]; sigB = Db.mean(1)[:, bB]
    acc_dec = top1(cos_rows(sigA, sigB))
    # strong geometry baseline (both d_model 768): align on paired activations
    W = torch.linalg.lstsq(resid_last(ma, ALIGN_PROMPTS, La), resid_last(mb, ALIGN_PROMPTS, Lb)).solution.numpy()
    VA = np.stack([va[k].numpy() for k in keys]); VB = np.stack([vb[k].numpy() for k in keys])
    acc_geo = top1(cos_rows(VA @ W, VB))

    # transfer effect: B-mechanism chosen by each method, measured on intended category answers in B
    pred_dec = cos_rows(sigA, sigB).argmax(1); pred_geo = cos_rows(VA @ W, VB).argmax(1)
    def transfer(pred):
        return float(np.mean([Db[int(pred[i])][:, ans_ids_b[keys[i]]].mean() for i in range(K)]))
    tr_dec, tr_geo, tr_or = transfer(pred_dec), transfer(pred_geo), transfer(np.arange(K))

    # seeds: bootstrap over probe axis
    accs = []
    for s in SEEDS:
        idx = np.random.default_rng(s).integers(0, Da.shape[1], Da.shape[1])
        accs.append(top1(cos_rows(Da[:, idx].mean(1)[:, bA], Db[:, idx].mean(1)[:, bB])))
    dm, ds = float(np.mean(accs)), float(np.std(accs))
    chance = 1.0/K; gap = dm-acc_geo
    verdict = ("SUPPORTED" if gap >= 0.15 and dm > chance+1e-9
               else "REJECTED (renamed bundle)" if gap <= 0.05 else "INCONCLUSIVE")
    res = {"round": 3, "models": ["gpt2", "pythia-160m"], "cross_family": True,
           "K": K, "categories": keys, "chance": chance, "broad_probe_words": len(broad),
           "match_accuracy": {"crc_decoupled_mean": dm, "crc_decoupled_std": ds,
                              "crc_decoupled_point": acc_dec, "geometry_strong": acc_geo},
           "transfer_effect_logit": {"crc_decoupled": tr_dec, "geometry": tr_geo, "oracle": tr_or},
           "crc_minus_geometry": gap, "verdict_on_decoupled_metric": verdict,
           "crc_predictions": [keys[int(j)] for j in pred_dec],
           "geometry_predictions": [keys[int(j)] for j in pred_geo],
           "seconds": round(time.time()-t0, 1)}
    (OUT/"results_r3.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False)); return 0

if __name__ == "__main__":
    sys.exit(main())
