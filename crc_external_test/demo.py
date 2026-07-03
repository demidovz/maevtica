import os, torch
torch.set_grad_enabled(False); torch.set_num_threads(2)
from transformer_lens import HookedTransformer
m=HookedTransformer.from_pretrained("gpt2",device="cpu"); m.eval()
prompts=["The farmer fed the","For dessert she ate an","Her favorite color is",
         "The ring was made of","In winter she wears a","The"]
for p in prompts:
    logits=m(m.to_tokens(p))[0,-1]
    top=torch.topk(logits,5).indices.tolist()
    words=[repr(m.to_string([i]).strip()) for i in top]
    print(f'  «{p} …»  → ИИ хочет сказать: {", ".join(words)}')
