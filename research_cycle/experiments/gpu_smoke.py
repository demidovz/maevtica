import time, torch
print("torch", torch.__version__, "| cuda_available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA НЕ доступна — torch не видит 3050"
dev = "cuda"
print("gpu:", torch.cuda.get_device_name(0),
      "| vram_total_GB:", round(torch.cuda.get_device_properties(0).total_memory/1e9, 2))

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(dev)
tok.pad_token = tok.eos_token
n_params = sum(p.numel() for p in model.parameters())
print(f"gpt2 loaded on GPU | params={n_params/1e6:.1f}M")

# крохотный дообучив: заставим модель заучить одну фразу — лосс должен падать
text = "The studio's mascot is a small blue robot named Kolibri who loves fixing bugs."
batch = tok([text]*4, return_tensors="pt", padding=True).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.train()
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
losses = []
for step in range(30):
    opt.zero_grad()
    out = model(**batch, labels=batch["input_ids"])
    out.loss.backward()
    opt.step()
    if step % 6 == 0 or step == 29:
        losses.append((step, round(out.loss.item(), 4)))
dt = time.time() - t0
peak = torch.cuda.max_memory_allocated()/1e9
print("loss по шагам:", losses)
print(f"30 шагов full fine-tune за {dt:.1f}s ({dt/30*1000:.0f} ms/шаг)")
print(f"пик VRAM: {peak:.2f} GB из 4.0")
ok = losses[-1][1] < losses[0][1] - 0.5
print("ВЕРДИКТ:", "GPU ДООБУЧАЕТ GPT-2 ✓ (лосс упал)" if ok else "лосс не упал — разобраться")
print("GPU_SMOKE_DONE")
