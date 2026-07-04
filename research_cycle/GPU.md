# GPU-окружение студии (ярус B)

Отдельный venv, чтобы не сломать рабочее CPU-окружение (`crc-venv311`, на нём
крутятся дневные/ночные кампании петли).

- **venv:** `~/.local/state/mst/gpu-venv` — torch 2.5.1+**cu121**, transformers, peft, accelerate, socksio.
- **железо:** RTX 3050 Laptop, **3.95 ГБ VRAM** (потолок), i7-12650H / 15 ГБ RAM.
- **проверено (2026-07-04):** полный fine-tune GPT-2 124М — 0.78 с/шаг, пик 2.53 ГБ; лосс падает. Тест: `experiments/gpu_smoke.py`.

## Как запускать
```
export HF_HOME=~/.local/state/mst/hf-cache TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
~/.local/state/mst/gpu-venv/bin/python <script>.py   # .to("cuda")
```
- `HF_HOME` → общий кэш моделей (gpt2 уже там); офлайн-флаги обходят SOCKS-прокси.
- Без офлайна нужен `socksio` (стоит) — иначе httpx падает на SOCKS-прокси.

## Потолки (4 ГБ)
- полный FT: ≤ ~160М (GPT-2, Pythia-160М) — влезает.
- LoRA: модели покрупнее (учим доли % весов).
- ≥ ~1B или тренировка с нуля «настоящих» — нужен облачный GPU.

Назначение: «дистилляция обратно в веса» из VISION_reflection_intelligence.md.
