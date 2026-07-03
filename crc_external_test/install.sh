#!/usr/bin/env bash
set -e
export $(grep -E '^(HTTP_PROXY|HTTPS_PROXY)=' ~/.config/mst/.env 2>/dev/null | xargs)
export UV_HTTP_TIMEOUT=300
VENV=~/.local/state/mst/crc-venv311
echo "[install] создаю venv (py3.11) через uv…"
uv venv --python 3.11 "$VENV" 2>&1
echo "[install] torch (cpu)…"
uv pip install --python "$VENV/bin/python" torch --index-url https://download.pytorch.org/whl/cpu 2>&1
echo "[install] transformer_lens + sklearn…"
uv pip install --python "$VENV/bin/python" transformer_lens scikit-learn 2>&1
echo "[install] ГОТОВО. Проверка импорта:"
"$VENV/bin/python" -c "import torch, transformer_lens, sklearn; print('torch',torch.__version__,'| tl',transformer_lens.__version__,'| sklearn OK')"
echo "[install] === DONE ==="
