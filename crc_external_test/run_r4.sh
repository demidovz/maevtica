#!/usr/bin/env bash
export $(grep -E '^(HTTP_PROXY|HTTPS_PROXY)=' ~/.config/mst/.env 2>/dev/null | xargs)
export HF_HOME=~/.local/state/mst/hf-cache TOKENIZERS_PARALLELISM=false
~/.local/state/mst/crc-venv311/bin/python crc_transfer_test_r4.py
echo "=== RUN4 DONE exit=$? ==="
