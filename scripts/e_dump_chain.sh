#!/bin/bash
# 交接包⑯：五档 plan_dump（只推理 12 episode）。首档 lam03 须已通过。
set -euo pipefail
cd /root/rivermind-data/ta-pp
export CLIP_VIT_B16_PATH=/root/rivermind-data/ta-pp/dataset/checkpoints/ViT-B-16.pt
export PYTHONUNBUFFERED=1
PY=/opt/conda/bin/python
LOGROOT=/root/rivermind-data/ta-pp/workspace_E_dump_chain
mkdir -p "$LOGROOT"
exec > >(tee -a "$LOGROOT/chain.log") 2>&1

echo "==== E DUMP CHAIN START $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
nvidia-smi || true

for cfg in "$@"; do
  echo "==== DUMP ${cfg} START $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  TA_CONFIG=${cfg}.yaml "$PY" diagnose.py 2>&1 | tee "$LOGROOT/${cfg}.log"
  echo "==== DUMP ${cfg} END $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
done

echo "==== E DUMP CHAIN DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
nvidia-smi || true
