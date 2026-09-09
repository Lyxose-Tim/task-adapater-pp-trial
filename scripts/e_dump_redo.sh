#!/bin/bash
# 交接包⑯ 重导：仅修 vis/sem/fused 的 fp16 round 溢出；episode 流仍 12、seed 916。
set -euo pipefail
cd /root/rivermind-data/ta-pp
export CLIP_VIT_B16_PATH=/root/rivermind-data/ta-pp/dataset/checkpoints/ViT-B-16.pt
export PYTHONUNBUFFERED=1
PY=/opt/conda/bin/python
LOGROOT=/root/rivermind-data/ta-pp/workspace_E_dump_chain
mkdir -p "$LOGROOT"
exec > >(tee -a "$LOGROOT/chain_redo.log") 2>&1

echo "==== E DUMP REDO START $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
nvidia-smi || true

for cfg in config_E_dump_lam03 config_E_dump_B0 config_E_dump_lam0 \
           config_E_dump_lam03_s917 config_E_dump_lam03_rho1; do
  echo "==== DUMP ${cfg} START $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  TA_CONFIG=${cfg}.yaml "$PY" diagnose.py 2>&1 | tee "$LOGROOT/${cfg}_redo.log"
  echo "==== DUMP ${cfg} END $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
done

echo "==== E DUMP REDO DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
nvidia-smi || true
