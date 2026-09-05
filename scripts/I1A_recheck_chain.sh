#!/bin/bash
# I1-A-recheck: 4 eps diagnose + eps01 iters=60. Survives Cursor disconnect.
set -euo pipefail
cd /root/rivermind-data/ta-pp
export CLIP_VIT_B16_PATH=/root/rivermind-data/ta-pp/dataset/checkpoints/ViT-B-16.pt
export PYTHONUNBUFFERED=1
CHAINLOG=/root/rivermind-data/I1A_recheck_chain.log
PY=/opt/conda/bin/python

log() { echo "[$(date -Iseconds)] $*" | tee -a "$CHAINLOG"; }

run_diag() {
  local cfg="$1"
  local logf="$2"
  log "START TA_CONFIG=$cfg"
  TA_CONFIG="$cfg" "$PY" diagnose.py > "$logf" 2>&1
  local rc=$?
  log "DONE TA_CONFIG=$cfg exit=$rc log=$logf"
  if [ "$rc" -ne 0 ]; then
    log "ABORT on $cfg"
    tail -n 40 "$logf" | tee -a "$CHAINLOG"
    exit "$rc"
  fi
}

: > "$CHAINLOG"
log "I1A_recheck chain start"
for f in \
  dataset/checkpoints/I1A_eps001_56.32.tar \
  dataset/checkpoints/I1A_eps005_best.tar \
  dataset/checkpoints/I1A_eps01_best.tar \
  dataset/checkpoints/I1A_eps05_best.tar; do
  if [ ! -f "$f" ]; then
    log "MISSING ckpt $f"
    exit 1
  fi
done

run_diag config_3a_I1A_recheck_eps001.yaml /root/rivermind-data/I1A_eps001_recheck.log
run_diag config_3a_I1A_recheck_eps005.yaml /root/rivermind-data/I1A_eps005_recheck.log
run_diag config_3a_I1A_recheck_eps01.yaml /root/rivermind-data/I1A_eps01_recheck.log
run_diag config_3a_I1A_recheck_eps05.yaml /root/rivermind-data/I1A_eps05_recheck.log
run_diag config_3a_I1A_eps01_iters60.yaml /root/rivermind-data/I1A_eps01_iters60.log
log "ALL_DONE"
