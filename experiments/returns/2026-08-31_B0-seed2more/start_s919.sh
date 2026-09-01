#!/bin/bash
set -euo pipefail
cd /root/rivermind-data/ta-pp
export CLIP_VIT_B16_PATH=/root/rivermind-data/ta-pp/dataset/checkpoints/ViT-B-16.pt
export TA_CONFIG=config_b0_seed919.yaml
export PYTHONUNBUFFERED=1
pkill -f 'nvidia-smi --query-gpu=timestamp' >/dev/null 2>&1 || true
nohup nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 30 \
  > /root/rivermind-data/nvidia_s919.csv 2>&1 &
echo $! > /root/rivermind-data/nvidia_s919.pid
nohup /opt/conda/bin/python run.py > /root/rivermind-data/b0_s919_console.log 2>&1 &
echo $! > /root/rivermind-data/b0_s919.pid
sleep 2
PY=$(cat /root/rivermind-data/b0_s919.pid)
echo PYTHON_PID=$PY
echo NVIDIA_PID=$(cat /root/rivermind-data/nvidia_s919.pid)
echo CWD=$(readlink /proc/$PY/cwd)
tr '\0' '\n' < /proc/$PY/environ | grep -E '^(TA_CONFIG|CLIP_VIT_B16_PATH|PWD)='
ps -fp $PY
head -5 /root/rivermind-data/b0_s919_console.log || true
