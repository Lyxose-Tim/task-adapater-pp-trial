#!/bin/bash
# 交接包⑮ I1-M9：主线 seed917 训练 + 4 行终表 + seed917 各 10000ep diagnose。
# 前置：20ep smoke 已通过（本脚本不含 smoke）。
set -euo pipefail
cd /root/rivermind-data/ta-pp
export CLIP_VIT_B16_PATH=/root/rivermind-data/ta-pp/dataset/checkpoints/ViT-B-16.pt
export PYTHONUNBUFFERED=1
PY=/opt/conda/bin/python
LOGROOT=/root/rivermind-data/ta-pp/workspace_M9_chain
mkdir -p "$LOGROOT"
exec > >(tee -a "$LOGROOT/chain.log") 2>&1

echo "==== M9 CHAIN START $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
nvidia-smi || true

echo "==== TRAIN seed917 START $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
TA_CONFIG=config_M9_seed917_train.yaml "$PY" run.py 2>&1 | tee "$LOGROOT/M9_seed917_train.log"
echo "==== TRAIN seed917 END $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="

"$PY" - <<'PY'
import os, glob
root = "workspace_M9_seed917_lam03/taskadapter/checkpoints/somethingcmn/5way_1shot_aug"
runs = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))
assert runs, "no train run dir under " + root
latestdir = runs[-1]
cands = []
for f in os.listdir(latestdir):
    if not f.endswith(".tar"):
        continue
    try:
        cands.append((float(f[:-4]), f))
    except ValueError:
        pass
assert cands, "no val-acc .tar in " + latestdir
best = max(cands)[1]
src = os.path.abspath(os.path.join(latestdir, best))
dst = "dataset/checkpoints/I1C_lam03_seed917_best.tar"
if os.path.lexists(dst):
    os.remove(dst)
os.symlink(src, dst)
print("LINKED", src, "->", dst, "val=", max(cands)[0])
PY

for cfg in config_M9_row1_B0 config_M9_row2_lam0 config_M9_row3_lam03 config_M9_row4_lam03rho1 config_M9_row3_seed917; do
  echo "==== DIAG ${cfg} START $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  TA_CONFIG=${cfg}.yaml "$PY" diagnose.py 2>&1 | tee "$LOGROOT/${cfg}.log"
  echo "==== DIAG ${cfg} END $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
done

echo "==== M9 CHAIN DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
nvidia-smi || true
