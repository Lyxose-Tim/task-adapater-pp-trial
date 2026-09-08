#!/bin/bash
# 交接包⑭ I1-mg：先训缺格 (λ=1,ρ=1)，再四格 2500ep 配对 robust。
# 前置：20ep smoke 已通过（本脚本不含 smoke）。
set -euo pipefail
cd /root/rivermind-data/ta-pp
export CLIP_VIT_B16_PATH=/root/rivermind-data/ta-pp/dataset/checkpoints/ViT-B-16.pt
export PYTHONUNBUFFERED=1
PY=/opt/conda/bin/python
LOGROOT=/root/rivermind-data/ta-pp/workspace_mg_chain
mkdir -p "$LOGROOT"
exec > >(tee -a "$LOGROOT/chain.log") 2>&1

echo "==== MG CHAIN START $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
nvidia-smi || true

# --- 3.1 训练缺格 (λ=1, ρ=1) ---
echo "==== TRAIN (1,1) START $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
TA_CONFIG=config_mg_lam1_rho1.yaml "$PY" run.py 2>&1 | tee "$LOGROOT/mg_lam1_rho1_train.log"
echo "==== TRAIN (1,1) END $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="

"$PY" - <<'PY'
import os, glob
root = "workspace_mg_lam1_rho1/taskadapter/checkpoints/somethingcmn/5way_1shot_aug"
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
dst = "dataset/checkpoints/I1C_lam1_rho1_best.tar"
if os.path.lexists(dst):
    os.remove(dst)
os.symlink(src, dst)
print("LINKED", src, "->", dst, "val=", max(cands)[0])
PY

# --- 3.2 四格 2500ep 配对复评 ---
for cfg in config_robust_mg_l03r0 config_robust_mg_l03r1 config_robust_mg_l1r0 config_robust_mg_l1r1; do
  echo "==== ROBUST ${cfg} START $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  TA_CONFIG=${cfg}.yaml "$PY" diagnose.py 2>&1 | tee "$LOGROOT/${cfg}.log"
  echo "==== ROBUST ${cfg} END $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
done

echo "==== MG CHAIN DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
nvidia-smi || true
