# 交接包⑨：M7 · 创新点 1 步 A（平衡 OT，ε 网格；云端，Cursor 执行）

代号：**I1-A** ｜ 产出：2026-09-03 ｜ 承接：Cursor → 云端 GPU ｜ 代码 `e17a9aa`

## 1. 目的与里程碑

**M7（前半）：创新点 1 实现就绪 + 步 A 平衡 OT 出数**。用不平衡 OT 软阶段分配替换
式(16) 固定窗口。步 A 固定 **λ=0、ρ=None（平衡）**，扫 **ε∈{0.01,0.05,0.1,0.5}**：
- **核心验证**：λ=0 平衡 OT 下 **OS 应≈0**（方案 §1.2 命题——顺序信息仅由 λ>0 注入的实证）；
- 给出"纯内容驱动"的精度基线，并按 π 熵淘汰 ε 过大（趋均匀）的档；
- 选出 ε*（Acc 与收敛残差综合），供步 B（扫 λ）用。

单测 U0–U4 本地全过（125 项，含 U1 排列不变 <1e-4、U2 带状极限、U3 梯度、U4 变 K）——**启动训练前置已满足**。

## 2. 版本与环境

- 代码：`main` @ **`e17a9aa`**（`ot_align.py` + `models.py` align_mode 分派 + `config_I1A_eps*.yaml`）。`git pull`（旧 clone `git fetch && git reset --hard origin/main`）。
- 环境/数据/CLIP：同 B0（torch 2.2.2、`data/full/`、`CLIP_VIT_B16_PATH`）。训练型 ~14.7GB，24GB 卡。
- **不改 B0**：`align_mode` 缺省 window，B0/3a/3b 配置逐位不变；OT 仅在本包 config 显式开启。

## 3. 执行步骤

### 3.0 smoke（必做，发全量前，手册 §8）
```bash
# 复制 config_mini.yaml → config_mini_ot.yaml，加：align_mode: ot / ot_eps: 0.05 /
# ot_lam: 0.0 / ot_rho: none / ot_iters: 30 / ot_weight: mass / frame_source: ca
TA_CONFIG=config_mini_ot.yaml python run.py
```
判据：端到端无报错、loss 有限无 NaN/Inf（fp32 岛）、checkpoint 存、终测出数。

### 3.1 每个 ε 一个闭环（train + dev 评测 + 3a 诊断）
```bash
# 以 ε=0.05 为例（001/01/05 同）
TA_CONFIG=config_I1A_eps005.yaml python run.py 2>&1 | tee I1A_eps005_console.log       # 训练10ep+dev终测1000ep
# 复诊断：复制 config_3a_final.yaml→config_3a_I1A_eps005.yaml，diagnose_episodes 改 1000、
#   checkpoint 改为上一步 best <acc>.tar，align_mode/ot_* 同训练配置（ot,ε=0.05,λ=0,ρ=none）
TA_CONFIG=config_3a_I1A_eps005.yaml python diagnose.py 2>&1 | tee I1A_eps005_diag.log
```
四个 ε 同结构，可串行（约 3.7h 训练 + 22min 诊断/档）。**π 熵接近上限（趋均匀）的档可提前淘汰、跳过其诊断并记录原因**。

## 4. 成功判据

- [ ] smoke 通过；U0–U4 前置已过（本地）；
- [ ] 四个 ε 各训练 10ep 完成、dev 1000ep 出 Acc、Sinkhorn 收敛残差记录；
- [ ] 各档复诊断出 **OS**（**平衡 OT 预期 OS≈0**；若显著≠0，回查诊断脚本排列是否施加在 enh 编码前，见手册 §七）；
- [ ] 记录每档 π 平均熵（趋均匀者淘汰）。

## 5. 判读（Claude 收包后，手册 §六 M7）

- **命题实证**：OS≈0 成立 → 方案 §1.2 命题在真实权重上确认（论文关键"预测→验证"表的一格）；
- ε*：在 OS≈0 的档中取 Acc 最高、π 熵未趋均匀、残差收敛者；
- 出数入 ledger（增补字段 align_mode/ot_eps/ot_lam/ot_rho/ot_iters/ot_weight/frame_source/ca_residual/conv_resid/pi_entropy/OS）。
- 下一包（步 B）：固定 ε*，扫 λ∈{0.1,0.3,1,3}，产 λ-Acc/λ-OS 双曲线（论文核心图，OS 应随 λ 单调上升）。

## 6. 需带回产物（放入 `experiments/returns/2026-09-03_M7-I1A/eps{001,005,01,05}/`）

每档：训练 `log.txt`+`console.log`、best `<acc>.tar`、dev 终测 Acc、复诊 `diagnose_3a_*.json`+`diag.log`、
`config` 快照、Sinkhorn 收敛残差与 π 熵（若脚本已记，否则从日志/诊断 JSON 提取）、`nvidia` 峰值与墙钟、异常全文。外加 smoke 日志。

收包后 Claude：核 OS≈0 命题 → 定 ε* → ledger/PROGRESS → 发步 B（λ 扫描）交接包。
