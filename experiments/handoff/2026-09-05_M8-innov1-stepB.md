# 交接包⑩：M8 · 创新点 1 步 B（顺序先验 λ 扫描；云端，Cursor 执行）

代号：**I1-B** ｜ 产出：2026-09-05 ｜ 承接：Cursor → 云端 GPU ｜ 代码 `e17a9aa`（+step B 配置）

## 1. 目的与里程碑

**M8（前半）：步 B 顺序先验 λ 扫描**。固定步 A 选出的 **ε\*=0.1**，扫
**λ∈{0.1, 0.3, 1, 3}**（ρ=None 平衡），产出 **λ-Acc 与 λ-OS 双曲线**（论文核心图）。

- 步 A 已实证：λ=0 时 **OS≡0**（方案 §1.2 命题，dev 复诊 OS=0.000）——顺序信息仅由 λ 注入。
- 步 B 核心问题：**OS 是否随 λ 单调上升**（预期是）；**Acc 是否有内点最优**（本步核心，需实验答）。

## 2. 版本与环境

- 代码：`main` HEAD（`ot_align.py`+`models.align_mode`，`e17a9aa` 未变；新增 `config_I1B_lam*.yaml`）。`git pull`。
- 环境/数据/CLIP/显存：同步 A（torch 2.2.2、`data/full/`、24GB 卡 ~14.7GB）。
- 步 A smoke 已过（OT 训练图通路验证：mini Loss 1.53→1.06、终测出数）；步 B 仅改 λ，**无需再 smoke**。

## 3. 执行步骤（每 λ 一个闭环，与步 A 同结构）

```bash
# 以 λ=1 为例（01/03/1/3 同）
TA_CONFIG=config_I1B_lam1.yaml python run.py 2>&1 | tee I1B_lam1_console.log        # 训练10ep + dev终测1000ep
# 复诊断：复制 config_3a_I1A_eps01.yaml→config_3a_I1B_lam1.yaml，改 checkpoint 为上步 best <acc>.tar、
#   ot_lam 改 1（其余 align_mode=ot/ot_eps=0.1/ot_rho=none 保持），diagnose_episodes=1000
TA_CONFIG=config_3a_I1B_lam1.yaml python diagnose.py 2>&1 | tee I1B_lam1_diag.log
```
四个 λ 同结构，可串行（约 3.7h 训练 + 22min 诊断/档，共 ~16h）。

## 4. 成功判据

- [ ] 四个 λ 各训练 10ep 完成、dev 1000ep 出 Acc；
- [ ] 各档复诊断出 OS（sanity 仍 true）；预期 OS 随 λ 单调上升（对照步 A λ=0 的 OS≡0）；
- [ ] λ-Acc 与 λ-OS 两条曲线数据点齐全（含步 A 的 λ=0 点：Acc 57.56 / OS 0.000）。

## 5. 判读（Claude 收包后，手册 §六 M8）

- **λ-OS 曲线**：应从 λ=0 的 0 单调上升 → 证明"顺序信息由 λ 可调注入"（方案 §1.2 后果，论文核心图之一）；
- **λ-Acc 曲线**：找内点最优 λ\*（若有）——OT 相对 B0(56.92) 与固定窗口的优劣就在这根曲线上；
- λ\* 供步 C（扫 ρ 不平衡 + 时序鲁棒性截断评测）；
- 入 ledger（config代号 I1-B-lam{λ} + 备注记 OS/Acc；开发规模标注）。

## 6. 需带回产物（放入 `experiments/returns/2026-09-05_M8-I1B/lam{01,03,1,3}/`）

每档：训练 `log.txt`+`console.log`、best `<acc>.tar`、dev 终测 Acc、复诊 `diagnose_3a_*.json`+`diag.log`、
`config` 快照、`nvidia` 峰值与墙钟、异常全文。

收包后 Claude：画 λ-Acc/λ-OS 双曲线 → 定 λ\* → ledger/PROGRESS → 发步 C（ρ 扫描 + 阶段 D 截断评测）交接包。
