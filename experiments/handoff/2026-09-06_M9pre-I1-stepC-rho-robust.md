# 交接包⑬：创新点 1 步 C · ρ 不平衡扫描 + 阶段 D 时序鲁棒截断评测（云端，Cursor 执行）

代号：**I1-C** ｜ 产出：2026-09-06 ｜ 承接：Cursor → 云端 GPU ｜ 代码 `149f555`

## 1. 目的与里程碑

M9 前置。**λ\* 已由用户拍板冻结＝0.3（基于 dev 证据）**，ε\*=0.1。本包固定 ε=0.1、λ=0.3，
**扫 ρ∈{None, 10, 1, 0.1}**（None=平衡=已训的 I1B_lam03，不重训），每档跑 **3a 诊断(OS)**
与 **阶段 D 四窗时序鲁棒截断评测**；并对 **B0(固定窗口)** 与 **OT 平衡(λ=0)** 补同口径截断评测
作对照。手册 §二步 C、§四、§三消融行 ④。

**用户护栏（务必遵守）**：λ **冻结 0.3**；如需额外确认只把配对 validation 从 1000 扩到 **2500**；
**不得用最终 10000ep test 重选 λ/ρ**。**M9 10000ep 终表与追加训练种子，待 λ/ρ 与 2×2 微网格全部冻结后**
另包统一跑（不在本包）。

## 2. 版本与环境

- 代码：`main` @ **`149f555`**（`dataset.sample_frame_ids`/`sample_window` 贯穿 + `diagnose.robust_eval` +
  `robust_eval` 分发）。`git pull`。
- **零漂移保证**：`sample_window=None` 逐位复现官方评测采样（单测 nf∈{8..100} 校验），训练路径与
  B0/现有评测数值不变；截断仅评测期生效。
- 验证记录：**推理型评测新增 + 训练沿用步 B**。本地 CPU 单测 **162/162**（含 +26 截断：window=None≡官方、
  三窗时间区间/1-based/单调/界内、边界重复占比）。**环境四坑自查**：CLIP git+ 装、`CLIP_VIT_B16_PATH`
  指权重、config 含 `method`、`num_workers` 按核数。
- ckpt 依赖（云端应在 `dataset/checkpoints/`；若缺从对应 returns 上传）：`b0_seed916_56.32.tar`、
  `I1A_eps01_best.tar`（λ=0）、`I1B_lam03_best.tar`（λ=0.3, ρ=None）。

## 3. 执行步骤

### 3.0 首跑小规模试跑（§9.4，仅验 robust_eval 新通路）
把 `config_robust_lam03.yaml` 的 `diagnose_episodes` 临时改 20 跑一次，确认
`workspace_robust_lam03/robust_eval/robust_eval_*.json` 正常出四窗（normal/head/tail/shrink）
Acc + dAcc + repeat_frac，且 **normal 窗 Acc ≈ 该档标准 C0**（口径自洽）；确认后改回 1000。

### 3.1 ρ 扫描训练（3 档，None 不训）
```bash
for tag in rho10 rho1 rho01; do
  TA_CONFIG=config_I1C_${tag}.yaml python run.py 2>&1 | tee I1C_${tag}_train.log
done
```
各档训练完毕后，将 best `<val>.tar` **重命名/软链为稳定名** `I1C_${tag}_best.tar` 放 `dataset/checkpoints/`
（与步 B `I1B_lam03_best.tar` 同惯例，便于下游配置固定引用）。

### 3.2 3a 诊断（OS，3 新档；ρ=None 复用步 B 的 I1B_lam03 诊断，不重跑）
由训练配置派生（`test_model:True` + `checkpoint:dataset/checkpoints/I1C_${tag}_best.tar` +
`frame_perm_seed:916`，其余不变），命名 `config_3a_I1C_${tag}.yaml`：
```bash
for tag in rho10 rho1 rho01; do
  TA_CONFIG=config_3a_I1C_${tag}.yaml python diagnose.py 2>&1 | tee I1C_${tag}_diag.log
done
```

### 3.3 阶段 D 四窗时序鲁棒评测（6 档）
- 3 档已固定配置：`config_robust_lam03.yaml`（ρ=None）、`config_robust_lam0.yaml`（λ=0）、
  `config_robust_B0.yaml`（固定窗口）。
- 3 档新 ρ 由 `config_robust_lam03.yaml` 派生：改 `checkpoint→I1C_${tag}_best.tar`、
  `ot_rho→{10,1,0.1}`、`work_dir/message`，命名 `config_robust_${tag}.yaml`。
```bash
for cfg in config_robust_B0 config_robust_lam0 config_robust_lam03 \
           config_robust_rho10 config_robust_rho1 config_robust_rho01; do
  TA_CONFIG=${cfg}.yaml python diagnose.py 2>&1 | tee ${cfg}.log
done
```
每档输出 `workspace_*/robust_eval/robust_eval_*.json`：四窗 Acc±CI、逐 episode 配对 ΔAcc±CI、repeat_frac。

## 4. 成功判据（量化，可勾选）

- [ ] 3 档 ρ 训练各 10ep 完成、loss 有限无 NaN、best 保存并改稳定名；
- [ ] 3 新档 3a 诊断出 OS（dev 1000ep）、sanity=true；对照 ρ=None(I1B_lam03) OS=0.30；
- [ ] 6 档 robust_eval 各出四窗 Acc + ΔAcc + repeat_frac；**每档 normal 窗 Acc ≈ 其标准评测 C0**（口径自洽 sanity）；
- [ ] 核心对照：截断 ΔAcc 在 B0 / OT平衡(λ0) / OT+λ(ρNone) / OT+λ+ρ{10,1,0.1} 间给出——
      检验手册预期"**不平衡档 ρ 退化最小**"是否成立（成立与否都如实报，若固定窗口反更稳则按 §四回到轴上解释）；
- [ ] repeat_frac 若某窗偏高（短视频多），报告分层说明。

## 5. 预计显存 / 时长 / 费用

- 训练：~2h24m/档 × 3 ≈ **7.2h**，显存 ~14.7GB。
- robust 评测：四窗 × 1000ep，约 ~35–45min/档 × 6 ≈ **~4h**，显存 ~1.7GB。
- 合计 ~11h（可断点：训练与评测分步）。**首跑标"待实测"**。

## 6. 需带回产物（放入 `experiments/returns/2026-09-06_M9pre-I1-stepC/`）

- `rho{10,1,01}/`：训练 `log.txt`、`I1C_*_train.log`、3a `diagnose_3a_*.json` + `I1C_*_diag.log`、config 快照；
- `robust/`：6 档 `robust_eval_*.json` + `.log` + config 快照；
- `nvidia-smi` 峰值、墙钟、异常全文（若有）。

收包后 Claude：解析 → 台账增补 `ot_rho / dAcc_head / dAcc_tail / dAcc_shrink`（手册 §八字段）→
定 ρ\*（主看截断 ΔAcc，Acc 不劣于 ρ=None）→ 发 **2×2 λ×ρ 微网格**复核包 → 全冻后发 **M9 10000ep 终表 + 种子**包。
