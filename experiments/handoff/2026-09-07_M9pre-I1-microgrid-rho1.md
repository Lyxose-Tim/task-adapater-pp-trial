# 交接包⑭：创新点 1 步 C · λ×ρ 2×2 微网格（ρ=1 真实补偿检验，云端，Cursor 执行）

代号：**I1-mg** ｜ 产出：2026-09-07 ｜ 承接：Cursor → 云端 GPU ｜ 代码 `30ed45d`

## 1. 目的与里程碑

M9 前置 · **ρ 的最后一次检验**。步 C dev 显示 ρ 无显著效应；机制诊断进一步发现 **ρ=10 的
τ=ρ/(ρ+ε)=0.99、col_residual≈0.008 ≈平衡 OT**，无法真正检验"不平衡补偿 λ=1 损失"。
故用户改用 **ρ=1（τ=0.91、col_residual~0.066 中度不平衡）**做 2×2 微网格：**λ∈{0.3,1}×ρ∈{None,1}**，
检验 **λ×ρ 交互**——有限 ρ 能否在 λ=1 时补偿正常精度或截断鲁棒性。**λ 冻结 0.3**（ρ 检验，不动 λ）。

## 2. 版本与环境

- 代码：`main` @ **`30ed45d`**（robust_eval 加 OS + stage_mass_relax/profile + per_episode dump；ot_plan_stats 加 stage_mass_mean）。`git pull`。
- 验证：本地 CPU 单测 **163/163**（含 col_residual 随 ρ 单调、stage_mass_mean 形状、window=None 逐位复现官方）。**环境四坑自查**（CLIP git+ / `CLIP_VIT_B16_PATH` / `method` / `num_workers`）。
- ckpt 依赖（云端应在 `dataset/checkpoints/`；缺则从对应 returns 上传）：`I1B_lam03_best.tar`(0.3,None)、`I1C_rho1_best.tar`(0.3,1)、`I1B_lam1_best.tar`(1,None)。仅 (1,1) 需新训。

## 3. 执行步骤

### 3.0 首跑 robust smoke（§9.4，20ep 验通路）
`config_robust_mg_l03r0.yaml` 的 `diagnose_episodes` 临时改 20 跑一次，确认输出四窗
`acc/OS(normal)/dAcc/stage_mass_relax(mean/median/p95)/stage_mass_profile[K]/per_episode_fused`，
且 normal Acc≈标准 C0；确认后改回 2500。

### 3.1 训练缺格 (λ=1, ρ=1)
```bash
TA_CONFIG=config_mg_lam1_rho1.yaml python run.py 2>&1 | tee mg_lam1_rho1_train.log
```
best `<val>.tar` **重命名/软链为** `I1C_lam1_rho1_best.tar` 放 `dataset/checkpoints/`（robust 配置已按此名固定引用）。

### 3.2 四格 2500ep 配对复评（同一批 2500 dev episode，seed 916）
```bash
for cfg in config_robust_mg_l03r0 config_robust_mg_l03r1 config_robust_mg_l1r0 config_robust_mg_l1r1; do
  TA_CONFIG=${cfg}.yaml python diagnose.py 2>&1 | tee ${cfg}.log
done
```
每格 `workspace_robust_mg_*/robust_eval/robust_eval_*.json`：四窗 acc±CI、normal 窗 OS±CI、
head/tail/shrink dAcc±CI、stage_mass_relax(mean/median/p95)、stage_mass_profile[K]、逐 episode `per_episode_fused`。

> DiD 与 I_Acc（交互项）由 Claude 收包时从四格 `per_episode_fused` 计算（四格同 episode 流对齐、跨配置配对）：
> `I_Acc = [Acc(1,1)−Acc(1,None)] − [Acc(0.3,1)−Acc(0.3,None)]`；三截断同型 `DiD_t`（对每窗做同样双重差）+ 配对 95% CI。

## 4. 成功判据（量化，可勾选）

- [ ] (1,1) 训练 10ep 完成、loss 有限无 NaN、best 改稳定名 `I1C_lam1_rho1_best.tar`；
- [ ] 四格各出四窗 acc + normal OS + 三截断 dAcc + stage_mass_relax + profile + per_episode；每格 normal Acc≈标准 C0；
- [ ] **stage_mass 机制自洽**：ρ=1 档 stage_mass_relax 显著>ρ=None 档(≈0)（确认 ρ=1 真不平衡）；掐头窗早阶段(k=0)质量较正常↓、去尾窗晚阶段(k=K−1)质量↓（截断合理改变阶段分布）；
- [ ] repeat_frac(2500ep) 记录（预期仍 ~4e-4 可忽略）。

（交互判读由 Claude 收包完成）：I_Acc 与三截断 DiD_t 的配对 95% CI —— 明确>0 则 ρ=1 在 λ=1 有补偿、议不平衡主线；含 0 或不利 → 冻结 **ε=0.1, λ=0.3, ρ=None**。**不再扩 ρ 网格，不用 M9 10000ep test 重选参**。

## 5. 预计显存 / 时长

- 训练 (1,1)：~2h24m，14.7GB。
- 四格 robust 2500ep（四窗+OS+stage_mass）：~1.5h/格 ×4 ≈ 6h，~1.7GB。
- 合计 ~8.5h（可断点）。**首跑标"待实测"**。

## 6. 需带回产物（放入 `experiments/returns/2026-09-07_M9pre-I1-microgrid/`）

- `train/`：(1,1) `log.txt` + `mg_lam1_rho1_train.log` + config 快照；
- `robust/`：四格 `robust_eval_*.json` + `.log` + config 快照；
- `nvidia-smi` 峰值、墙钟、异常全文（若有）。

收包后 Claude：算 I_Acc + 三截断 DiD_t（配对 CI）→ 检验 λ×ρ 交互与 stage_mass 机制 →
**冻结 ρ*（预期 None，除非交互明确为正）** → 发 M9 10000ep 终表包（①B0/②OTλ0/③+λ0.3/④+ρ* + 截断 + 1 种子）。
