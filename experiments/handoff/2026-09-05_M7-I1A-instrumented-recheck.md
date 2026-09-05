# 交接包⑪：M7 步 A · OT 数值健康补测（插桩复诊，不重训；云端，Cursor 执行）

代号：**I1-A-recheck** ｜ 产出：2026-09-05 ｜ 承接：Cursor → 云端 GPU ｜ 代码 `dcadf44`

## 1. 目的

对步 A 已训的 **4 个 ε checkpoint（不重训）** 用插桩后的 `diagnose.py` 补测，确认 OT 数值健康后
再固定 ε*=0.1 启动步 B。插桩为**纯日志、π.detach()，分数/梯度/显存不变**（本地单测已验
side-effect-free + 分数逐元素一致），故**步 A 无需重训**。全部为**推理型**（加载 ckpt，不训练）。

## 2. 版本与环境

- 代码：`main` @ **`dcadf44`**（`ot_align.ot_plan_stats` + `models.ot_diagnostics` + 插桩版 `diagnose.py`）。`git pull`。
- 环境/数据/CLIP：同步 A。ckpt 用 workspace_I1A_eps{001,005,01,05} 各自的 best `<acc>.tar`（云端应仍在；若已清，从 `experiments/returns/2026-09-03_M7-I1A/eps*/` 上传）。

## 3. 执行步骤（推理型，无训练）

### 3.1 四档 ε 插桩复诊（点 3）
对每个 ε，用**与原表相同的配对 episode**（seed 916、diagnose_episodes=1000、align_mode=ot、
对应 ot_eps/λ=0/ρ=none）复跑插桩 diagnose：
```bash
# 复用原 config_3a_I1A_eps{tag}.yaml（returns 里有；checkpoint 指向该档 best <acc>.tar）
TA_CONFIG=config_3a_I1A_eps01.yaml python diagnose.py 2>&1 | tee I1A_eps01_recheck.log
```
新 JSON 将含：`ot_plan_stats`（pi_entropy_norm / row_residual / col_residual / stage_mass_min /
total_mass 的 mean/p95/p99/max）、`u1_sem_max_abs_diff_C0_vs_C2`（正序 vs 乱序 sem 逐元素最大差）、
`per_episode_c0_fused`。四档各一次（~22min/档）。

### 3.2 iters 30 vs 60 只推理对照（点 4，仅 ε=0.1）
复制 `config_3a_I1A_eps01.yaml` → `config_3a_I1A_eps01_iters60.yaml`，改 `ot_iters: 60`，同 ckpt 复跑：
```bash
TA_CONFIG=config_3a_I1A_eps01_iters60.yaml python diagnose.py 2>&1 | tee I1A_eps01_iters60.log
```
对照 30 vs 60：C0 Acc、`per_episode_c0_fused` 预测一致率、row/col_residual（残差应随 iters 降或持平）。

## 4. 成功判据（数值健康 → 放行步 B）

- [ ] 四档 `u1_sem_max_abs_diff_C0_vs_C2` 的 max **≈0**（平衡 OT/λ=0 下 U1 逐元素成立，比 OS=0 更严；期望 <1e-2，理想 <1e-3）；
- [ ] ε*=0.1 的 `pi_entropy_norm` mean **未趋 1**（未塌成均匀）；row_residual/col_residual mean 小（平衡档 <1e-2）；stage_mass_min>0（无阶段被饿死）；
- [ ] iters 30 vs 60：C0 Acc 与逐 episode 预测**基本一致**（一致率高、残差不劣化）→ 确认 iters=30 足够；
- [ ] 插桩前后同 ckpt 的 C0 Acc 与原步 A 表一致（分数未被插桩改变）。

## 5. 需带回产物（放入 `experiments/returns/2026-09-05_M7-I1A-recheck/`）

- `eps{001,005,01,05}/diagnose_3a_*.json`（含 ot_plan_stats + u1 + per_episode）+ `*_recheck.log`；
- `eps01_iters60/diagnose_3a_*.json` + 日志；
- 异常全文（若有）。

收包后 Claude：核四项判据 → U1 逐元素确认 + π 熵/残差健康 + iters 充分 →
eps 0.1 vs 0.05 用 per_episode_c0_fused 算配对 ΔAcc 及 95% CI（点 6）→
**数值健康则固定 ε*=0.1 放行步 B（交接包⑩，不重训步 A）**；若某项异常再定位（手册 §七）。
