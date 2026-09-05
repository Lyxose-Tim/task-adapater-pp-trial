# 交接包⑫：步 B 前置 · OT 内容贡献 λ 探针（只推理，不训练；云端，Cursor 执行）

代号：**I1-probe** ｜ 产出：2026-09-05 ｜ 承接：Cursor → 云端 GPU ｜ 代码 `6d742a2`

## 1. 目的与里程碑

M8（创新点 1 步 B）**放行门控**。复诊⑪显示平衡 OT（λ=0）在 ε≥0.05 时 π 近均匀
（全局熵→1），故步 B 是否仍固定 ε*=0.1 需先确认：**λ>0 时 OT 是内容自适应的软阶段
分配，还是退化为纯几何软窗口（≈B0 固定窗口）**。本包用**现有 3 个步 A checkpoint
（ε=0.01/0.05/0.1，不重训）**在同一批 1000 配对 episode 上只前向扫 λ，量化内容贡献。

判据全过（尤其 ε=0.1）→ 固定 ε*=0.1 立即放行完整步 B（交接包⑩）；ε=0.1 呈 position-only
软窗口 → 停，不启动全网格，比较 ε=0.05/0.01 内容贡献后再选 ε。

## 2. 版本与环境

- 代码：`main` @ **`6d742a2`**（`ot_align` 探针数学 + `models.ot_probe` + `diagnose.probe`；
  纯 detach，隔离打分/梯度/显存）。`git pull`。
- 环境/数据/CLIP：同复诊⑪。ckpt 用步 A 三档 best：`I1A_eps001_56.32.tar`、
  `I1A_eps005_best.tar`、`I1A_eps01_best.tar`（云端应仍在；若清，从
  `experiments/returns/2026-09-03_M7-I1A/eps*/` 上传到 `dataset/checkpoints/`）。
- 验证记录：**推理型，无训练图改动**（等价复诊⑪的推理型交接）。本地 CPU 单测
  136/136（含探针 7 项：位置-only 内容无关/λ=0 均匀、条件熵随 λ↓、带状随 λ↑、
  内容 L1 有内容>0/恒定→0、成对 L1 内容>0 纯几何=0、detach、模型级 `ot_probe` 接线）
  + `diagnose.py` 编译/汇总仿真通过。**环境四坑自查**：CLIP 用 git+ 装；
  `CLIP_VIT_B16_PATH` 指向权重；config 含 `method: taskadapter`；`num_workers` 按核数。

## 3. 执行步骤（推理型，无训练）

### 3.1 首跑小规模试跑（§9.4，先确认通路）
临时把 `config_I1_probe_eps01.yaml` 的 `diagnose_episodes` 改 20 跑一次，确认 GPU 上
`ot_probe=true` 走 `probe()`、`workspace_I1_probe_eps01/ot_probe/probe_lambda_*.json`
正常出五量 + aux acc/OS，无 NaN；确认后改回 1000。

### 3.2 三档 ε 全量探针（各 1000 配对 episode，同 seed 916）
```bash
for tag in eps001 eps005 eps01; do
  TA_CONFIG=config_I1_probe_${tag}.yaml python diagnose.py 2>&1 | tee I1_probe_${tag}.log
done
```
每档对 λ∈{0,0.1,0.3,1,3} 只前向，JSON 的 `per_lambda[λ]` 含（逐 (q,c)/(q)/(c) 汇总
mean/p95/p99/max/min）：
- `row_cond_entropy`  逐帧条件分配熵/logK（∈[0,1]，λ↑ 应↓＝分配变锐）；
- `band_mass`  最近对角带质量占比（∈[0,1]，λ↑ 应↑）；
- `content_l1_full_vs_pos`  ‖π_full−π_pos‖₁（π_pos=OT(λD) 纯几何；→0＝软窗口退化）；
- `cross_class_l1` / `cross_query_l1`  plan 随类/query 内容成对 L1（**π_pos 对二者恒 0**）；
- `aux_acc_fused` / `aux_acc_sem` / `aux_OS_fused`  **仅辅助，不据此选 λ**
  （注：λ 只在推理时施加、ckpt 系 λ=0 训练，故此 acc/OS 非正式曲线，正式曲线是步 B 产物）。

## 4. 成功判据（数值健康 + 内容贡献双门槛 → 放行）

对每档 ε（判据主看 **ε=0.1**）：
- [ ] **band mass 随 λ 单调上升**（λ=0 → λ=3）；
- [ ] **中等 λ（0.3~1）下 π_full 不退化为 π_pos**：`content_l1_full_vs_pos` mean 显著
  >数值噪声（Sinkhorn fp32 确定性噪声 ~1e-6；指标性阈值 mean ≳1e-2）；
- [ ] **plan 随内容变化非噪声级**：`cross_class_l1`、`cross_query_l1` mean 显著>0
  （π_pos 恒 0 为对照；指标性阈值 mean ≳1e-3）；
- [ ] `row_cond_entropy` 随 λ 下降（辅证锐化，与 band mass 一致）。

**放行逻辑**（收包后 Claude 判读）：
- ε=0.1 四项全过 → **固定 ε*=0.1，立即放行完整步 B（交接包⑩，config_I1B_lam*，不重训步 A）**；
- ε=0.1 仅 position-only（content_l1 与跨类/跨query L1 均趋噪声、band 上升全由几何驱动）
  → **停，不启动全网格**；比较 ε=0.05/0.01 的 content_l1、cross_*_l1，若更小 ε 内容贡献
  显著则改选之，否则升级方案（回报用户）。

## 5. 预计显存 / 时长 / 费用

- 显存：~1.7 GB（视觉前向为主，OT 问题 7×3 微型）。
- 时长：~25–30 min/档（视觉前向与复诊⑪同量级 + 每 episode ~30 个微型 OT 解的小开销）；
  三档链 ~1.5h。**首跑标"待实测"**。

## 6. 需带回产物（放入 `experiments/returns/2026-09-05_M8pre-I1-probe/{eps001,eps005,eps01}/`）

- 各档 `workspace_I1_probe_*/ot_probe/probe_lambda_*.json` + `I1_probe_*.log`；
- config 快照、`nvidia-smi` 峰值、墙钟、异常全文（若有）。

收包后 Claude：核四判据（主看 ε=0.1）→ 内容贡献健康则固定 ε*=0.1 放行步 B⑩；
否则按放行逻辑比较更小 ε 或回报。ledger 记 I1-probe 行（内容贡献量），不改步 A/复诊数值。
