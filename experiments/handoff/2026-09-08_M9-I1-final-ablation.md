# 交接包⑮：创新点 1 收官 · M9 消融终表（10000ep）+ 主线第二种子（云端，Cursor 执行）

代号：**I1-M9** ｜ 产出：2026-09-08 ｜ 承接：Cursor → 云端 GPU ｜ 代码 `0159bf8`（诊断代码 @ `30ed45d`，M4 已验证）

## 1. 目的与里程碑

创新点 1 **收官（M9）**。用户已冻结**最终主线 ε=0.1 / λ=0.3 / ρ=None**（平衡 OT + 顺序先验；
标题定为"语义条件、顺序正则化的软阶段分配"，不平衡作扩展消融）。本包出 **10000ep 消融终表 4 行 +
主线第二训练种子**（训练方差）。**不重训 ①②③④ 的既有 ckpt**（同一 1000×10ep 训练日程，只补 10000ep 评测）。

**护栏（用户）**：④ 无论正负如实保留；**不用 10000ep test 重选 λ/ρ**；(1,1) 不上 10000ep；
微网格/I_Acc/DiD/stage_mass 全入附录（已在 dev 完成）。

## 2. 版本与环境

- 代码：`main` @ **`0159bf8`**（M9 配置；`diagnose.py` 诊断路径 @ `30ed45d`，与 M4（`fe63d69`）行为一致、
  由 P1/P2 官方单测护定）。`git pull`。本地全量单测 **163/163**。
- ckpt 依赖（云端应在 `dataset/checkpoints/`；缺则从对应 returns 上传）：
  `b0_seed916_56.32.tar`(①)、`I1A_eps01_best.tar`(②)、`I1B_lam03_best.tar`(③)、`I1C_rho1_best.tar`(④)。
- **环境四坑自查**（CLIP git+ / `CLIP_VIT_B16_PATH` / `method` / `num_workers`）。

## 3. 执行步骤

### 3.1 主线第二训练种子（唯一新训）
```bash
TA_CONFIG=config_M9_seed917_train.yaml python run.py 2>&1 | tee M9_seed917_train.log
```
best `<val>.tar` **重命名/软链为** `I1C_lam03_seed917_best.tar` 放 `dataset/checkpoints/`。

### 3.2 终表 4 行 + 第二种子 各 10000ep 诊断（复用 ckpt，不重训）
```bash
for cfg in config_M9_row1_B0 config_M9_row2_lam0 config_M9_row3_lam03 \
           config_M9_row4_lam03rho1 config_M9_row3_seed917; do
  TA_CONFIG=${cfg}.yaml python diagnose.py 2>&1 | tee ${cfg}.log
done
```
每档 `workspace_M9_*/diagnose_3a/diagnose_3a_*.json`：C0 融合 Acc±CI、OS(C0−C2mean)±CI、C0−C1、
sanity、逐 episode C0。**均 seed 916 同一 episode 流（可跨行配对）**。
> ① B0 若愿复用 M4（`returns/2026-09-01_M4-3a-final`，C0=56.78±0.40、OS=0.21±0.13，同 10000ep）可跳过 row1；
> 本包仍列 row1 配置以求**同一代码版本**下四行一致，择一即可。

## 4. 成功判据（量化，可勾选）

- [ ] seed917 训练 10ep 完成、loss 有限无 NaN、best 改稳定名 `I1C_lam03_seed917_best.tar`；
- [ ] 四行（①②③④）各出 C0 Acc±CI 与 OS±CI（10000ep，CI 预期 ~0.13–0.40）、sanity=true；
- [ ] 主线 ③ 两种子（916/917）Acc 给出训练方差；
- [ ] 终表可回溯：每行 commit＋ckpt＋returns 原始 JSON。

（收包判读由 Claude 完成）：组装 4 行终表（Acc/OS）→ 核心叙事——③ vs ① 是否精度中性且 OS 显著注入；
② 平衡 OT vs ① 固定窗口；④ ρ=1 是否如 dev 预期不超过 ③。附录挂 dev 微网格 I_Acc/DiD/stage_mass。

## 5. 预计显存 / 时长

- seed917 训练：~2h24m，14.7GB。
- 5×10000ep 诊断（含 C0–C4）：~3.7h/档（对照 M4）× 5 ≈ 18–19h，显存 ~1.6GB。
- 合计 ~21h（**强烈建议按档断点**：先训 seed917，再逐 config 跑，每档回传即可增量收包）。首跑标"待实测"。

## 6. 需带回产物（放入 `experiments/returns/2026-09-08_M9-I1-final/`）

- `seed917/`：训练 `log.txt` + `M9_seed917_train.log` + config 快照；
- `rows/`：`config_M9_row{1,2,3,4}_*` 与 `config_M9_row3_seed917` 的 `diagnose_3a_*.json` + `.log` + config 快照；
- `nvidia-smi` 峰值、墙钟、异常全文（若有）。

收包后 Claude：终表入 ledger（10000ep 终值行）+ PROGRESS；M9 验收 → **创新点 1 收官**；
随后 阶段 E（π 热图，需 plan-dump 小工具，另包）与**创新点 2**（复用 ot_align）启动。
