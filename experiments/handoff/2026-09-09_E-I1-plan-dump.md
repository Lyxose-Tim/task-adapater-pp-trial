# 交接包⑯：创新点 1 · 阶段 E · π 热图数据导出（只推理 plan_dump，云端，Cursor 执行）

代号：**I1-E-dump** ｜ 产出：2026-09-09 ｜ 承接：Cursor → 云端 GPU ｜ 代码 `39cd82f`（主线 tag `innov1-final`）

## 1. 目的与里程碑

阶段 E（M9 验收口径的论文机制图）。**只推理**导出**固定 12 个 episode**（seed 916，内容盲、非挑样）的
OT 内部量，供本地**离线绘图**（绘图脚本只读离线 json/npz，不再前向）。主图三组：
**A 固定窗口(B0) → B 纯内容 OT(λ=0) → C 内容+顺序先验 OT(λ=0.3)**；ρ=1 入附录（stage-mass relaxation）。
**不改任何训练逻辑**（主线已冻 `innov1-final`）。

## 2. 版本与环境

- 代码：`main` @ **`39cd82f`**（`plan_dump` + `ot_stage_dump` + `ot_dump` + `dataset._PATH_RECORDER`）。`git pull`。
- 验证：本地 CPU 单测 **164/164**（含 ot_stage_dump 形状/cost=content+λD/plan 一致）。plan_dump 强制
  `num_workers=0`（路径溯源需主进程）。**环境四坑自查**。
- ckpt 依赖（云端应在 `dataset/checkpoints/`）：`b0_seed916_56.32.tar`、`I1A_eps01_best.tar`、
  `I1B_lam03_best.tar`、`I1C_lam03_seed917_best.tar`（M9 产）、`I1C_rho1_best.tar`。

## 3. 执行步骤（只推理，各 12 episode，秒级/分钟级）

### 3.0 首档验通路
先跑 `config_E_dump_lam03.yaml`，确认 `workspace_E_dump_lam03/plan_dump/plan_dump_*.json`（含
episodes[].label_idx/y_query/pred/vis/sem/fused/query_meta）与 `*.npz`（plan[E,NQ,C,T,K]/cost/mass/D[T,K]）
正常、query_meta 的 path/frame_id 齐全；无误再跑其余四档。

### 3.1 五档导出
```bash
for cfg in config_E_dump_B0 config_E_dump_lam0 config_E_dump_lam03 \
           config_E_dump_lam03_s917 config_E_dump_lam03_rho1; do
  TA_CONFIG=${cfg}.yaml python diagnose.py 2>&1 | tee ${cfg}.log
done
```
- B0 档：npz 存 `window_mask[T,3]`（式(16) 固定窗口），json 存 B0 的 pred/GT/分数/溯源。
- 四个 OT 档：npz 存 cost/plan/mass/D。**五档均 seed 916 同一 12 episode 流**（可跨档对齐同一批视频）。

## 4. 成功判据（可勾选）

- [ ] 五档各出 `plan_dump_*.json` + `*.npz`；每档 12 episode，GT/pred/vis/sem/fused 齐全；
- [ ] OT 档 npz：plan/cost 形状 [12,NQ,C,T,K]、mass [12,NQ,C,K]、D [T,K]；B0 档 window_mask[T,3]；
- [ ] `query_meta` 每 query 的 path/frame_id/num_frames/label 齐全且非空（溯源成功）；
- [ ] 五档 ckpt_md5 记录在 json（可回溯）。

（绘图与热图验收由 Claude 本地完成，读离线文件不前向）：三组主图 A/B/C；ρ=1 附录（stage-mass relaxation
＞平衡档、且不超过主线）；热图验收——λ=0 显 query/class 条件化差异（非全均匀）、λ=0.3 更带状但不退化为
所有样本一致的固定模板；seed917 同 12 episode 轻量机制复核；并冻结 λ-Acc/λ-OS 曲线、截断图、主图图注、
一页创新点 1 结论。若前 12 episode 的 pretend 类 <3，Claude 按内容盲规则（扩 episode 索引）透明调整后重导。

## 5. 预计显存 / 时长

- 每档 12 episode 只推理：~数分钟（视觉前向为主），显存 ~1.6GB；五档合计 <1h。**首跑标"待实测"**。

## 6. 需带回产物（放入 `experiments/returns/2026-09-09_E-I1-plan-dump/`）

- 五档 `workspace_E_dump_*/plan_dump/plan_dump_*.{json,npz}` + `.log`；异常全文（若有）。
  （本地已有 SSv2 帧，缩略图由本地绘图脚本按 query_meta 的 path/frame_id 读取，无需回传帧。）

收包后 Claude：本地绘图 → 阶段 E 图/图注/一页结论冻结 → **创新点 1 完全收官** → 转创新点 2。
