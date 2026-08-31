# 交接包③：B0 多种子定标 + 缺口探针（云端，Cursor 执行）

代号：**B0-multiseed** ｜ 产出：2026-08-30 ｜ 承接：Cursor → 云端 GPU

## 1. 目的与里程碑

M2 收尾。回答两问，然后锁定 B0：
1. **B0 中心±散度**：现有单跑（seed 916, 56.92±0.41）不足以定标；补 seed 917/918 两跑，B0 = 3 种子 mean±std。
2. **缺口探针（欠训假设）**：审计（`experiments/reviews/2026-08-30_B0缺口审计.md`）定位相对论文 63.6 的 ~4pt 缺口最可能来自训练时长未钉死。跑一次 30-epoch 变体，看 val 轨迹是否继续爬升。

三跑均为 **P1/P2 修复 × torch 2.2.2**（B0 家族）。

## 2. 版本与环境（沿用上次 B0 云端，几乎零准备）

- 代码：`git pull` 到 `main` @ **`e765bb6`**（训练相关改动为 `e28b26d`：seed 由 config 可配）。若云端是 clone：`git fetch && git checkout e765bb6`；若是 tar 同步：仅需更新 `Task-Adapter-pp/{run.py,config_b0*.yaml}`。
- 环境：与 2026-08-30_B0_torch222 / official 同一环境（torch 2.2.2+cu121、CLIP JIT 已装、`CLIP_VIT_B16_PATH` 已设、数据 `data/full/` 已抽帧 6400/2400/2400）。**无需重装、无需重抽帧。**
- 四坑：均已在代码内修复；仅确认 `CLIP_VIT_B16_PATH` 仍指向 `dataset/checkpoints/ViT-B-16.pt`。

## 3. 执行命令（三跑，可串行或并行按显存）

单跑显存 ~14.7GB，24GB 卡一次一跑；多卡可并行。三配置已在仓内备好：

```bash
cd Task-Adapter-pp
# 多种子（各 10 epoch，约 3h/跑）
TA_CONFIG=config_b0_seed917.yaml python run.py 2>&1 | tee b0_s917_console.log
TA_CONFIG=config_b0_seed918.yaml python run.py 2>&1 | tee b0_s918_console.log
# 缺口探针（30 epoch，约 3× 训练时长 + 1× 终测，约 7–8h）
TA_CONFIG=config_b0_long.yaml   python run.py 2>&1 | tee b0_long_console.log
```

- 日志/ckpt 落在各自 `workspace_b0_s917|s918|long/.../<时间戳>/log.txt`（work_dir 已隔离，不互相覆盖，也不动原 B0）。
- 断点续跑：改对应 config 的 `load_weights/checkpoint/start_epoch`（同上次 B0 协议）。
- 显存采样：`nvidia-smi --query-gpu=memory.used --format=csv -l 30 > nvidia_<代号>.csv`。

## 4. 成功判据（逐项勾选）

- [ ] 三跑各自 10/10/30 epoch 完成、loss 有限单调下降、无 NaN/Inf；
- [ ] 各跑终测 10000 episode 完整出数；
- [ ] **多种子**：s917/s918 终测 Acc 落在 B0 邻域（大致 56–58 区间即符合预期）；
- [ ] **缺口探针**：`config_b0_long` 逐 epoch val（epoch 9–29 共 21 点）出齐 —— 关键是 val 曲线形态，不是单点。

## 5. 判读规则（Claude 收包后据此定 B0 / M2）

- **多种子**：B0 = mean±std over {916:56.92, 917, 918}。若 std 小（≲0.5pt）→ 基线稳定；若大 → 训练方差高，需在论文报告方差。
- **缺口探针**：
  - val 随 epoch 明显继续爬升、30ep 终测显著高于 10ep（趋近 60+）→ **缺口=欠训**：改 B0 协议为延长版，重跑多种子（新交接包），M2 待定标再议。
  - val 早早平台（10–15 ep 后不再升）、30ep 终测≈10ep → **缺口不可约**：按 §5 以自训 B0（10ep 协议）定论 M2，~4pt 记为复现差。

## 6. 需带回产物（放入 `experiments/returns/2026-08-30_B0-multiseed/`）

按三跑分子目录 `s917/ s918/ long/`，各含：
1. `log.txt` 全文 + `b0_<代号>_console.log`；
2. best ckpt `<val_acc>.tar`（long 跑必带——若定为新 B0 协议要用）；
3. `nvidia_<代号>.csv` 显存峰值 + 训练/终测墙钟；
4. 实际 config 快照；
5. 异常全文（若有）。

收包后 Claude：解析三跑 → 定 B0（mean±std 或延长协议）→ 写 `experiments/ledger.csv` B0 定义行 → M2 验收结论 → 触发 M3。
