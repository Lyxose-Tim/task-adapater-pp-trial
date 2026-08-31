# Windows 11：SSv2-Small 数据与正式实验手册

本文所有命令都从仓库根目录 `D:\task-adapter-pp-trial` 执行。程序直接用 Decord 解码 VP9 WebM；默认不转 MP4、不抽取 JPEG。`dataset/SSv2` 下的原始分卷和标签 ZIP 是只读输入，任何成功或失败路径都不得删除、改名或覆盖它们。

## 1. 环境与解包

```powershell
conda activate task_adapter_pp
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = (Get-Location).Path

python scripts/prepare_ssv2.py extract --source dataset/SSv2
python scripts/prepare_ssv2.py build-small `
  --source dataset/SSv2 --per-class 100 --seed 916 --workers 4
```

`extract` 顺序读取 `20bn-something-something-v2-00` 和 `-01`，使用 `dataset/SSv2/extracted.incomplete` 暂存，成功后原子落到 `dataset/SSv2/extracted`。最终视频目录为 `dataset/SSv2/extracted/videos`，完成记录为 `dataset/SSv2/extracted/extraction.complete.json`。失败时保留 `.incomplete` 和错误报告，源分卷保持不变。

`build-small` 默认读取官方标签 ZIP、`data/fsar_splits/somethingcmn.json`，以及已固化的 CMN 官方精确清单 `data/fsar_splits/ssv2_cmn/{train,val,test}.list`，生成：

```text
dataset/smsm_cmn/annotations/train.txt
dataset/smsm_cmn/annotations/val.txt
dataset/smsm_cmn/annotations/test.txt
dataset/smsm_cmn/ssv2_small_provenance.json
```

正式实验必须使用上述 CMN 清单。脚本会自动发现并校验三份文件的类别顺序、每类 100 个唯一 ID，以及 ID 与官方 `train.json + validation.json` 的标签一致性。仅在研究非标准重建划分时才显式添加 `--stable-hash-fallback`；该模式不得用于论文基线。

## 2. 数据验收

```powershell
python scripts/prepare_fsar_data.py validate `
  --annotation-dir dataset/smsm_cmn/annotations `
  --path-root dataset/SSv2/extracted/videos `
  --expected-class-counts 64,12,24 `
  --min-samples-per-class 100 `
  --verify-frame-counts `
  --output outputs/data/ssv2_validation.json

python scripts/prepare_fsar_data.py audit `
  --annotation-dir dataset/smsm_cmn/annotations `
  --path-root dataset/SSv2/extracted/videos `
  --sample-count 3 --seed 916 `
  --output outputs/data/ssv2_audit.json `
  --preview-dir outputs/data/ssv2_previews

python scripts/run_fsar.py validate `
  --config configs/innovation3_ssv2_smoke.yaml
```

验收门槛：官方解包得到 220,847 个 WebM；入选标注合计 10,000 行；train/val/test 分别为 64/12/24 类和 6,400/1,200/2,400 个视频；每类 100 个样本；三组类别互斥；路径、帧数和源文件 SHA-256 全部通过。人工检查 `ssv2_previews` 中每组 3 张八帧接触表的标签与时间顺序。

如果仅少数 WebM 无法由 Decord 解码，先保留失败清单，再只用 FFmpeg 将失败文件转为 H.264 MP4；不要全量转码，且必须保留原 WebM 和替代路径映射。

## 3. 创新点 3：固定顺序与独立输出

以下 YAML 是不可变实验配方，每个阶段使用不同目录：

| 阶段 | 配置 | 输出 |
|---|---|---|
| M0 smoke | `innovation3_ssv2_smoke.yaml` | `outputs/innovation3/ssv2/smoke` |
| B0 训练与 10k 终评 | `innovation3_ssv2_b0.yaml` | `outputs/innovation3/ssv2/b0` |
| 3a 开发诊断（1k） | `innovation3_ssv2_3a_dev.yaml` | `outputs/innovation3/ssv2/3a/development` |
| 3a 最终诊断（10k） | `innovation3_ssv2_3a_final.yaml` | `outputs/innovation3/ssv2/3a/final` |
| 3b 训练与 10k 终评 | `innovation3_ssv2_3b.yaml` | `outputs/innovation3/ssv2/3b/train` |

```powershell
python scripts/run_fsar.py smoke `
  --config configs/innovation3_ssv2_smoke.yaml --device cuda:0

python scripts/run_fsar.py train `
  --config configs/innovation3_ssv2_b0.yaml --device cuda:0

python scripts/run_fsar.py diagnose `
  --config configs/innovation3_ssv2_3a_dev.yaml `
  --weights outputs/innovation3/ssv2/b0/best.pt --device cuda:0

python scripts/run_fsar.py diagnose `
  --config configs/innovation3_ssv2_3a_final.yaml `
  --weights outputs/innovation3/ssv2/b0/best.pt --device cuda:0

python scripts/run_fsar.py train `
  --config configs/innovation3_ssv2_3b.yaml --device cuda:0
```

`train` 会恢复该目录下的最佳权重并自动执行 10,000 个 test episode。B0 的 `metrics.json` 若与论文参考值 63.6 相差超过 1 个百分点，停止 3b 和创新点 1/2/4，先核对 CMN 精确视频清单、类别顺序、帧采样和路径标注。

需要对 3b 权重执行与 3a 相同的配对诊断时，复用诊断配置但给出独立输出目录：

```powershell
python scripts/run_fsar.py diagnose `
  --config configs/innovation3_ssv2_3a_dev.yaml `
  --weights outputs/innovation3/ssv2/3b/train/best.pt `
  --output outputs/innovation3/ssv2/3b/diagnosis/development --device cuda:0

python scripts/run_fsar.py diagnose `
  --config configs/innovation3_ssv2_3a_final.yaml `
  --weights outputs/innovation3/ssv2/3b/train/best.pt `
  --output outputs/innovation3/ssv2/3b/diagnosis/final --device cuda:0
```

不要在同一输出目录重跑不同配置。若需要复跑种子，必须同时复制 YAML、修改 `runtime.seed`，并用 `--output outputs/.../seed_<seed>` 指定新目录。

## 4. 创新点 1 → 2 → 4

只在创新点 3 验收后执行。各阶段配置和输出互相独立：

```powershell
# Innovation 1: balanced -> ordered -> unbalanced OT
python scripts/run_fsar.py train --config configs/innovation1_balanced.yaml --device cuda:0
python scripts/run_fsar.py train --config configs/innovation1_ordered.yaml --device cuda:0
python scripts/run_fsar.py train --config configs/innovation1_unbalanced.yaml --device cuda:0

# Innovation 2 requires corpus/classes_somethingcmn_v2.json to have passed QC.
python scripts/run_fsar.py train --config configs/innovation2_ssv2.yaml --device cuda:0

# Combined M13: innovation 1 + 2 + probability/adaptive fusion hooks.
python scripts/run_fsar.py train --config configs/innovation124_ssv2.yaml --device cuda:0
python scripts/run_fsar.py calibrate-fusion `
  --config configs/innovation124_ssv2.yaml `
  --weights outputs/innovation124/ssv2/best.pt --episodes 1000 --device cuda:0 `
  --output outputs/innovation124/ssv2/fusion_calibration
python scripts/run_fsar.py evaluate-adaptive `
  --config configs/innovation124_ssv2.yaml `
  --weights outputs/innovation124/ssv2/best.pt --episodes 10000 --device cuda:0 `
  --output outputs/innovation124/ssv2/adaptive_final
```

创新点 2 的 v2 语料生成与 QC 命令仍以根目录 `README.md` 为准；在文件缺失或类序不匹配时，`innovation2_ssv2.yaml` 和 `innovation124_ssv2.yaml` 会在启动 GPU 前失败。
