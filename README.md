# Task-Adapter++：创新点 3 / 1 / 2 / 4 实验框架

仓库结构对齐上游 `github.com/Jaulin-Bage/Task-Adapter-pp`（2026-08-31 重构）：

- **官方训练/评测代码在仓库根目录**（`run.py`/`models.py`/`dataset.py`/`utils.py`/`module_adapter.py`/`module_sem_adapter.py`/`config*.yaml`/`corpus/`）——我们的改进（P1/P2 修复、环境坑、可配 seed）直接就在这些根文件里。
- `fsar/`——独立、可测试的诊断与创新点**工具层**（`import` 根目录的共享编码器模块；另有独立的 `fsar.utils`/`fsar.model`）。默认配置复现固定窗口与逐元素乘积；创新功能均通过显式开关启用，不会静默改变基线。
- ZSL/持续学习轨道已迁出至独立仓库 `D:\task-adapter-zslcl`（隔离，互不引用）。

## 分支

| 分支 | 内容 |
|---|---|
| `main` | 我们的改进代码（结构同上游，改进就在根目录）＋ `fsar/` 工具层 ＋ 实验台账 |
| `official-baseline` | 纯上游源码（Jaulin-Bage/Task-Adapter-pp @ b55a59f），唯一内容，作对照参照。`git diff official-baseline main` 即我们相对官方的全部改动 |

## B0 基线状态（M2，收尾中）

SSv2-Small 5-way 1-shot（CMN 划分）自训基线。**论文 63.6 仅作参照、不作对照**（P2 修复使基线相对论文移动属预期，见 CLAUDE.md §5）。

后续对照基线 B0 = **P1/P2 修复 × torch 2.2.2**（官方未修 59.34 带 P2 bug、其分支级分数不可作参照）。3 种子(916/917/918)终测 56.92/59.68/56.70 → **mean±std = 57.77 ± 1.66**（seed917 离群，方差偏大，正补 seed 919/920 到 n=5 收紧）。相对论文 ~5.8pt 缺口经审计（`experiments/reviews/2026-08-30_B0缺口审计.md`）+ 30-epoch 探针（val 振荡非爬升）+ 官方未修亦仅 59.34，三重佐证为**不可约复现差**（非训练时长、非我方 bug）。台账见 `experiments/ledger.csv`。

## 已实现内容

- 创新点 3：修复分支命名 P1 与 `[query, class]` 朝向 P2；六种子动作排列、查询帧扰动、配对置信区间、order margin/InfoNCE；真实 CLIP O-MSA 等变性门控。
- 创新点 1：fp32、log-domain、单侧非平衡 Sinkhorn OT；硬帧边际、软阶段边际、位置先验、平衡显式分支、传输图与残差诊断。
- 创新点 2：K=2–5 的 v2 corpus schema；provider-agnostic 生成；全类文本 QC；仅 base 视频 QC；validation 配方选择；严格拒绝 test 视频；support-only 在线候选选择与 ragged K。
- 创新点 4a：旧乘积、概率乘积和 logit 凸组合集中实现；全局 α、缓存分数伪验证、平坦回退及 KIP 硬门控。
- 数据：直接用 `decord` 解码视频，不需要把所有视频膨胀成 JPEG；支持按 `[0.25,1]` 等时间区间采样。

## 环境

推荐 Conda 环境名为 `task_adapter_pp`。若本机仍保留旧名，则使用 `tsa_mlt`：

```powershell
conda activate task_adapter_pp
$env:PYTHONPATH = (Get-Location).Path
```

官方 OpenAI CLIP ViT-B/16 JIT 权重已放在：

```text
dataset/checkpoints/ViT-B-16.pt
```

权重解析优先级为 `--checkpoint`、`CLIP_CHECKPOINT`、配置文件；缺失时会在 GPU 工作前失败。

## 数据状态

| 数据集 | 本地状态 | 标注/备注 |
|---|---:|---|
| UCF101 | 13,320 videos | 70/10/21 类拆分，标注在 `dataset/UCF101/annotations/` |
| HMDB51 | 6,766 videos | 31/10/10 类拆分，标注在 `dataset/HMDB51/annotations/` |
| SSv2 | 220,847 WebM | 已无损串流解包，并按 CMN 官方精确清单构建 10,000 个 SSv2-Small 样本；原始分卷保留 |
| Kinetics | 已删除 | 2026-07-18 按 R-05 决议整体删除本地 K400 数据；CMN 清单仍在 `data/fsar_splits/kinetics_cmn/` |

SSv2 的只读源文件保护、解包、标注验收和正式实验命令见
[`docs/codex/SSV2_WINDOWS.md`](docs/codex/SSV2_WINDOWS.md)。下载入口和本地状态见
[`docs/codex/DATASETS.md`](docs/codex/DATASETS.md)。

重新扫描现有 UCF101/HMDB51 视频并生成标注：

```powershell
python scripts/prepare_available_datasets.py all --workers 12
```

脚本只探测 AVI 视频头并写 `path num_frames label`，不会全量抽帧。生成报告位于各数据集的 `annotations/preparation_report.json`。

## 运行

先做无 GPU 的路径与语料校验：

```powershell
python scripts/run_fsar.py validate --config configs/innovation3_hmdb51.yaml
python scripts/run_fsar.py validate --config configs/innovation124_ucf101.yaml
```

创新点 3 的 SSv2 正式配置互相独立，不共享输出目录：

```powershell
python scripts/run_fsar.py smoke --config configs/innovation3_ssv2_smoke.yaml --device cuda:0
python scripts/run_fsar.py train --config configs/innovation3_ssv2_b0.yaml --device cuda:0
python scripts/run_fsar.py diagnose --config configs/innovation3_ssv2_3a_dev.yaml `
  --weights outputs/innovation3/ssv2/b0/best.pt --device cuda:0
python scripts/run_fsar.py train --config configs/innovation3_ssv2_3b.yaml --device cuda:0
python scripts/check_order_equivariance.py --config configs/innovation3.yaml `
  --checkpoint dataset/checkpoints/ViT-B-16.pt --device cuda:0
```

3a 的 10,000-episode 终评和创新点 1→2→4 的完整顺序见 SSv2 运行手册。

创新点 1 消融配置：

```text
configs/innovation3.yaml          fixed-window B0
configs/innovation1_balanced.yaml OT, lambda=0, rho=inf
configs/innovation1_ordered.yaml  OT, lambda>0, rho=inf
configs/innovation1_unbalanced.yaml OT, lambda>0, finite rho
configs/innovation2_ssv2.yaml     previous row + v2 corpus
configs/innovation124_ssv2.yaml   previous innovations + adaptive fusion hooks
```

在已有 UCF101/HMDB51 上运行组合路径：

```powershell
python scripts/run_fsar.py smoke --config configs/innovation124_hmdb51.yaml --device cuda:0
python scripts/run_fsar.py calibrate-fusion --config configs/innovation124_hmdb51.yaml `
  --device cuda:0 --episodes 600
python scripts/run_fsar.py evaluate-adaptive --config configs/innovation124_ucf101.yaml `
  --device cuda:0 --episodes 100
```

`evaluate-adaptive` 只从支持视频构造伪查询；真实查询不参与 α 选择。各伪视图计算一次双分支分数，α 网格扫描为缓存张量上的纯算术。

## 创新点 2 语料

生成前先检查 1,200 个请求（100 类 × 4 个 K × 每 K 3 次），不会导入 provider 或访问网络：

```powershell
python scripts/generate_corpus_v2.py `
  --classes data/fsar_splits/somethingcmn.json `
  --dataset somethingcmn `
  --output outputs/innovation2/somethingcmn_generation_jobs.json `
  --samples-per-k 3 --dry-run
```

实际生成必须显式提供自己的 `python.module:object` provider；仓库不会隐式选择服务，也不会读取未配置的 API key：

```powershell
python scripts/generate_corpus_v2.py --emit-provider-template local_provider.py
python scripts/generate_corpus_v2.py <同上参数> `
  --provider local_provider:provider `
  --output corpus/classes_somethingcmn_v2.json
python scripts/qc_corpus_v2.py text `
  --input corpus/classes_somethingcmn_v2.json `
  --output corpus/classes_somethingcmn_v2_qc.json
```

启用 v2 时，将配置中的 `corpus_v2.enabled` 改为 `true`。加载器会要求文件存在并验证 K=2–5 覆盖；support-only 选择对同分候选优先较小 K。

## 测试

CPU 单测位于 `scripts/tests/`（`pyproject.toml` 已配置 testpaths）：

```powershell
python -m pytest -q --basetemp .pytest-run
```

关键入口：`fsar/model.py`、`fsar/ot.py`、`fsar/corpus_v2.py`、`fsar/fusion.py`、`fsar/experiment.py` 与 `fsar/cli.py`。
