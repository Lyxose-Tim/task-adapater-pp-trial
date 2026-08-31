# 数据集获取与本地状态

更新时间：2026-07-17。

## 已准备

- UCF101：`dataset/UCF101/UCF-101`，13,320 个 AVI；官方 Task-Adapter 70/10/21 类拆分已写入 `dataset/UCF101/annotations`。
- HMDB51：`dataset/HMDB51`，6,766 个 AVI；官方 Task-Adapter 31/10/10 类拆分已写入 `dataset/HMDB51/annotations`。
- OpenAI CLIP ViT-B/16：`dataset/checkpoints/ViT-B-16.pt`；SHA-256 `5806E77CD80F8B59890B7E101EABD078D9FB84E6937F9E85E4ECB61988DF416F`。

## Something-Something V2

Something-Something V2 官方页面：

- https://www.qualcomm.com/developer/software/something-something-v-2-dataset
- 官方下载说明：https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/20bn-something-something_download_instructions_-_091622-v2.pdf

官方说明的压缩下载量为 19.4 GB，满足 20 GB 阈值；下载按钮要求账户登录并接受研究许可，自动化程序不能代签。本地现已有两个视频分卷和标签 ZIP：

```text
dataset/SSv2/20bn-something-something-v2-00
dataset/SSv2/20bn-something-something-v2-01
dataset/SSv2/20bn-something-something-download-package-labels.zip
```

使用 `scripts/prepare_ssv2.py` 串流解包并生成 SSv2-Small/CMN 标注。脚本不删除或改写上述源文件；详细命令和验收标准见 [`SSV2_WINDOWS.md`](SSV2_WINDOWS.md)。

## 超过 20 GB，按要求跳过

Task-Adapter++ 使用 **Kinetics-400** 派生的 Kinetics-100/CMN few-shot 子集（100 类，每类 100 个视频，64/12/24 类拆分），不是 Kinetics-600 或 Kinetics-700。精确的 CMN 视频清单已保存到 `data/fsar_splits/kinetics_cmn/{train,val,test}.list`：

- CMN 官方清单：https://github.com/ffmpbgrnn/CMN/tree/master/kinetics-100
- FSL-Video 数据说明：https://github.com/MCG-NJU/FSL-Video#data-preparation （其原 NJU Box 外链当前已失效）

若只能获取完整数据，则完整 Kinetics-400 远超 20 GB，按约定不自动下载：

- 官方/基金会下载器：https://github.com/cvdfoundation/kinetics-dataset
- K400 train 分卷列表：https://s3.amazonaws.com/kinetics/400/train/k400_train_path.txt
- K400 validation 分卷列表：https://s3.amazonaws.com/kinetics/400/val/k400_val_path.txt
- K400 annotations：https://s3.amazonaws.com/kinetics/400/annotations/train.csv

仓库已保留论文所需的 Kinetics 100 类顺序拆分 `data/fsar_splits/kinetics.json` 和语料 `corpus/classes_kinetics.yml`；视频数据部分暂时跳过。

## 完整性检查

```powershell
python scripts/prepare_fsar_data.py validate `
  --annotation-dir dataset/HMDB51/annotations `
  --expected-class-counts 31,10,10

python scripts/prepare_fsar_data.py audit `
  --annotation-dir dataset/UCF101/annotations `
  --output outputs/data/ucf101_audit.json
```

标注解析从行尾读取 `num_frames label`，因此视频路径中即使包含空格也不会被截断。
