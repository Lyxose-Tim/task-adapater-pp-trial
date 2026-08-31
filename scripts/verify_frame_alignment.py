#!/usr/bin/env python3
"""R-08：官方预抽帧管线 vs fsar webm 直读管线的采样协议对齐验证。

对 mini 子集（scripts/make_mini_subset.py 的产物）逐段验证：
1. **索引协议**：官方 dataset.py 评测采样公式
   ``frame_id = int(tick/2 + tick*x), tick = N/8``（0 基内容索引，文件名 +1）
   与 fsar ``sample_frame_indices(N, 8, strategy='interval')`` 是否逐段相等；
2. **像素一致性**：官方路径读到的 JPEG 帧 vs decord 直接解码的同一帧，
   平均绝对差应仅为 JPEG 压缩量级（阈值默认 mean<5/255）。

已知且不在本验证范围内的差异：train_aug 随机采样两侧策略不同
（官方 ``i*⌊N/8⌋+randint(⌊N/8⌋)`` vs fsar linspace 区间随机）——训练期
数据增广的随机性差异，不影响评测口径的可比性，已记录于 PROGRESS。

用法（仓库根目录）::

    python scripts/verify_frame_alignment.py \
        --annotations data/mini/smsm_cmn/annotations/train.txt \
        --video-root dataset/SSv2/extracted/videos \
        --output outputs/innovation3/ssv2/frame_alignment_r08.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fsar.data import sample_frame_indices  # noqa: E402


def official_eval_indices(num_frames: int, num_segments: int = 8) -> list[int]:
    """Task-Adapter-pp/dataset.py 评测分支的逐行复刻（0 基内容索引）。"""
    tick = num_frames / float(num_segments)
    return [int(tick / 2.0 + tick * x) for x in range(num_segments)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="data/mini/smsm_cmn/annotations/train.txt")
    parser.add_argument("--video-root", default="dataset/SSv2/extracted/videos")
    parser.add_argument("--output", default="outputs/innovation3/ssv2/frame_alignment_r08.json")
    parser.add_argument("--pixel-mean-threshold", type=float, default=5.0)
    args = parser.parse_args()

    from decord import VideoReader, cpu
    from PIL import Image

    rows = []
    index_mismatches = 0
    worst_mean = 0.0
    for line in Path(args.annotations).read_text().splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        frame_dir = Path(parts[0])
        num_frames = int(parts[1])
        clip_name = frame_dir.name + ".webm"

        official = official_eval_indices(num_frames)
        fsar = sample_frame_indices(num_frames, 8, strategy="interval")
        indices_equal = official == list(fsar)
        if not indices_equal:
            index_mismatches += 1

        reader = VideoReader(str(Path(args.video_root) / clip_name), ctx=cpu(0), num_threads=1)
        per_frame_mean = []
        for content_index in official:
            jpeg = np.asarray(
                Image.open(frame_dir / f"img_{content_index + 1:05d}.jpg").convert("RGB"),
                dtype=np.int16,
            )
            direct = reader[content_index].asnumpy().astype(np.int16)
            if jpeg.shape != direct.shape:
                per_frame_mean.append(255.0)
                continue
            per_frame_mean.append(float(np.abs(jpeg - direct).mean()))
        clip_mean = max(per_frame_mean) if per_frame_mean else 255.0
        worst_mean = max(worst_mean, clip_mean)
        rows.append(
            {
                "clip": clip_name,
                "num_frames": num_frames,
                "official_indices": official,
                "fsar_indices": list(fsar),
                "indices_equal": indices_equal,
                "max_frame_mean_abs_diff": clip_mean,
            }
        )

    passed = index_mismatches == 0 and worst_mean <= args.pixel_mean_threshold
    report = {
        "passed": passed,
        "clips": len(rows),
        "index_mismatches": index_mismatches,
        "worst_mean_abs_diff": worst_mean,
        "pixel_mean_threshold": args.pixel_mean_threshold,
        "note": "indices are 0-based content indices; official JPEG filenames add 1",
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"clips={len(rows)} index_mismatches={index_mismatches} "
        f"worst_mean_abs_diff={worst_mean:.3f} passed={passed}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
