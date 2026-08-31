#!/usr/bin/env python3
"""Build the local smoke subset for the official Task-Adapter-pp pipeline.

CLAUDE.md §6/§8：仅从训练划分抽样（默认 8 类 × 8 段），把 WebM 解码为官方
代码要求的帧目录（img_00001.jpg 起），并生成官方 annotation 格式
（`帧目录绝对路径 帧数 类别id`）。输出布局镜像 run.py 的硬编码路径：
``<output>/smsm_cmn/annotations/{train,val,test}.txt``（三者同内容——smoke
只验证通路；正式训练不用本子集）。

用法（仓库根目录执行）::

    python scripts/make_mini_subset.py \
        --annotations dataset/smsm_cmn/annotations/train.txt \
        --video-root dataset/SSv2/extracted/videos \
        --output data/mini --classes 8 --clips 8 --seed 916
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="dataset/smsm_cmn/annotations/train.txt")
    parser.add_argument("--video-root", default="dataset/SSv2/extracted/videos")
    parser.add_argument("--output", default="data/mini")
    parser.add_argument("--classes", type=int, default=8)
    parser.add_argument("--clips", type=int, default=8)
    parser.add_argument("--seed", type=int, default=916)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--splits-out",
        default="train.txt,val.txt,test.txt",
        help="写出的 annotation 文件名（逗号分隔）。smoke 默认三份同内容；"
        "本地验证训练分两次调用：训练划分写 train.txt、测试划分写 "
        "val.txt,test.txt（官方 run.py 对 somethingcmn 的 val 与 test 同文件）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from decord import VideoReader, cpu
    except ImportError:
        print("decord is required", file=sys.stderr)
        return 1
    from PIL import Image

    annotations = Path(args.annotations)
    video_root = Path(args.video_root)
    output = Path(args.output)
    frames_root = output / "frames"
    annotation_dir = output / "smsm_cmn" / "annotations"
    frames_root.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)

    by_label: dict[int, list[str]] = defaultdict(list)
    for line in annotations.read_text().splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        by_label[int(parts[2])].append(parts[0])

    rng = random.Random(args.seed)
    eligible = sorted(label for label, clips in by_label.items() if len(clips) >= args.clips)
    if len(eligible) < args.classes:
        print(f"only {len(eligible)} classes have >= {args.clips} clips", file=sys.stderr)
        return 1
    chosen_labels = rng.sample(eligible, args.classes)

    rows = []
    for label in sorted(chosen_labels):
        for clip_name in rng.sample(sorted(by_label[label]), args.clips):
            video_path = video_root / clip_name
            if not video_path.is_file():
                print(f"missing video: {video_path}", file=sys.stderr)
                return 1
            clip_dir = frames_root / Path(clip_name).stem
            reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
            total = len(reader)
            existing = len(list(clip_dir.glob("img_*.jpg"))) if clip_dir.is_dir() else 0
            if existing != total:
                clip_dir.mkdir(parents=True, exist_ok=True)
                # 全帧导出：官方管线按 1-based 文件名 img_%05d.jpg 读帧
                for index in range(total):
                    frame = reader[index].asnumpy()
                    Image.fromarray(frame).save(
                        clip_dir / f"img_{index + 1:05d}.jpg", quality=args.jpeg_quality
                    )
            rows.append(f"{clip_dir.resolve()} {total} {label}")
            print(f"class {label}: {clip_name} -> {total} frames")

    body = "\n".join(rows) + "\n"
    for split in (name.strip() for name in args.splits_out.split(",") if name.strip()):
        (annotation_dir / split).write_text(body)
    print(
        f"subset ready: {len(rows)} clips, {args.classes} classes -> "
        f"{annotation_dir} [{args.splits_out}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
