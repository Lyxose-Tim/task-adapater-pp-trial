"""创新点 1 阶段 D·时序鲁棒截断采样单测（手册 §四）。

核心保证：sample_window=None 逐位复现官方评测采样（B0/现有评测零漂移）；
三截断窗（掐头/去尾/收缩）落在预期时间区间、1-based、单调、界内；边界重复占比正确。
纯函数、CPU 秒级。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from dataset import sample_frame_ids, truncation_repeat_fraction  # noqa: E402

HEAD, TAIL, SHRINK = (0.25, 1.0), (0.0, 0.75), (0.125, 0.875)


@pytest.mark.parametrize("nf", [8, 9, 15, 30, 47, 64, 100])
def test_window_none_reproduces_official(nf):
    # 官方 else 分支：tick=nf/8, frame=int(tick/2+tick*x)+1 —— 必须逐位一致（评测口径不变）
    tick = nf / 8.0
    official = np.array([int(tick / 2.0 + tick * x) for x in range(8)]) + 1
    got = sample_frame_ids(nf, 8, None)
    assert np.array_equal(got, official), (nf, got.tolist(), official.tolist())


@pytest.mark.parametrize("win", [None, HEAD, TAIL, SHRINK])
@pytest.mark.parametrize("nf", [8, 15, 30, 64])
def test_frame_ids_1based_monotone_in_range(win, nf):
    fid = sample_frame_ids(nf, 8, win)
    assert fid.shape == (8,)
    assert fid.min() >= 1 and fid.max() <= nf                 # 1-based、界内（clip 保护）
    assert np.all(fid[1:] >= fid[:-1])                        # 单调非降


def test_truncation_windows_target_expected_temporal_regions():
    nf = 64
    head = sample_frame_ids(nf, 8, HEAD)                      # 丢前 25% → 从 ~0.25nf 起
    tail = sample_frame_ids(nf, 8, TAIL)                      # 丢后 25% → 至 ~0.75nf 止
    shrink = sample_frame_ids(nf, 8, SHRINK)                  # 取中间 75%
    assert head.min() > tail.min()                            # 掐头起点更靠后
    assert tail.max() < head.max()                            # 去尾终点更靠前
    assert tail.min() < shrink.min() and shrink.max() < head.max()   # 收缩居中
    assert head.min() >= int(0.25 * nf)                       # 掐头确实丢掉前 25%
    assert tail.max() <= int(0.75 * nf) + 1                   # 去尾确实丢掉后 25%


def test_boundary_repeat_fraction():
    # 有效帧数 num_frames·(hi−lo) < num_segments 的视频会重复采样
    assert truncation_repeat_fraction([["a", "30", "0"]], 8, None) == 0.0   # 全片不重复
    # nf=8: shrink 有效 8·0.75=6 <8 → 重复; nf=64: 64·0.75=48 ≥8 → 不重复
    vids = [["a", "8", "0"], ["b", "64", "0"]]
    assert truncation_repeat_fraction(vids, 8, SHRINK) == 0.5
    # 短视频掐头：nf=20 有效 20·0.75=15 ≥8 不重复; nf=9 有效 6.75<8 重复
    vids2 = [["a", "20", "0"], ["b", "9", "0"], ["c", "10", "0"]]
    assert truncation_repeat_fraction(vids2, 8, HEAD) == pytest.approx(2 / 3)  # 9,10 短


def test_small_video_repeats_but_stays_in_range():
    # nf=6（<8）：即便全片也会因 int 截断出现重复，但必须仍 1-based 且 ≤nf
    fid = sample_frame_ids(6, 8, None)
    assert fid.min() >= 1 and fid.max() <= 6
    assert len(np.unique(fid)) < 8                            # 确有重复
