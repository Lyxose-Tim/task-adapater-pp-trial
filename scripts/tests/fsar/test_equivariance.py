"""check_text_order_equivariance 检查器自身的行为测试。

注意（R-09）：本文件的 mock 模型构造上天然等变，只验证检查器逻辑，
**不构成 O-1 结论的证据**。O-1（O-MSA 子动作轴排列等变）的实际验证在：
- scripts/tests/fsar/test_model.py::test_released_order_msa_is_permutation_equivariant
  （随机初始化的真实 ResidualAttentionBlock，CPU）
- scripts/check_order_equivariance.py（真实 CLIP 权重门控，须在真机复跑）
"""

import torch

from fsar.equivariance import check_text_order_equivariance


class _EquivariantModel:
    def eval(self):
        return self

    def encode_text_stages(self, class_indices, permutation=None):
        base = torch.arange(3 * len(class_indices) * 4).reshape(3, len(class_indices), 4).float()
        return base if permutation is None else base[list(permutation)]


def test_full_forward_equivariance_gate():
    result = check_text_order_equivariance(_EquivariantModel(), torch.tensor([0, 1]))
    assert result["passed"]
    assert result["fast_path_allowed"]
    assert result["max_abs"] == 0.0

