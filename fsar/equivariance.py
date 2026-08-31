"""Empirical permutation-equivariance gate for the released semantic encoder."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence

import torch

from fsar.order import STAGE_PERMUTATIONS


@torch.no_grad()
def check_text_order_equivariance(
    model,
    class_indices: torch.Tensor,
    *,
    permutations: Iterable[Sequence[int]] = STAGE_PERMUTATIONS,
    atol: float = 1.0e-3,
    rtol: float = 1.0e-3,
) -> Dict[str, Any]:
    """Compare true permuted prompt forwards with indexed original features.

    Passing this check is the mandatory gate for using the zero-extra-forward
    negative-permutation fast path in Innovation 3b.
    """

    model.eval()
    original = model.encode_text_stages(class_indices)
    rows = []
    passed = True
    global_max_abs = 0.0
    for permutation in permutations:
        permutation = tuple(int(value) for value in permutation)
        actual = model.encode_text_stages(class_indices, permutation=permutation)
        expected = original[list(permutation)]
        difference = (actual.float() - expected.float()).abs()
        max_abs = float(difference.max().item())
        reference = float(expected.float().abs().max().item())
        tolerance = float(atol + rtol * reference)
        row_passed = max_abs <= tolerance
        passed = passed and row_passed
        global_max_abs = max(global_max_abs, max_abs)
        rows.append(
            {
                "permutation": list(permutation),
                "max_abs": max_abs,
                "reference_max_abs": reference,
                "tolerance": tolerance,
                "passed": row_passed,
            }
        )
    return {
        "passed": passed,
        "max_abs": global_max_abs,
        "atol": float(atol),
        "rtol": float(rtol),
        "fast_path_allowed": passed,
        "permutations": rows,
    }


__all__ = ["check_text_order_equivariance"]

