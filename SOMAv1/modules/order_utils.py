# modules/order_utils.py
# Utilities for re-ordering patch tensors

from __future__ import annotations
import torch

def variable_wise_order(x: torch.Tensor) -> torch.Tensor:
    """
    Flatten patches inside each variable.
    Input : (B, N, P, L)
    Output: (B, N*P, L)
    """
    return x.reshape(x.size(0), -1, x.size(-1))


def interleaved_order(x: torch.Tensor) -> torch.Tensor:
    """
    Interleave variables and patches along the sequence dimension.
    Input : (B, N, P, L)
    Output: (B, N*P, L) after permuting
    """
    return x.permute(0, 2, 1, 3).contiguous().reshape(x.size(0), -1, x.size(-1))
