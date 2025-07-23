# modules/alignment.py
# Author: jinkz

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableCost(nn.Module):
    """
    Learnable cost function used in entropy-regularised OT.
    Given embeddings X (B, N, d) and Y (B, M, d),
    the network outputs a cost matrix (B, N, M).
    """

    def __init__(self, dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        m = y.size(1)
        # Broadcast to (B, N, M, d)
        x_rep = x.unsqueeze(2).expand(-1, -1, m, -1)
        y_rep = y.unsqueeze(1).expand(-1, n, -1, -1)
        feat = torch.cat([x_rep, y_rep, (x_rep - y_rep).abs()], dim=-1)
        return self.net(feat).squeeze(-1)  # (B, N, M)


# ---------- Sinkhorn utilities ---------- #
def sinkhorn_knopp(cost: torch.Tensor,
                   reg: float = 0.1,
                   iters: int = 50) -> torch.Tensor:
    """
    Differentiable Sinkhorn iteration for entropy-regularised OT.

    Args:
        cost:  (B, N, M) pair-wise cost matrix.
        reg:   Regularisation strength λ.
        iters: Number of Sinkhorn iterations.

    Returns:
        Transport matrix  (B, N, M)
    """
    b, n, m = cost.shape
    k = torch.exp(-cost / reg)                     # kernel matrix
    u = torch.ones(b, n, device=cost.device) / n
    v = torch.ones(b, m, device=cost.device) / m

    for _ in range(iters):
        u = 1.0 / (k @ v.unsqueeze(-1)).squeeze(-1).clamp_min(1e-9)
        v = 1.0 / (k.transpose(1, 2) @ u.unsqueeze(-1)).squeeze(-1).clamp_min(1e-9)

    return u.unsqueeze(-1) * k * v.unsqueeze(1)


def sinkhorn_ot_alignment(x_src: torch.Tensor,
                          x_tgt: torch.Tensor,
                          reg: float = 0.1,
                          iters: int = 50) -> torch.Tensor:
    """
    Compute soft alignment matrix between x_src and x_tgt.

    Args:
        x_src: (B, N, d) source embeddings.
        x_tgt: (B, M, d) target embeddings.

    Returns:
        Transport matrix  (B, N, M)
    """
    x_src = F.normalize(x_src, dim=-1)
    x_tgt = F.normalize(x_tgt, dim=-1)
    cosine = x_src @ x_tgt.transpose(1, 2)        # similarity
    cost = 1.0 - cosine                           # convert to cost
    return sinkhorn_knopp(cost, reg, iters)


# ---------- MMD & InfoNCE (optional) ---------- #
def gaussian_kernel(x: torch.Tensor,
                    y: torch.Tensor,
                    sigma: float = 1.0) -> torch.Tensor:
    diff = x.unsqueeze(2) - y.unsqueeze(1)  # (B, N, M, d)
    dist2 = diff.pow(2).sum(-1)
    return torch.exp(-dist2 / (2.0 * sigma ** 2))


def mmd_loss(x: torch.Tensor,
             y: torch.Tensor,
             sigma: float = 1.0) -> torch.Tensor:
    k_xx = gaussian_kernel(x, x, sigma).mean(dim=(1, 2))
    k_yy = gaussian_kernel(y, y, sigma).mean(dim=(1, 2))
    k_xy = gaussian_kernel(x, y, sigma).mean(dim=(1, 2))
    return (k_xx + k_yy - 2 * k_xy).mean()


def info_nce_loss(z1: torch.Tensor,
                  z2: torch.Tensor,
                  temperature: float = 0.1) -> torch.Tensor:
    """
    Flatten (B, P, d) -> (B*P, d) then compute InfoNCE loss.
    """
    b, p, d = z1.shape
    logits = (z1.reshape(b * p, d) @ z2.reshape(b * p, d).T) / temperature
    labels = torch.arange(b * p, device=z1.device)
    return F.cross_entropy(logits, labels)
