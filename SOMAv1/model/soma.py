# model/soma.py
# Author: Your Name
# License: MIT

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

# --- local imports (ensure these modules exist) ----------------------------- #
from layers.embedding import PatchEmbedding          # patch tokenisation
from layers.encoder import Encoder                   # Mamba-based encoder stack
from modules.alignment import (
    LearnableCost,
    sinkhorn_ot_alignment,
    mmd_loss,
)
from modules.order_utils import (
    variable_wise_order,
    interleaved_order,
)
from modules.fusion import GatedFusion

# --------------------------------------------------------------------------- #
class SOMA(nn.Module):
    """
    Semantic-guided Order-aware Mamba Architecture (SOMA).

    *   Semantic Transport Mechanism (STM) aligns statistical text tokens to
        time-series patches via entropy-regularised OT.
    *   Semantic Dual-Mamba Block (SDMB) is realised through a Mamba encoder
        stack that injects the aligned semantic cues.
    *   Order-aware Scanning Strategy supports both variable-wise and
        interleaved traversal, combined via gated fusion.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()

        # ---------------- basic settings ----------------------------------- #
        self.seq_len: int = cfg["seq_len"]
        self.pred_len: int = cfg["pred_len"]
        d_model: int = cfg["d_model"]
        patch_len: int = cfg.get("patch_len", 8)
        stride: int = cfg.get("stride", patch_len)     # non-overlapping

        # ---------------- patch embedding ---------------------------------- #
        self.patch_embed = PatchEmbedding(
            embed_dim=d_model,
            patch_len=patch_len,
            stride=stride,
            dropout=cfg["dropout"],
        )
        self.num_patches: int = self.patch_embed.num_patches

        # ---------------- text backbone ------------------------------------ #
        bert_path: str | Path = cfg["bert_path"]
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path).eval()
        self.text_proj = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, d_model),
        )

        # ---------------- alignment + fusion ------------------------------- #
        self.cost_net = LearnableCost(d_model)
        self.fusion_gate = GatedFusion(d_model)

        # ---------------- encoder stack ------------------------------------ #
        self.encoder = Encoder.build_mamba_stack(cfg)

        # ---------------- prediction head ---------------------------------- #
        proj_in = cfg["d_ff"] * self.num_patches
        self.head = nn.Linear(proj_in, self.pred_len * cfg["enc_in"])

        # ---------------- misc flags --------------------------------------- #
        self.use_norm = cfg.get("use_norm", True)
        self.var_only_scan = cfg.get("var_only_scan", False)

    # --------------------------------------------------------------------- #
    @staticmethod
    def _make_prompts(stats: List[List[str]]) -> List[str]:
        """
        Convert per-variable statistics into prompt strings.
        Each element of `stats` is a list of tokens (min, max, trend, ...).
        """
        return [" ".join(tokens) for tokens in stats]

    # --------------------------------------------------------------------- #
    def forward(
        self,
        x_enc: torch.Tensor,          # (B, L, N)
        epoch: int | None = None,
        *,
        return_align_loss: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x_enc  Input series, shape (batch, seq_len, n_vars).
        Returns:
            forecast  (B, pred_len, N)
            L_align   alignment loss (optional)
        """
        b, l, n = x_enc.shape
        assert l == self.seq_len, "Sequence length mismatch"

        # 1. optional Z-score normalisation (per variable)
        if self.use_norm:
            mean = x_enc.mean(1, keepdim=True)
            std = x_enc.std(1, keepdim=True).clamp_min(1e-5)
            x_norm = (x_enc - mean) / std
        else:
            x_norm = x_enc

        # 2. statistics → prompt tokens
        stats_tokens: List[List[str]] = []
        for i in range(n):
            series_i = x_norm[:, :, i]                         # (B, L)
            mn = series_i.min(1).values
            mx = series_i.max(1).values
            med = series_i.median(1).values
            trend = torch.sign(series_i.diff(dim=1).sum(1))    # +1 or -1
            tokens = [
                f"min {mn[j]:.4f}",
                f"max {mx[j]:.4f}",
                f"median {med[j]:.4f}",
                "trend upward" if trend[j] > 0 else "trend downward",
            ]
            stats_tokens.append(tokens)
        # Convert to list of strings (one per batch*var)
        prompts = self._make_prompts(stats_tokens)

        # 3. BERT embedding → project to d_model
        ids = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).input_ids.to(x_enc.device)
        text_emb = self.bert.get_input_embeddings()(ids)       # (B*N, T_tok, 768)
        text_proj = self.text_proj(text_emb.mean(1))           # (B*N, d_model)
        text_proj = text_proj.reshape(b, n, -1)                # (B, N, d_model)

        # 4. patch embedding for time series
        ts_patch, _ = self.patch_embed(x_norm)                 # (B, N*P, d_model)

        # 5. OT alignment
        cost = self.cost_net(text_proj, ts_patch)
        align_mat = sinkhorn_ot_alignment(text_proj, ts_patch, reg=0.5)
        text_aligned = torch.bmm(align_mat, text_proj)         # (B, N*P, d_model)

        # 6. gated fusion
        fused = self.fusion_gate(ts_patch, text_aligned)

        # 7. optional interleaved scan
        if not self.var_only_scan:
            inter = interleaved_order(
                fused.reshape(b, n, self.num_patches, -1)
            )
            inter, _ = self.encoder(inter)
            fused = fused + inter  # simple residual merge

        # 8. encoder pass
        enc_out, _ = self.encoder(fused)                       # (B, N*P, d_model)

        # 9. prediction head
        flat = enc_out.reshape(b, -1)                          # (B, N*P*d_model)
        forecast = self.head(flat).reshape(b, self.pred_len, n)

        # 10. de-normalise
        if self.use_norm:
            forecast = forecast * std[:, 0, :].unsqueeze(1) + mean[:, 0, :].unsqueeze(1)

        if return_align_loss:
            # return forecast, self.lambda_align * align_loss
            return forecast
            
        return forecast, None
