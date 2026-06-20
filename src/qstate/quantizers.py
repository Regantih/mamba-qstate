"""Post-training quantizers for the Mamba-2 recurrent SSM state.

The recurrent state in Mamba-2 has shape (batch, n_heads, head_dim, d_state)
and is overwritten every decoding step. We quantize it *in place* between
steps to study how aggressively it tolerates low precision.

All quantizers are symmetric/asymmetric affine and operate per a configurable
granularity (per-tensor, per-head, per-channel) so we can probe where the
error budget lives.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

Granularity = Literal["per_tensor", "per_head", "per_channel"]


@dataclass
class QuantConfig:
    bits: int = 8                      # 8 or 4 (set 16/32 to disable)
    granularity: Granularity = "per_head"
    symmetric: bool = True
    # Reduction dims define the scale-sharing group. For state shape
    # (B, H, P, N) [batch, heads, head_dim, d_state]:
    #   per_tensor  -> one scale for all of H,P,N
    #   per_head    -> one scale per head (reduce over P, N)
    #   per_channel -> one scale per (head, d_state) column (reduce over P)

    def enabled(self) -> bool:
        return self.bits < 16


def _reduce_dims(granularity: Granularity) -> tuple[int, ...]:
    # State is (B, H, P, N). Keep batch (0) always separate.
    if granularity == "per_tensor":
        return (1, 2, 3)
    if granularity == "per_head":
        return (2, 3)
    if granularity == "per_channel":
        return (2,)
    raise ValueError(f"unknown granularity {granularity}")


def quantize_state(state: torch.Tensor, cfg: QuantConfig) -> torch.Tensor:
    """Fake-quantize the recurrent state: quantize then dequantize.

    Returns a tensor of the same dtype/shape with quantization error baked in.
    This is the operation injected between decoding steps.
    """
    if not cfg.enabled():
        return state

    orig_dtype = state.dtype
    x = state.float()
    dims = _reduce_dims(cfg.granularity)
    qmax = (1 << (cfg.bits - 1)) - 1          # e.g. 127 for 8-bit, 7 for 4-bit

    if cfg.symmetric:
        amax = x.abs().amax(dim=dims, keepdim=True).clamp_min(1e-8)
        scale = amax / qmax
        q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
        deq = q * scale
    else:
        xmin = x.amin(dim=dims, keepdim=True)
        xmax = x.amax(dim=dims, keepdim=True)
        qmin = -(1 << (cfg.bits - 1))
        qmaxa = (1 << (cfg.bits - 1)) - 1
        scale = ((xmax - xmin) / (qmaxa - qmin)).clamp_min(1e-8)
        zp = torch.round(qmin - xmin / scale)
        q = torch.clamp(torch.round(x / scale) + zp, qmin, qmaxa)
        deq = (q - zp) * scale

    return deq.to(orig_dtype)


def quant_error(state: torch.Tensor, cfg: QuantConfig) -> dict[str, float]:
    """Diagnostic: relative L2 and max abs error of a single quantization."""
    deq = quantize_state(state, cfg)
    err = (deq - state).float()
    denom = state.float().norm().clamp_min(1e-8)
    return {
        "rel_l2": (err.norm() / denom).item(),
        "max_abs": err.abs().max().item(),
    }
