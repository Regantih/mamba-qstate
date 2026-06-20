"""Divergence metrics between full-precision and quantized generation.

Core hypothesis test: does KL(full || quant) over the next-token distribution
grow super-linearly with generation length?
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """KL(P || Q) at one step, where P=full-precision, Q=quantized.

    logits_* : (vocab,) or (B, vocab). Reduced to a scalar mean over batch.
    """
    if logits_p.dim() == 1:
        logits_p = logits_p.unsqueeze(0)
        logits_q = logits_q.unsqueeze(0)
    logp = F.log_softmax(logits_p.float(), dim=-1)
    logq = F.log_softmax(logits_q.float(), dim=-1)
    p = logp.exp()
    kl = (p * (logp - logq)).sum(dim=-1)   # (B,)
    return kl.mean().item()


def top1_agreement(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """Fraction of positions where argmax(P) == argmax(Q)."""
    if logits_p.dim() == 1:
        logits_p = logits_p.unsqueeze(0)
        logits_q = logits_q.unsqueeze(0)
    return (logits_p.argmax(-1) == logits_q.argmax(-1)).float().mean().item()


def fit_growth_exponent(steps, kl_values) -> dict[str, float]:
    """Fit KL(t) ~ a * t^b in log-log space.

    b ~ 1  -> linear (errors stay local / additive)
    b > 1  -> super-linear (errors COMPOUND) — supports our hypothesis
    b < 1  -> sub-linear (errors self-correct / saturate)

    Returns the exponent b, prefactor a, and R^2 of the log-log fit.
    """
    import numpy as np

    t = np.asarray(steps, dtype=float)
    k = np.asarray(kl_values, dtype=float)
    mask = (t > 0) & (k > 0)
    t, k = t[mask], k[mask]
    if len(t) < 3:
        return {"exponent_b": float("nan"), "prefactor_a": float("nan"), "r2": float("nan")}
    lt, lk = np.log(t), np.log(k)
    b, loga = np.polyfit(lt, lk, 1)
    pred = b * lt + loga
    ss_res = ((lk - pred) ** 2).sum()
    ss_tot = ((lk - lk.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"exponent_b": float(b), "prefactor_a": float(np.exp(loga)), "r2": float(r2)}
