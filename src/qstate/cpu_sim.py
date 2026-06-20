"""A tiny *synthetic* SSM decoder for CPU smoke-testing the Experiment-0 pipeline.

This is NOT Mamba-2. It is a deliberately faithful caricature of the property we
care about: a fixed-size recurrent state h_t that is linearly updated and
overwritten each step, then read out to logits. It lets us validate the full
measurement harness (state perturbation -> KL vs length -> growth-exponent fit)
without a GPU, and gives a directional sanity check that the harness can detect
compounding when it exists.

On the GPU box, the real Mamba-2 model replaces this; the harness code is shared.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SyntheticSSMConfig:
    d_state: int = 256
    vocab: int = 1000
    # Spectral radius of the state-transition. >=1 makes perturbations persist /
    # grow (compounding regime); <1 makes them decay (self-correcting regime).
    spectral_radius: float = 1.0
    seed: int = 0


class SyntheticSSM:
    """h_{t+1} = A h_t + B x_t ;  logits_t = C h_t.

    A is constructed with a controlled spectral radius so we can dial the regime.
    """

    def __init__(self, cfg: SyntheticSSMConfig):
        self.cfg = cfg
        g = torch.Generator().manual_seed(cfg.seed)
        n = cfg.d_state
        # Random orthogonal-ish A scaled to target spectral radius.
        M = torch.randn(n, n, generator=g) / (n ** 0.5)
        # Symmetric part -> real eigenvalues, easy to rescale.
        A = (M + M.t()) / 2
        eigmax = torch.linalg.eigvalsh(A).abs().max().clamp_min(1e-6)
        self.A = A * (cfg.spectral_radius / eigmax)
        self.B = torch.randn(n, generator=g) / (n ** 0.5)
        self.C = torch.randn(cfg.vocab, n, generator=g) / (n ** 0.5)

    def init_state(self) -> torch.Tensor:
        return torch.zeros(self.cfg.d_state)

    def step(self, h: torch.Tensor, x_scalar: float) -> torch.Tensor:
        return self.A @ h + self.B * x_scalar

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.C @ h
