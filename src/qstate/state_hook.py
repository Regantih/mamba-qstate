"""Hooks to intercept and modify the Mamba-2 recurrent state during decoding.

Mamba-2's official `mamba_ssm` package keeps the recurrent state in an
`InferenceParams.ssm_states[layer_idx]` tensor that is updated in place each
step. We wrap the per-step decode so that, after the state is updated, we
optionally:
  1. inject calibrated noise (Experiment 0), and/or
  2. fake-quantize it (8/4-bit), unless this step is a full-precision refresh.

This module is written to be import-safe on CPU (no mamba_ssm required) so the
logic and tests run in the sandbox; the live hooks attach on the GPU box.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from .quantizers import QuantConfig, quantize_state


@dataclass
class StatePolicy:
    """Controls what happens to the recurrent state each decode step."""

    quant: Optional[QuantConfig] = None      # None -> no quantization
    refresh_interval: int = 0                # k: every k steps keep full precision; 0 = never refresh
    noise_std: float = 0.0                   # relative Gaussian noise (Exp 0); 0 = off
    seed: int = 1337
    _step: int = field(default=0, init=False, repr=False)
    _gen: Optional[torch.Generator] = field(default=None, init=False, repr=False)

    def reset(self) -> None:
        self._step = 0
        self._gen = None

    def _is_refresh_step(self) -> bool:
        # Step counting is 1-indexed for the "every k steps" semantics.
        if self.refresh_interval <= 0:
            return False
        return (self._step % self.refresh_interval) == 0

    def apply(self, state: torch.Tensor) -> torch.Tensor:
        """Apply the policy to a freshly-updated recurrent state tensor.

        Returns the (possibly modified) state to write back. Order:
          - increment step
          - if refresh step: return state untouched (full precision)
          - else: inject noise (if any), then quantize (if any)
        """
        self._step += 1

        if self._is_refresh_step():
            return state

        out = state
        if self.noise_std > 0:
            if self._gen is None:
                self._gen = torch.Generator(device=state.device)
                self._gen.manual_seed(self.seed)
            # Relative noise: scale by per-state RMS so std is dimensionless.
            rms = state.float().pow(2).mean().sqrt().clamp_min(1e-8)
            noise = torch.randn(
                state.shape, generator=self._gen, device=state.device, dtype=torch.float32
            )
            out = (state.float() + self.noise_std * rms * noise).to(state.dtype)

        if self.quant is not None and self.quant.enabled():
            out = quantize_state(out, self.quant)

        return out


def attach_state_policy(inference_params, policy: StatePolicy, layer_indices=None):
    """Wrap an mamba_ssm InferenceParams so ssm_states get policy-processed.

    Call once before generation. `layer_indices=None` applies to all layers.
    Implementation note: the live wrapping monkeypatches the per-layer step on
    the GPU box; kept as a thin adapter so the policy logic stays testable.
    """
    raise NotImplementedError(
        "attach_state_policy is wired on the GPU box where mamba_ssm is installed; "
        "StatePolicy.apply() carries the testable logic."
    )
