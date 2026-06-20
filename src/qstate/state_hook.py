"""Hooks to intercept and modify the Mamba-2 recurrent state during decoding.

Mamba-2 (`mamba_ssm`) keeps each layer's recurrent state in
`inference_params.key_value_memory_dict[layer_idx] = (conv_state, ssm_state)`.
`ssm_state` has shape (batch, nheads, headdim, d_state) and is updated *in place*
by `Mamba2.step(hidden_states, conv_state, ssm_state)` during single-token
decoding.

Our strategy: monkeypatch each `Mamba2` layer's `step` so that AFTER the original
update we apply a `StatePolicy` to the `ssm_state` tensor in place (noise /
quantization / refresh). The policy logic lives in `StatePolicy.apply()` and is
unit-tested on CPU; only the attach point below needs the real model.

Import-safe on CPU (no mamba_ssm required): `StatePolicy` works standalone;
`attach_state_policy` only touches model internals when actually called.
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
        # 1-indexed "every k steps" semantics.
        if self.refresh_interval <= 0:
            return False
        return (self._step % self.refresh_interval) == 0

    def apply(self, state: torch.Tensor) -> torch.Tensor:
        """Apply the policy to a freshly-updated recurrent state tensor.

        Order: increment step -> refresh step returns state untouched (full
        precision) -> else inject noise (if any), then quantize (if any).
        Returns the (possibly modified) state to write back.
        """
        self._step += 1

        if self._is_refresh_step():
            return state

        out = state
        if self.noise_std > 0:
            if self._gen is None:
                self._gen = torch.Generator(device=state.device)
                self._gen.manual_seed(self.seed)
            rms = state.float().pow(2).mean().sqrt().clamp_min(1e-8)
            noise = torch.randn(
                state.shape, generator=self._gen, device=state.device, dtype=torch.float32
            )
            out = (state.float() + self.noise_std * rms * noise).to(state.dtype)

        if self.quant is not None and self.quant.enabled():
            out = quantize_state(out, self.quant)

        return out


# --------------------------------------------------------------------------- #
# Live attach point for real Mamba-2 (mamba_ssm). GPU box only.
# --------------------------------------------------------------------------- #
class _PolicyHandle:
    """Handle returned by attach_state_policy: lets caller reset() per generation
    and remove() to restore original step methods."""

    def __init__(self, policies, restorers):
        self._policies = policies      # list[StatePolicy] (per patched layer)
        self._restorers = restorers    # list[callable] to undo monkeypatch

    def reset(self) -> None:
        for p in self._policies:
            p.reset()

    def remove(self) -> None:
        for r in self._restorers:
            r()
        self._restorers.clear()


def _iter_mamba2_layers(model):
    """Yield (layer_idx, mixer_module) for every Mamba2 mixer in the model.

    Works with mamba_ssm's MambaLMHeadModel: model.backbone.layers[i].mixer.
    Falls back to scanning modules that expose a `step` and `layer_idx`.
    """
    try:
        from mamba_ssm.modules.mamba2 import Mamba2  # noqa: F401
    except Exception:
        Mamba2 = None  # type: ignore

    found = False
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "layers"):
        for i, block in enumerate(backbone.layers):
            mixer = getattr(block, "mixer", None)
            if mixer is not None and hasattr(mixer, "step"):
                found = True
                yield getattr(mixer, "layer_idx", i), mixer
    if found:
        return
    # Fallback: scan all modules.
    for name, mod in model.named_modules():
        if hasattr(mod, "step") and hasattr(mod, "layer_idx"):
            yield mod.layer_idx, mod


def attach_state_policy(model, policy_factory, layer_indices=None):
    """Patch each Mamba2 layer's `step` to post-process ssm_state via a StatePolicy.

    Args:
        model: a loaded mamba_ssm MambaLMHeadModel.
        policy_factory: callable() -> StatePolicy. Called once per patched layer so
            each layer gets an independent step counter / RNG. (Pass a lambda that
            returns a fresh StatePolicy with your desired quant/refresh/noise.)
        layer_indices: optional iterable of layer indices to patch; None = all.

    Returns:
        _PolicyHandle with .reset() (call before each generation) and .remove().

    The patched step calls the original step (which updates ssm_state in place),
    then overwrites ssm_state's contents with policy.apply(ssm_state) via copy_,
    so the in-place cache tensor carries the modified state into the next step.
    """
    want = set(layer_indices) if layer_indices is not None else None
    policies, restorers = [], []

    for layer_idx, mixer in _iter_mamba2_layers(model):
        if want is not None and layer_idx not in want:
            continue
        policy = policy_factory()
        policies.append(policy)

        orig_step = mixer.step

        def make_patched(orig_step, policy):
            def patched_step(hidden_states, conv_state, ssm_state, *args, **kwargs):
                out = orig_step(hidden_states, conv_state, ssm_state, *args, **kwargs)
                # ssm_state was updated in place by orig_step; apply policy and
                # write back into the same cache tensor.
                with torch.no_grad():
                    new_state = policy.apply(ssm_state)
                    if new_state is not ssm_state:
                        ssm_state.copy_(new_state)
                return out
            return patched_step

        mixer.step = make_patched(orig_step, policy)

        def make_restorer(mixer=mixer, orig_step=orig_step):
            def restore():
                mixer.step = orig_step
            return restore

        restorers.append(make_restorer())

    if not policies:
        raise RuntimeError(
            "attach_state_policy found no Mamba2 layers with a `step` method. "
            "Is this a mamba_ssm MambaLMHeadModel?"
        )

    return _PolicyHandle(policies, restorers)
