"""Sanity check: prove the Exp-0 harness can DETECT compounding when present.

We sweep the synthetic SSM's spectral radius. With rho>1 (unstable transition),
a single early perturbation should grow geometrically => super-linear KL. This
confirms the growth-exponent estimator is not biased toward 'sub-linear'.

This is a methods-validation script, not a result about Mamba-2 itself.
"""
from __future__ import annotations

import torch

from qstate.cpu_sim import SyntheticSSM, SyntheticSSMConfig
from qstate.metrics import kl_divergence, fit_growth_exponent
from qstate.utils import set_seed


def measure(rho: float, n_steps=400, log_every=8, seed=1337, single_kick=True):
    set_seed(seed)
    ssm = SyntheticSSM(SyntheticSSMConfig(spectral_radius=rho, seed=seed))
    rng = torch.Generator().manual_seed(seed)
    h_full = ssm.init_state()
    h_pert = ssm.init_state()
    inputs = torch.randn(n_steps, generator=rng).tolist()
    steps, kls = [], []
    for t in range(n_steps):
        x = inputs[t]
        h_full = ssm.step(h_full, x)
        h_pert = ssm.step(h_pert, x)
        # One-time kick at t=0 to study propagation of a single error.
        if single_kick and t == 0:
            rms = h_pert.pow(2).mean().sqrt().clamp_min(1e-8)
            h_pert = h_pert + 0.05 * rms * torch.randn(h_pert.shape, generator=rng)
        if (t + 1) % log_every == 0:
            steps.append(t + 1)
            kls.append(kl_divergence(ssm.logits(h_full), ssm.logits(h_pert)))
    return fit_growth_exponent(steps, kls)


if __name__ == "__main__":
    print(f"{'spectral_radius':>16} {'b':>8} {'R^2':>8}  interpretation")
    for rho in [0.95, 1.0, 1.02, 1.05]:
        fit = measure(rho)
        b = fit["exponent_b"]
        interp = "COMPOUNDING (super-linear)" if b > 1.15 else (
            "linear" if b > 0.85 else "decaying")
        print(f"{rho:>16.2f} {b:>8.3f} {fit['r2']:>8.3f}  {interp}")
    print("\nIf rho>1 yields b>1, the harness correctly detects compounding.")
