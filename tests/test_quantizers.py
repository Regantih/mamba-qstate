"""Unit tests for quantizers, metrics, and state policy — CPU-only, no GPU needed."""
import math

import torch

from qstate.quantizers import QuantConfig, quantize_state, quant_error
from qstate.metrics import kl_divergence, fit_growth_exponent
from qstate.state_hook import StatePolicy


def _fake_state(b=2, h=4, p=8, n=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, h, p, n, generator=g)


def test_disabled_when_16bit():
    s = _fake_state()
    cfg = QuantConfig(bits=16)
    out = quantize_state(s, cfg)
    assert torch.equal(out, s)


def test_8bit_lower_error_than_4bit():
    s = _fake_state()
    e8 = quant_error(s, QuantConfig(bits=8, granularity="per_head"))
    e4 = quant_error(s, QuantConfig(bits=4, granularity="per_head"))
    assert e8["rel_l2"] < e4["rel_l2"], (e8, e4)
    # 8-bit per-head should be quite small.
    assert e8["rel_l2"] < 0.02


def test_finer_granularity_helps():
    s = _fake_state()
    e_tensor = quant_error(s, QuantConfig(bits=4, granularity="per_tensor"))
    e_chan = quant_error(s, QuantConfig(bits=4, granularity="per_channel"))
    assert e_chan["rel_l2"] <= e_tensor["rel_l2"] + 1e-6


def test_shape_dtype_preserved():
    s = _fake_state().half()
    out = quantize_state(s, QuantConfig(bits=8))
    assert out.shape == s.shape and out.dtype == s.dtype


def test_kl_self_is_zero():
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(3, 100, generator=g)
    assert abs(kl_divergence(logits, logits)) < 1e-6


def test_growth_exponent_recovers_known_power():
    steps = list(range(1, 200))
    kl = [0.01 * (t ** 1.5) for t in steps]   # known super-linear b=1.5
    fit = fit_growth_exponent(steps, kl)
    assert abs(fit["exponent_b"] - 1.5) < 0.05
    assert fit["r2"] > 0.999


def test_refresh_step_skips_quantization():
    s = _fake_state()
    pol = StatePolicy(quant=QuantConfig(bits=4), refresh_interval=3, noise_std=0.0)
    pol.reset()
    o1 = pol.apply(s)   # step 1 -> quantized
    o2 = pol.apply(s)   # step 2 -> quantized
    o3 = pol.apply(s)   # step 3 -> REFRESH (full precision)
    assert not torch.equal(o1, s)
    assert torch.equal(o3, s)


def test_noise_is_deterministic_with_seed():
    s = _fake_state()
    a = StatePolicy(noise_std=0.1, seed=42); a.reset()
    b = StatePolicy(noise_std=0.1, seed=42); b.reset()
    assert torch.equal(a.apply(s), b.apply(s))
