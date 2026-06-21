import torch

def quantize_affine(t, bits, dim_group=None):
    """Per-group symmetric affine quantization. Groups over last dim by default.
    t: tensor; quantize each row over its last dimension independently."""
    if bits >= 16:
        return t
    qmax = 2 ** (bits - 1) - 1
    # group along last dim: compute scale per (all-but-last) slice
    amax = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = amax / qmax
    q = torch.clamp(torch.round(t / scale), -qmax - 1, qmax)
    return q * scale

def quant_state_(ssm_state, bits):
    """In-place quantize the recurrent SSM state. Shape (b, nheads, headdim, d_state).
    Per-head, per-headdim-row affine over d_state."""
    if bits >= 16:
        return
    q = quantize_affine(ssm_state.float(), bits).to(ssm_state.dtype)
    ssm_state.copy_(q)

def quantize_asym(t, bits, dim_group=None):
    """Per-group asymmetric (affine with zero-point) quantization over last dim."""
    if bits >= 16:
        return t
    qmax = 2 ** bits - 1
    tmin = t.amin(dim=-1, keepdim=True)
    tmax = t.amax(dim=-1, keepdim=True)
    scale = (tmax - tmin).clamp_min(1e-8) / qmax
    zp = torch.round(-tmin / scale)
    q = torch.clamp(torch.round(t / scale) + zp, 0, qmax)
    return (q - zp) * scale

def quant_state_asym_(ssm_state, bits):
    """In-place asymmetric quantization of the recurrent SSM state."""
    if bits >= 16:
        return
    qd = quantize_asym(ssm_state.float(), bits).to(ssm_state.dtype)
    ssm_state.copy_(qd)
