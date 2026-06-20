"""qstate: quantized recurrent state for Mamba-2 inference."""
from .quantizers import QuantConfig, quantize_state, quant_error
from .metrics import kl_divergence, top1_agreement, fit_growth_exponent
from .state_hook import StatePolicy
from .utils import set_seed, RunManifest, DEFAULT_SEED

__version__ = "0.1.0"
__all__ = [
    "QuantConfig",
    "quantize_state",
    "quant_error",
    "kl_divergence",
    "top1_agreement",
    "fit_growth_exponent",
    "StatePolicy",
    "set_seed",
    "RunManifest",
    "DEFAULT_SEED",
]
