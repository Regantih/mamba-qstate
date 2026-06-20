"""Reproducibility and run-manifest utilities."""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_SEED = 1337


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed Python, NumPy, and Torch (incl. CUDA) for reproducible runs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Determinism (may slow kernels; acceptable for eval).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu"


def _pkg_versions() -> dict[str, str]:
    out: dict[str, str] = {"python": sys.version.split()[0]}
    for pkg in ("torch", "transformers", "numpy", "mamba_ssm"):
        try:
            mod = __import__(pkg)
            out[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[pkg] = "not-installed"
    return out


@dataclass
class RunManifest:
    """Captures everything needed to reproduce a run."""

    run_name: str
    seed: int = DEFAULT_SEED
    config: dict[str, Any] = field(default_factory=dict)
    git_sha: str = field(default_factory=_git_sha)
    gpu: str = field(default_factory=_gpu_name)
    packages: dict[str, str] = field(default_factory=_pkg_versions)
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def save(self, out_dir: str | Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "manifest.json"
        path.write_text(json.dumps(asdict(self), indent=2))
        return path
