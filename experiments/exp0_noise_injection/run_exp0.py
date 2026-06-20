"""Experiment 0 — does recurrent-state perturbation error compound with length?

Two backends:
  * device: cuda     -> real Mamba-2 via mamba_ssm (run on the GPU box)
  * device: cpu_sim  -> synthetic SSM (sandbox smoke test + directional signal)

Output: results/exp0/<run_name>/ with per-condition KL-vs-step curves, the
fitted growth exponent b, and a manifest. b>1 supports the compounding hypothesis.

Usage:
  python run_exp0.py --config ../../configs/exp0_default.yaml
  python run_exp0.py --config ../../configs/exp0_default.yaml --device cpu_sim
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from qstate.metrics import kl_divergence, fit_growth_exponent
from qstate.utils import RunManifest, set_seed


# --------------------------------------------------------------------------- #
# CPU-sim backend: synthetic SSM, fully self-contained.
# --------------------------------------------------------------------------- #
def run_cpu_sim(cfg: dict) -> dict:
    from qstate.cpu_sim import SyntheticSSM, SyntheticSSMConfig

    set_seed(cfg["seed"])
    n_steps = cfg["max_new_tokens"]
    log_every = cfg["log_every"]
    n_prompts = cfg["num_prompts"]
    # Use a near-critical transition so the harness must actually measure the
    # regime rather than having it baked to one answer.
    ssm = SyntheticSSM(SyntheticSSMConfig(spectral_radius=1.0, seed=cfg["seed"]))

    conditions = []
    for noise in cfg["noise_levels"]:
        for refresh in cfg["refresh_intervals"]:
            conditions.append((noise, refresh))

    results = {}
    rng = torch.Generator().manual_seed(cfg["seed"])

    for noise, refresh in conditions:
        # Average KL(step) over prompts.
        kl_accum = {}
        for p in range(n_prompts):
            inputs = torch.randn(n_steps, generator=rng).tolist()
            h_full = ssm.init_state()
            h_pert = ssm.init_state()
            nstep = 0
            for t in range(n_steps):
                x = inputs[t]
                h_full = ssm.step(h_full, x)
                h_pert = ssm.step(h_pert, x)
                nstep += 1
                # Apply perturbation to the *perturbed* trajectory's state,
                # unless this is a refresh step (state reset to full precision).
                is_refresh = refresh > 0 and (nstep % refresh == 0)
                if is_refresh:
                    h_pert = h_full.clone()
                elif noise > 0:
                    rms = h_pert.pow(2).mean().sqrt().clamp_min(1e-8)
                    h_pert = h_pert + noise * rms * torch.randn(
                        h_pert.shape, generator=rng
                    )
                if (t + 1) % log_every == 0:
                    kl = kl_divergence(ssm.logits(h_full), ssm.logits(h_pert))
                    kl_accum.setdefault(t + 1, []).append(kl)
        steps = sorted(kl_accum)
        kl_mean = [sum(kl_accum[s]) / len(kl_accum[s]) for s in steps]
        fit = fit_growth_exponent(steps, kl_mean)
        results[f"noise{noise}_refresh{refresh}"] = {
            "steps": steps,
            "kl_mean": kl_mean,
            "fit": fit,
        }
    return results


# --------------------------------------------------------------------------- #
# CUDA backend: real Mamba-2. Skeleton wired for the GPU box.
# --------------------------------------------------------------------------- #
def run_cuda(cfg: dict) -> dict:
    """Real Mamba-2 path. Requires mamba_ssm + a checkpoint on a CUDA box.

    Outline (kept explicit so the GPU run is turnkey):
      1. Load model + tokenizer (cfg['model']).
      2. For each prompt: run full-precision generation, caching per-step logits
         AND the recurrent ssm_states.
      3. Re-run with a StatePolicy (noise/quant/refresh) attached via
         qstate.state_hook, caching per-step logits.
      4. Compute KL(full || policy) per step; average over prompts.
      5. Fit growth exponent; dump curves.
    """
    raise NotImplementedError(
        "CUDA backend runs on the GPU box. Install mamba_ssm + causal_conv1d, "
        "download the checkpoint, then this path executes end-to-end. "
        "Use --device cpu_sim in the sandbox."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None, help="override config device")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.device:
        cfg["device"] = args.device

    out_dir = Path(cfg["out_dir"]) / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    RunManifest(run_name=cfg["run_name"], seed=cfg["seed"], config=cfg).save(out_dir)

    if cfg["device"] == "cpu_sim":
        results = run_cpu_sim(cfg)
    else:
        results = run_cuda(cfg)

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    # Console summary: the headline number is the growth exponent b.
    print(f"\n=== Experiment 0 summary ({cfg['device']}) ===")
    print(f"{'condition':<28} {'b (exponent)':>13} {'R^2':>8}  regime")
    for name, r in results.items():
        b = r["fit"]["exponent_b"]
        r2 = r["fit"]["r2"]
        regime = "COMPOUNDING" if b > 1.15 else ("linear" if b > 0.85 else "sub-linear")
        print(f"{name:<28} {b:>13.3f} {r2:>8.3f}  {regime}")
    print(f"\nWrote {out_dir/'results.json'}")


if __name__ == "__main__":
    main()
