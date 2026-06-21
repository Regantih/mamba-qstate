"""Experiment 0 — does recurrent-state perturbation error compound with length?

Two backends:
  * device: cuda     -> real Mamba-2 via mamba_ssm (run on the GPU box / RunPod)
  * device: cpu_sim  -> synthetic SSM (sandbox smoke test + directional signal)

Output: results/exp0/<run_name>/ with per-condition KL-vs-step curves, the
fitted growth exponent b, and a manifest. b>1 supports the compounding hypothesis.

Methodology (cuda): for each prompt we first run a full-precision (FP) reference
decode, greedily, recording the chosen token at every step AND the FP next-token
logits. We then re-run the SAME token sequence (teacher-forced) with a StatePolicy
attached to the recurrent state, recording the policy run's logits at each step.
KL(FP || policy) is measured step-by-step on identical inputs, so the only
difference is the perturbed/quantized recurrent state — isolating the effect we
care about from trajectory divergence.

Usage:
  python run_exp0.py --config ../../configs/exp0_default.yaml                 # cuda
  python run_exp0.py --config ../../configs/exp0_default.yaml --device cpu_sim
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from qstate.metrics import kl_divergence, fit_growth_exponent
from qstate.quantizers import QuantConfig
from qstate.state_hook import StatePolicy, attach_state_policy
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
    ssm = SyntheticSSM(SyntheticSSMConfig(spectral_radius=1.0, seed=cfg["seed"]))

    conditions = [(nz, rf) for nz in cfg["noise_levels"] for rf in cfg["refresh_intervals"]]
    results = {}
    rng = torch.Generator().manual_seed(cfg["seed"])

    for noise, refresh in conditions:
        kl_accum = {}
        for _ in range(n_prompts):
            inputs = torch.randn(n_steps, generator=rng).tolist()
            h_full = ssm.init_state()
            h_pert = ssm.init_state()
            nstep = 0
            for t in range(n_steps):
                x = inputs[t]
                h_full = ssm.step(h_full, x)
                h_pert = ssm.step(h_pert, x)
                nstep += 1
                is_refresh = refresh > 0 and (nstep % refresh == 0)
                if is_refresh:
                    h_pert = h_full.clone()
                elif noise > 0:
                    rms = h_pert.pow(2).mean().sqrt().clamp_min(1e-8)
                    h_pert = h_pert + noise * rms * torch.randn(h_pert.shape, generator=rng)
                if (t + 1) % log_every == 0:
                    kl = kl_divergence(ssm.logits(h_full), ssm.logits(h_pert))
                    kl_accum.setdefault(t + 1, []).append(kl)
        steps = sorted(kl_accum)
        kl_mean = [sum(kl_accum[s]) / len(kl_accum[s]) for s in steps]
        results[f"noise{noise}_refresh{refresh}"] = {
            "steps": steps, "kl_mean": kl_mean, "fit": fit_growth_exponent(steps, kl_mean),
        }
    return results


# --------------------------------------------------------------------------- #
# CUDA backend: real Mamba-2.
# --------------------------------------------------------------------------- #
def _load_model_and_tok(model_name: str, device: str):
    from transformers import AutoTokenizer
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    # Mamba checkpoints use the GPT-NeoX-20B tokenizer.
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = MambaLMHeadModel.from_pretrained(model_name, device=device, dtype=torch.float16)
    model.eval()
    return model, tok


def _get_prompts(cfg: dict, tok, device: str):
    """Return a list of prompt input_id tensors (1, L) on device."""
    n = cfg["num_prompts"]
    src = cfg.get("prompt_source", "builtin")
    texts = []
    if src == "pile_val":
        try:
            from datasets import load_dataset
            ds = load_dataset("NeelNanda/pile-10k", split="train", streaming=True)
            for i, ex in enumerate(ds):
                if i >= n:
                    break
                texts.append(ex["text"][:2000])
        except Exception as e:
            print(f"[warn] pile load failed ({e}); falling back to builtin prompts.")
            texts = []
    if not texts:
        base = [
            "The history of scientific discovery shows that",
            "In a distant galaxy, an old engineer explained that",
            "The most important principle of good writing is",
            "When markets become volatile, experienced investors tend to",
            "The recipe begins by carefully preparing the",
            "According to the latest research in neuroscience,",
            "She opened the ancient book and read the first line:",
            "The fundamental theorem of calculus states that",
        ]
        texts = [base[i % len(base)] for i in range(n)]
    prompts = []
    for txt in texts:
        ids = tok(txt, return_tensors="pt").input_ids.to(device)
        prompts.append(ids)
    return prompts


def _make_inference_params(model, batch_size: int, max_seqlen: int):
    from mamba_ssm.utils.generation import InferenceParams
    ip = InferenceParams(max_seqlen=max_seqlen, max_batch_size=batch_size)
    ip.key_value_memory_dict = model.allocate_inference_cache(batch_size, max_seqlen)
    return ip


@torch.no_grad()
def _fp_reference(model, prompt_batch, max_new_tokens, max_seqlen, log_every):
    B = prompt_batch.shape[0]
    ip = _make_inference_params(model, B, max_seqlen)
    out = model(prompt_batch, inference_params=ip, num_last_tokens=1)
    ip.seqlen_offset += prompt_batch.shape[1]
    logits = out.logits[:, -1, :]
    chosen = []
    kept = []
    for t in range(max_new_tokens):
        if (t + 1) % log_every == 0:
            kept.append(logits)
        nxt = logits.argmax(dim=-1, keepdim=True)
        chosen.append(nxt)
        out = model(nxt, inference_params=ip, num_last_tokens=1)
        ip.seqlen_offset += 1
        logits = out.logits[:, -1, :]
    chosen = torch.cat(chosen, dim=1)
    fp_logits = torch.stack(kept, dim=0) if kept else logits.new_zeros((0, B, logits.shape[-1]))
    return chosen, fp_logits

@torch.no_grad()
def _policy_run(model, prompt_batch, chosen_tokens, policy_factory, max_seqlen, log_every):
    B = prompt_batch.shape[0]
    S = chosen_tokens.shape[1]
    handle = attach_state_policy(model, policy_factory)
    handle.reset()
    try:
        ip = _make_inference_params(model, B, max_seqlen)
        out = model(prompt_batch, inference_params=ip, num_last_tokens=1)
        ip.seqlen_offset += prompt_batch.shape[1]
        logits = out.logits[:, -1, :]
        kept = []
        for t in range(S):
            if (t + 1) % log_every == 0:
                kept.append(logits)
            nxt = chosen_tokens[:, t:t+1]
            out = model(nxt, inference_params=ip, num_last_tokens=1)
            ip.seqlen_offset += 1
            logits = out.logits[:, -1, :]
        return torch.stack(kept, dim=0) if kept else logits.new_zeros((0, B, logits.shape[-1]))
    finally:
        handle.remove()





def _policy_factory(bits: int, refresh: int, seed: int,
                    granularity: str = "per_head", symmetric: bool = True):
    qcfg = QuantConfig(bits=bits, granularity=granularity, symmetric=symmetric)
    quant = qcfg if qcfg.enabled() else None  # bits>=16 -> FP control
    return lambda: StatePolicy(
        quant=quant,
        refresh_interval=refresh,
        noise_std=0.0,
        seed=seed,
    )
def run_cuda(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    device = "cuda"
    model, tok = _load_model_and_tok(cfg["model"], device)
    prompts = _get_prompts(cfg, tok, device)
    max_new = cfg["max_new_tokens"]
    log_every = cfg["log_every"]

    conditions = [(bw, rf) for bw in cfg["bit_widths"] for rf in cfg["refresh_intervals"]]
    results = {}

    trunc = cfg.get('prompt_trunc', 64)
    prompt_batch = torch.cat([p[:, -trunc:] for p in prompts], dim=0)
    max_seqlen = trunc + max_new + 1
    chosen, fp_logits = _fp_reference(model, prompt_batch, max_new, max_seqlen, log_every)
    log_steps = [t + 1 for t in range(max_new) if (t + 1) % log_every == 0]

    for bits, refresh in conditions:
        key = "bits" + str(bits) + "_refresh" + str(refresh)
        if bits >= 16 and refresh == 0:
            pol_logits = fp_logits
        else:
            pf = _policy_factory(bits, refresh, cfg["seed"], cfg.get("quant_granularity","per_head"), cfg.get("quant_symmetric",True))
            pol_logits = _policy_run(model, prompt_batch, chosen, pf, max_seqlen, log_every)
        steps = log_steps
        kl_mean = [kl_divergence(fp_logits[i], pol_logits[i]) for i in range(len(steps))]
        fit = fit_growth_exponent(steps, kl_mean)
        results[key] = {"steps": steps, "kl_mean": kl_mean, "fit": fit}
        print("  done " + key + ": b=" + format(fit["exponent_b"], ".3f"))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None, help="override config device (cuda|cpu_sim)")
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

    print(f"\n=== Experiment 0 summary ({cfg['device']}) ===")
    print(f"{'condition':<28} {'b (exponent)':>13} {'R^2':>8}  regime")
    for name, r in results.items():
        b = r["fit"]["exponent_b"]; r2 = r["fit"]["r2"]
        regime = "COMPOUNDING" if b > 1.15 else ("linear" if b > 0.85 else "sub-linear")
        print(f"{name:<28} {b:>13.3f} {r2:>8.3f}  {regime}")
    print(f"\nWrote {out_dir/'results.json'}")
    print("\n>>> PASTE THIS FILE BACK:", (out_dir / "results.json").resolve())


if __name__ == "__main__":
    main()
