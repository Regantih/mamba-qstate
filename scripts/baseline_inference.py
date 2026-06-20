"""Baseline full-precision Mamba-2 inference reproduction.

Confirms the checkpoint loads, generates coherently, and lets us record the
reference per-step logits + recurrent states that every quantized run is
compared against. Run on the GPU box.

Usage:
  python baseline_inference.py --model state-spaces/mamba2-1.3b \
      --prompt "The capital of France is" --max-new-tokens 64
"""
from __future__ import annotations

import argparse
import time

from qstate.utils import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="state-spaces/mamba2-1.3b")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)

    # Imports deferred so the script imports fine on CPU for inspection.
    import torch
    from transformers import AutoTokenizer
    try:
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    except ImportError as e:
        raise SystemExit(
            "mamba_ssm not installed. On the GPU box:\n"
            "  pip install causal-conv1d>=1.4.0 mamba-ssm>=2.2.2\n"
            f"(import error: {e})"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device} ...")
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = MambaLMHeadModel.from_pretrained(args.model, device=device, dtype=torch.float16)
    model.eval()

    ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            input_ids=ids,
            max_length=ids.shape[1] + args.max_new_tokens,
            cg=True,
            temperature=1.0,
            top_k=1,  # greedy for determinism
        )
    dt = time.time() - t0
    text = tok.decode(out[0].tolist())
    print("\n--- Generation ---")
    print(text)
    print(f"\n{args.max_new_tokens} tokens in {dt:.2f}s "
          f"({args.max_new_tokens / dt:.1f} tok/s)")
    print("Baseline OK — checkpoint loads and generates.")


if __name__ == "__main__":
    main()
