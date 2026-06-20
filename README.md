# Quantized Recurrent State for Mamba Inference

**Research question:** KV-cache quantization research assumes quantization errors stay *local*. In state-space models (SSMs) like Mamba-2, memory is a single fixed-size **recurrent state** that is overwritten at every decoding step. Does this make quantization error **compound** over generation length, and can a **periodic full-precision refresh** contain it?

This repo studies:
- **(a) Aggressiveness:** How far can the recurrent SSM state be quantized (8-bit / 4-bit) before quality breaks?
- **(b) Compounding:** Does KL divergence vs. full precision grow *super-linearly* with generation length?
- **(c) Containment:** Does periodically refreshing the state to full precision (every *k* steps) bound the error?

> **Why it's novel:** SSM recurrent-state quantization is a wide-open gap. KV-cache quantization (for Transformers) is crowded. The local-vs-compounding error framing is the paper's anchor.

---

## Status

| Milestone | State |
|---|---|
| Repo scaffold | ✅ |
| Experiment 0 (noise-injection de-risk) | 🔜 |
| PTQ module (8/4-bit + refresh) | 🔜 |
| Main 3×3×3 sweep | ⬜ |
| Paper draft | ⬜ |

**Hard deadline:** arXiv submission by **July 31, 2026**.

---

## Repository layout

```
mamba-qstate/
├── src/qstate/          # Core library: quantizers, state hooks, refresh logic, metrics
├── experiments/
│   └── exp0_noise_injection/   # Week-1 de-risk: calibrated noise → KL vs length
├── evals/               # Long-context retrieval, QA, perplexity harnesses
├── configs/             # YAML configs for sweeps (bit-width × refresh × task)
├── scripts/             # Baseline reproduction, run launchers, plotting
├── tests/               # Unit tests (quantizer round-trip, refresh correctness)
├── results/             # Output JSON/CSV (gitignored except summaries)
├── paper/               # arXiv paper source + companion essay
├── requirements.txt
├── pyproject.toml
└── LICENSE              # MIT
```

## Quickstart

```bash
# 1. Environment (on a rented H100 box, CUDA 12.x)
pip install -r requirements.txt
pip install -e .

# 2. Reproduce baseline (full-precision) inference
python scripts/baseline_inference.py --model state-spaces/mamba2-1.3b --prompt "The capital of France is"

# 3. Experiment 0 — does error compound with length?
python experiments/exp0_noise_injection/run_exp0.py --config configs/exp0_default.yaml
```

## Reproducibility

- All randomness seeded via `qstate.utils.set_seed` (default `1337`).
- Every run writes a `manifest.json` capturing git SHA, config, package versions, GPU type, and seed.
- Configs are version-controlled YAML; results reference the config hash.

## Compute

See [`docs/COMPUTE.md`](docs/COMPUTE.md). Summary: rent on-demand H100s (Vast.ai / RunPod / Voltage Park for cheap de-risk; Lambda / Jarvislabs for the stable main sweep). NVIDIA DGX credits (~Jul 11) are treated as free upside for final scaled runs only.

## License

MIT — see [LICENSE](LICENSE).
