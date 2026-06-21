# Quantized Recurrent State for Mamba Inference: Do Compression Errors Compound?

*Consolidated arXiv outline. All numbers measured on state-spaces/mamba2-1.3b over Pile (NeelNanda/pile-10k), greedy decoding, seed 1337. Artifacts under results/exp1/.*

## Abstract
SSMs such as Mamba-2 replace the Transformer KV cache with a single fixed-size recurrent state that is overwritten every step. We ask whether post-training quantization of this state stays local (like KV-cache quantization) or compounds over generation length. The answer is bit-width dependent: 8-bit is effectively lossless, while 4-bit and 3-bit compound and fall off a perplexity cliff. A periodic full-precision refresh every k steps reverses the compounding and restores quality at low overhead.

## 1. Contributions
- A compounding diagnostic: per-step KL(fp16 || quant) of the recurrent-state-quantized model vs full precision, fit to a power law to extract a growth exponent b.
- A measured quality-compression frontier across {16,8,4,3}-bit recurrent state on Pile perplexity.
- A refresh-containment mechanism: periodic full-precision state resets that drive the growth exponent negative at low memory overhead.

## 2. Setup
Model state-spaces/mamba2-1.3b; greedy decoding; seed 1337. Data: Pile (NeelNanda/pile-10k). Metric: per-step KL(fp16 || quant) for compounding, and per-token perplexity for the quality frontier. Grid: bits in {16,8,4,3} x horizon in {128,512}, refresh=0; plus a refresh sweep k in {0,16,64} at 4-bit, H=512. Single on-demand GPU (RunPod). Scripts: experiments/exp1_real_quant/{run,resume,refresh_sweep,ppl_fast}.py.

## 3. Results
### 3.1 Compounding depends on bit-width
- 8-bit: no compounding (b=-0.15 at H=128; b=0.13 at H=512); effectively lossless.
- 4-bit: compounds; positive exponent (b=1.13 at H=128; b=0.49 at H=512), terminal KL grows with horizon (0.073 to 0.131).
- 3-bit: diverges; severe compounding (b=0.81 at H=128; b=1.13 at H=512), terminal KL 0.27 to 1.80.
Figure: results/exp1/figs/exp1_kl_growth.png, exp1_terminal_kl.png. Data: results/exp1/sweep.json; results/exp1/diagnostics/horizon_crossover.json.

### 3.2 Quality-compression frontier (per-token PPL)
| Bits | Compression | PPL | dPPL vs fp16 | Terminal KL (H=512) | On frontier |
|------|-------------|-------|--------------|---------------------|-------------|
| 16   | 1.0x        | 10.51 | 0.0%         | 0.000               | baseline    |
| 8    | 2.0x        | 10.54 | +0.3%        | 0.004               | yes         |
| 4    | 4.0x        | 29.19 | +178%        | 0.131               | no          |
| 3    | 5.3x        | 23.69 | +125%        | 1.800               | no          |
The only Pareto-optimal compressed point is 8-bit (2x smaller state, no measurable quality loss). Below 8-bit the recurrent overwrite compounds quantization error into a >2x perplexity cliff, so static sub-8-bit state quantization is not usable without refresh. Data: results/exp1/ppl.json; results/exp1/diagnostics/frontier.json.

### 3.3 Refresh containment (4-bit, H=512)
A periodic full-precision refresh every k steps contains 4-bit compounding. Without refresh (k=0) the exponent is b=0.57 (terminal KL 0.281). Refreshing every 16 steps drops it to b=-1.19 (terminal KL 0.0022, ~130x lower); every 64 steps gives b=-2.43 (terminal KL 0.0010, ~280x lower). A negative exponent means the periodic reset removes error faster than the recurrence accumulates it, so compounding is eliminated even at low (k=64) refresh overhead.
Figures: results/exp1/figs/refresh_kl.png, refresh_b_vs_k.png. Data: results/exp1/refresh.json; results/exp1/diagnostics/refresh_summary.json.

## 4. Discussion and Limitations
Recurrent-state quantization is not analogous to KV-cache quantization below 8-bit: the single overwritten state turns local rounding error into a compounding process, producing a sharp quality cliff. 8-bit is safe; sub-8-bit needs periodic full-precision refresh, which trades a small memory overhead for a return to near-lossless behavior. Limitations: single model scale (1.3b), PTQ only (no QAT), perplexity-only quality axis at NP=8 (3-bit estimate is high-variance), and a Python per-token reference loop rather than a fused kernel.

## 5. Future Work
- Hybrid Mamba-MLA: combine SSM recurrence with a partial KV path to absorb compounding.
- QAT for the recurrent state; learned adaptive refresh schedules; hardware realization of low-bit state storage.
- Scale study across model sizes and longer horizons; broader quality axes (retrieval, QA).

## 6. Reproducibility
Repo: github.com/Regantih/mamba-qstate. Compounding: experiments/exp1_real_quant/run.py + sweep.json. Frontier: ppl_fast.py + ppl.json + diagnostics/frontier.json. Refresh: refresh_sweep.py + refresh.json. Figures: scripts/plot_exp0.py, scripts/plot_refresh.py. Seed 1337; mamba2-1.3b; Pile (NeelNanda/pile-10k).
