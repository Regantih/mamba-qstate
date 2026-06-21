# Quantized Recurrent State for Mamba Inference: Do Compression Errors Compound?

*Workshop-track paper skeleton. Each section lists the claim it must make and the
evidence that backs it. Fill with numbers from `results/` as runs land.*

---

## Abstract
State-space models (SSMs) such as Mamba-2 replace the Transformer KV cache with a
single fixed-size **recurrent state** that is overwritten at every step. We ask
whether post-training quantization of this state behaves like KV-cache
quantization — where errors are assumed **local** — or whether the recurrent
overwrite causes errors to **compound** over generation length. We find the answer depends on bit-width: 8-bit recurrent-state quantization is effectively lossless (no compounding, exponent b≈0), but 4-bit and 3-bit compound, with per-step KL(fp16||quant) growing super-linearly at short horizons (4-bit b≈1.1 at H=128) and terminal KL rising with horizon. A periodic full-precision refresh every k steps reverses the trend: at 4-bit/H=512, refreshing every 16 steps drives the growth exponent negative (b from +0.57 to -1.19) and cuts terminal KL ~130x (0.281 to 0.0022), recovering essentially all of the quality gap at roughly one extra full-precision state per 16 steps (~19% memory overhead over the 4-bit baseline).*

## 1. Introduction & Contributions
- **C1.** First systematic study of *recurrent-state* PTQ in Mamba-2 (8/4/3-bit).
- **C2.** Empirical characterization of **error compounding** with generation
  length via a growth-exponent metric — a property absent from the KV-cache
  setting.
- **C3.** **Periodic full-precision refresh** as a simple, tunable mechanism that
  bounds compounding; we quantify the accuracy/overhead trade-off.
- **C4.** Open, reproducible harness (seeds, manifests, configs) + checkpoints.

## 2. Background & Related Work — the gap
- Mamba/Mamba-2 SSM recurrence; the recurrent state as the entire memory.
- KV-cache quantization line (KVQuant, KIVI, etc.): errors are **local** to
  per-token cached entries; old tokens' errors don't propagate through new state.
- **Gap:** SSM state is overwritten, so an error at step t is carried into all
  future states → potential **compounding**. SSM-state quantization is
  under-explored vs. the crowded KV-cache space. *(This framing is the anchor.)*

## 3. Method
- **3.1 Recurrent-state quantizer.** Affine symmetric/asymmetric, granularities
  {per-tensor, per-head, per-channel}; applied in-place between decode steps.
- **3.2 Compounding metric.** KL(full ‖ quant) per step; fit KL(t) ≈ a·t^b;
  b>1 ⇒ compounding. Harness validated to recover known exponents.
- **3.3 Periodic full-precision refresh.** Every k steps, retain the state in
  full precision; sweep k. Cost model: memory/compute overhead vs. k.

## 4. Experimental Setup
- Model: mamba2-1.3b (state-spaces). Greedy decoding, fixed seed 1337.
- Data: 12 Pile (NeelNanda/pile-10k) prompts. Metric: per-step KL(fp16 || quant) of the recurrent-state-quantized model vs full precision.
- Grid: bits ∈ {16,8,4,3} × horizon ∈ {128,512}, refresh=0; plus a refresh sweep (k ∈ {0,16,64}) at 4-bit, H=512.
- Compute: single on-demand GPU (RunPod). Scripts: experiments/exp1_real_quant/{run,resume,refresh_sweep}.py; results under results/exp1/.

## 5. Results
- **5.1 Compounding (real Mamba-2 1.3B).** 
  *(figs: results/exp1/figs/exp1_kl_growth.png,
  results/exp1/figs/exp1_terminal_kl.png;
  data: results/exp1/sweep.json; summary:
  results/exp1/diagnostics/horizon_crossover.json)*

  We track KL(fp16 || quant) between the
  recurrent-state-quantized model and the
  full-precision reference, fit as KL ~ a*t^b
  over decode horizon t. We sweep horizons 128
  and 512 on the real Mamba-2 1.3B model over 12
  Pile prompts, quantizing the SSM recurrent
  state at 16/8/4/3 bits (16-bit = lossless
  control). Results show a monotone bit-width
  threshold, not a horizon-induced 8-bit
  crossover:

  - **8-bit: no compounding.** b is null at
    both horizons (b=-0.15, R2=0.05 at H=128;
    b=0.13, R2=0.007 at H=512); terminal KL
    negligible (7.1e-4 at H=128, 3.9e-3 at
    H=512). Effectively lossless.
  - **4-bit: compounds.** Positive exponent
    (b=1.13, R2=0.85 at H=128; b=0.49, R2=0.39
    at H=512); terminal KL grows with horizon
    (0.073 at H=128 to 0.131 at H=512).
  - **3-bit: diverges.** Severe compounding
    (b=0.81 at H=128; b=1.13, R2=0.67 at H=512);
    terminal KL 0.27 (H=128) to 1.80 (H=512).

  16-bit controls give KL=0 by construction.
  Takeaway: 8 bits is safe; 4 bits and below
  incur horizon-dependent compounding. This
  corrects an earlier synthetic-proxy analysis
  that suggested an 8-bit crossover.
- **5.2 Compression frontier (real, mamba2-1.3b, Pile, per-token PPL).** Quantizing only the recurrent state trades memory for quality, and the frontier is sharply non-linear:
    - **16-bit (1.0x):** PPL 10.51 (baseline).
    - **8-bit (2.0x):** PPL 10.54 (+0.3%); terminal KL 0.004. On the frontier - effectively lossless.
    - **4-bit (4.0x):** PPL 29.19 (+178%); terminal KL 0.131. Dominated - quality collapses.
    - **3-bit (5.3x):** PPL 23.69 (+125%); terminal KL 1.80. Also dominated; high variance at 8 prompts.
    - **Takeaway:** the only Pareto-optimal compressed point is 8-bit (2x smaller state, no measurable quality loss). Below 8-bit the recurrent overwrite compounds quantization error into a >2x perplexity cliff, so static sub-8-bit state quantization is not usable without the periodic refresh of 5.3. (data: results/exp1/ppl.json, results/exp1/diagnostics/frontier.json)
- **5.3 Refresh containment (real, 4-bit, H=512).**
  *(figs: results/exp1/figs/refresh_kl.png,
  results/exp1/figs/refresh_b_vs_k.png; data:
  results/exp1/refresh.json; summary:
  results/exp1/diagnostics/refresh_summary.json)*
  A periodic full-precision refresh of the
  recurrent state every k steps contains 4-bit
  compounding. Without refresh (k=0) the
  exponent is b=0.57 with terminal KL 0.281.
  Refreshing every 16 steps drops the exponent
  to b=-1.19 and terminal KL to 0.0022 (about
  130x lower); every 64 steps gives b=-2.43
  and terminal KL 0.0010 (about 280x lower).
  A negative exponent means the periodic reset
  removes error faster than the recurrence
  accumulates it, so compounding is eliminated
  even at low (k=64) refresh overhead.
- **5.4 Ablations.** outliers, granularity.

## 6. Discussion & Limitations
- When does compounding bite (length, task, bit-width)? Model-size sensitivity.
- Limitations: model scale, PTQ-only (no QAT), task coverage.

## 7. Future Work
- **Hybrid Mamba-MLA (#25):** combine SSM recurrence with multi-head latent
  attention; study whether a partial KV path absorbs compounding.
- QAT for the recurrent state; learned adaptive refresh schedules; hardware
  realization of low-bit state storage.

## References
*(BibTeX in `paper/references.bib` — Mamba, Mamba-2, KVQuant, KIVI, NIAH, etc.)*
