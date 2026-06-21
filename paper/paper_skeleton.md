# Quantized Recurrent State for Mamba Inference: Do Compression Errors Compound?

*Workshop-track paper skeleton. Each section lists the claim it must make and the
evidence that backs it. Fill with numbers from `results/` as runs land.*

---

## Abstract
State-space models (SSMs) such as Mamba-2 replace the Transformer KV cache with a
single fixed-size **recurrent state** that is overwritten at every step. We ask
whether post-training quantization of this state behaves like KV-cache
quantization — where errors are assumed **local** — or whether the recurrent
overwrite causes errors to **compound** over generation length. We (i) inject
calibrated noise into the recurrent state and show KL divergence vs. full
precision grows [super-linearly / ~linearly] with token count (exponent
b = ___); (ii) quantize the state to 8/4/3-bit and map the quality–compression
frontier on retrieval, QA, and perplexity; and (iii) show a **periodic
full-precision refresh** every k steps drives the growth exponent toward 1 and
recovers [__]% of the quality gap at [__]% memory overhead. *(Numbers TBD.)*

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
- Models: mamba2-1.3b (de-risk), mamba2-2.7b (main). Greedy decoding, fixed seed.
- Tasks: NIAH long-context retrieval, SQuAD-v2 subset (EM/F1), Pile-val PPL.
- Grid: bit ∈ {8,4,3} × refresh ∈ {0,64,16} × task; + 3 ablations.
- Compute: on-demand H100 (see docs/COMPUTE.md); full reproducibility manifests.

## 5. Results
- **5.1 Compounding.** Exp-0 + main: b across bit-widths/lengths. *(figure: KL-vs-step, log-log)*
  Across all conditions the recurrent-state quantization error is **sub-linear**, not compounding: the fitted KL growth exponent b stays well below 1 in every quantized case. On Mamba-2 1.3B over 16 pile_val prompts decoded for 512 tokens, 8-bit per-head symmetric quantization gives b in {0.47, 0.63, 0.56} for refresh k in {0, 16, 64}; 4-bit gives b in {0.41, 0.41, 0.39}. Fit quality R^2 ranges 0.29-0.66 (strongest for 4-bit). Periodic full-precision refresh (k in {16, 64}) does not raise b, consistent with no compounding error build-up to contain. Full-precision controls (>=16-bit) yield KL~0 (b=nan). See `paper/figures/exp0_kl_growth.png` (log-log KL vs generation length). **This inverts the original hypothesis: the SSM recurrent state appears robust to aggressive quantization, with error accumulating sub-linearly rather than compounding.**
- **5.2 Compression frontier.** Quality vs. bits per task. *(figure: Pareto)*
- **5.3 Refresh containment.** b and quality vs. k; overhead trade-off. *(figure)*
- **5.4 Ablations.** Outlier protection, granularity, refresh schedule.

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
