# Two-week sprint plan — Quantized Recurrent State for Mamba Inference

**Hard deadline:** arXiv submission **Fri Jul 31, 2026**.
**Working window:** Sat Jun 20 → Fri Jul 31 (~6 weeks of calendar, but the
research is scoped as a **focused 2-week build** + buffer). Plan below is the
2-week core sprint; the remaining slack absorbs DGX upside runs and revisions.

**Decision gates (ask the user):** ⛳ marks a checkpoint. Spend ≥ $50 also gates.

---

## Week 1 — De-risk the hypothesis (cheap, fast)

### Day 1 — Sat Jun 20  ✅ (today)
- Repo scaffold: src/evals/experiments/paper, MIT, seeds, CI-able tests.
- Core lib: quantizers (8/4-bit, granularities), metrics (KL + growth-exponent),
  StatePolicy (noise + quant + refresh).
- Exp 0 pipeline written; CPU-sim smoke test green; **harness validated** to
  detect compounding when present. Unit tests 8/8 pass.

### Day 2 — Sun Jun 21
- Rent first **spot H100** (Vast/RunPod). Install mamba_ssm + checkpoint.
- Run `baseline_inference.py` on mamba2-1.3b → confirm coherent generation.
- Wire the live `attach_state_policy` hook to mamba_ssm `ssm_states`.

### Day 3 — Mon Jun 22  ⛳ GATE 1: hypothesis check
- Run **Experiment 0** on real mamba2-1.3b: noise sweep × refresh, KL vs length.
- Fit growth exponent b. **If b > 1 → hypothesis holds, proceed.** If b ≈ 1 →
  pivot framing to "refresh enables aggressive quant" (still publishable).
- Report b, curves, and go/no-go to user.

### Day 4 — Tue Jun 23
- Finalize PTQ module against real states (per-head default). Validate 8-bit
  near-lossless, characterize 4-bit break point on short generations.
- Land outlier-protection option (ablation A1 hook).

### Day 5 — Wed Jun 24
- Stand up eval harness: NIAH retrieval, SQuAD subset, Pile perplexity.
- Dry-run each on 1.3B at 8-bit to confirm metrics + logging plumbing.

### Day 6–7 — Thu–Fri Jun 25–26
- Pilot a **2×2 corner** of the main grid on 1.3B (8/4-bit × refresh 0/16) to
  lock configs, runtime/cost per cell, and result schema.
- ⛳ GATE 2: review pilot numbers + budget burn; approve main sweep spend.

---

## Week 2 — Main results + paper

### Day 8 — Mon Jun 29
- Launch **main 3×3×3** on stable H100 (Lambda/Jarvislabs), mamba2-2.7b.
  bit∈{8,4,3} × refresh∈{0,64,16} × task∈{retrieval,qa,ppl}. Checkpoint per cell.

### Day 9 — Tue Jun 30
- Monitor sweep; backfill failures. Begin **Results** plots (KL-vs-length,
  accuracy-vs-bits, refresh containment curve).

### Day 10 — Wed Jul 1
- Run **3 ablations** (outlier-protect, granularity, warmup-refresh schedule).
- Draft Method + Experiments sections from the actual run configs.

### Day 11 — Thu Jul 2
- Lock figures. Write Results narrative around the headline: compounding +
  refresh containment + the 8/4-bit Pareto frontier.

### Day 12 — Fri Jul 3
- Full paper draft (Abstract → Future Work incl. Hybrid Mamba-MLA #25).
- ⛳ GATE 3: user reviews full draft.

### Day 13 — (buffer) week of Jul 6
- Incorporate feedback; tighten Related Work (local-vs-compounding framing).
- Companion ~1,500-word essay draft.

### Day 14 — (buffer) week of Jul 6
- Reproducibility pass: fresh-clone → tests → one cell reproduces from manifest.
- Polish repo README + release tag.

---

## Slack: Jul 7 → Jul 31
- **~Jul 11:** DGX credits arrive → optional scaled confirmation (larger model /
  longer context) as upside, not critical path.
- arXiv formatting, final proofread, endorsement/submission logistics.
- **Jul 31:** submit. ⛳ GATE 4: final user sign-off before posting.

## Risk register
| Risk | Mitigation |
|---|---|
| b ≈ 1 (no compounding) | Reframe to refresh-enabled aggressive quant; A/B already in plan |
| mamba_ssm state hook friction | StatePolicy logic is decoupled + unit-tested; only the attach point is GPU-specific |
| Spot preemption | Checkpoint per eval cell; restartable configs |
| 4-bit collapse | Outlier protection (A1) + per-channel granularity (A2) as rescue levers |
| Budget overrun | Spend gates at $50; 1.3B de-risk before 2.7B sweep |
