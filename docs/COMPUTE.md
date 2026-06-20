# Compute plan

**Strategy:** rent on-demand H100s now; do not wait on NVIDIA DGX (~3 weeks out,
arriving ~Jul 11). Two tiers: cheap spot for de-risking, stable on-demand for the
main sweep. DGX credits = free upside for final scaled runs only.

## Tiers

| Tier | Provider | ~Price/hr (H100) | Use for |
|---|---|---|---|
| De-risk (spot) | Vast.ai / RunPod / Voltage Park | $1.33–2.00 | Exp 0, smoke tests, 1.3B passes |
| Main (stable) | Lambda / Jarvislabs | $2.49–2.69 | 3×3×3 sweep, 2.7B, ablations |
| Upside (free) | NVIDIA DGX (~Jul 11) | — | Final scaled / larger-model confirmation |

> Verify live prices at run time — spot rates move. Treat the table as planning bands.

## Budget envelope ($150–250 target)

| Phase | Est. GPU-hours | Tier | Est. cost |
|---|---|---|---|
| Exp 0 de-risk (1.3B) | 4–6 | spot | $8–12 |
| Baseline + PTQ bring-up | 3–5 | spot | $6–10 |
| Main 3×3×3 (2.7B) | 30–45 | stable | $75–120 |
| 3 ablations | 15–25 | stable | $40–65 |
| Buffer / reruns | — | — | $20–40 |
| **Total** | **~55–80 hr** | | **~$150–245** |

## Spend gates (per project rules)
- **< $50:** proceed autonomously, log it.
- **≥ $50 single action OR crossing cumulative $50:** ask before launching.
- Always: prefer spot for anything restartable; checkpoint eval state so a
  preempted spot box loses < 1 cell of work.

## Operational checklist (per box)
1. Launch H100, CUDA 12.x image.
2. `pip install -r requirements.txt && pip install causal-conv1d mamba-ssm && pip install -e .`
3. `python -m pytest tests/ -q` (must pass before spend).
4. `python scripts/baseline_inference.py ...` (confirm checkpoint).
5. Run target config; results sync to `results/` + push summaries to repo.
6. **Tear down the box** when idle — spot meters by the second.

## Cost log
Append every launch here: `date | provider | gpu | hrs | $ | purpose`.
```
2026-06-__ | ____ | H100 | __ | $__ | ____
```
