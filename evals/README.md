# Eval harness

Three tasks probe different failure modes of recurrent-state quantization:

| Task | Probes | Metric | Why it stresses the state |
|---|---|---|---|
| **Long-context retrieval** (needle-in-a-haystack) | Information *persistence* in the compressed state | retrieval accuracy @ context length | The needle must survive in the overwritten recurrent state across thousands of steps — the harshest test of compounding error. |
| **QA** (SQuAD-v2 subset) | Task quality under quantization | exact-match / F1 | Realistic downstream quality signal. |
| **Perplexity** (Pile val) | Distributional fidelity | bits-per-byte / PPL | Cheap, dense, sensitive to small per-step drift. |

## Headline diagnostic
Across all tasks we also log **KL(full ‖ quant) vs. generation step** and fit the
growth exponent *b* (`qstate.metrics.fit_growth_exponent`). The central claim is
tested by whether *b > 1* (compounding) and whether refresh drives *b → 1*.

## Files (to implement on the GPU box)
- `niah.py` — synthesize needle-in-a-haystack contexts at configurable lengths.
- `qa.py` — SQuAD-v2 subset loader + EM/F1 scorer.
- `perplexity.py` — streaming Pile-val perplexity.
- `harness.py` — shared loop: load model, attach `StatePolicy`, run task, dump JSON.

Each writes `results/<task>/<condition>/{metrics.json,kl_curve.json,manifest.json}`.
