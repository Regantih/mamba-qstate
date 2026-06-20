# Companion essay (target ~1,500 words) — outline

*Audience: technical but broad (blog / arXiv companion). Written last, once
results land. Outline below; prose to follow.*

## Working title
"When Memory Is a Single Number: Does Compression Error Compound in State-Space Models?"

## Beats
1. **Hook (~150w).** Transformers remember everything (KV cache); Mamba remembers
   in one small, constantly-overwritten state. Compressing that memory is a
   different kind of risk.
2. **The local-error assumption (~250w).** Why KV-cache quantization gets away
   with aggressive bit-widths: each cached token's error sits still.
3. **The twist (~250w).** In an SSM the state is overwritten every step, so an
   error doesn't sit still — it rides forward. Hypothesis: errors compound.
4. **How we tested it cheaply (~300w).** Noise-injection de-risk before any big
   GPU spend; KL-vs-length growth exponent; a harness validated to detect
   compounding when it truly exists.
5. **What we found (~300w).** [compounding exponent], [8/4-bit frontier],
   [refresh containment]. Plain-language takeaways.
6. **Why the fix is elegant (~150w).** Periodic full-precision refresh: one knob,
   bounds the drift, modest overhead.
7. **What's next (~100w).** Hybrid Mamba-MLA, QAT, adaptive refresh.

## Notes
- Keep one clear figure (KL-vs-length with/without refresh).
- Tie back to reproducibility: seeds, manifests, public repo.
