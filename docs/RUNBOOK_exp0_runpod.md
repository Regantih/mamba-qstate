# RunPod runbook — Experiment 0 (real mamba2-1.3b)

Goal: run the noise-injection de-risk on a real H100 and produce
`results/exp0/exp0_noise_injection/results.json` with the growth exponent **b**
per condition. Estimated wall time ~10–25 min on one H100; cost ~$8–12.

## 0. Pod
- GPU: 1× H100 (80GB). Spot/Community is fine.
- Template: a PyTorch + CUDA 12.x image (e.g. "RunPod PyTorch 2.x CUDA 12.x").
- Open a web terminal or SSH in.

## 1. Clone + environment
```bash
cd /workspace
git clone https://github.com/Regantih/mamba-qstate
cd mamba-qstate

pip install -r requirements.txt
# Mamba-2 CUDA kernels (must match the image's CUDA/torch):
pip install causal-conv1d>=1.4.0
pip install mamba-ssm>=2.2.2
pip install -e .
pip install datasets        # only needed if prompt_source: pile_val
```

If `mamba-ssm` build fails, it's almost always a CUDA/torch mismatch. Check:
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvcc --version
```
Pick a template whose CUDA matches torch's `version.cuda`, then reinstall mamba-ssm.

## 2. Pre-spend sanity (no GPU cost beyond idle)
```bash
python -m pytest tests/ -q          # expect: 8 passed
```

## 3. Confirm the checkpoint loads & generates
```bash
python scripts/baseline_inference.py \
  --model state-spaces/mamba2-1.3b \
  --prompt "The capital of France is" --max-new-tokens 32
```
Expect coherent text + a tok/s line + "Baseline OK".

## 4. Run Experiment 0
```bash
python experiments/exp0_noise_injection/run_exp0.py \
  --config configs/exp0_default.yaml
```
- Default config: mamba2-1.3b, 512 new tokens, 16 prompts (builtin),
  noise ∈ {0, 0.004, 0.01, 0.03, 0.1} × refresh ∈ {0, 16, 64}, KL logged every 8 steps.
- The run prints a summary table (b + R² + regime per condition) and ends with:
  `>>> PASTE THIS FILE BACK: .../results.json`

### Faster/cheaper first pass (optional)
Edit `configs/exp0_default.yaml` before running:
- `num_prompts: 8`, `max_new_tokens: 256` → roughly halves cost, still shows the trend.
Then run the full config if the trend looks real.

### Want real-text prompts?
Set `prompt_source: pile_val` in the config (needs `datasets`, downloads a small slice).

## 5. Where the output lands
```
results/exp0/exp0_noise_injection/
├── results.json     <-- PASTE THIS BACK
└── manifest.json    (git SHA, config, package versions, GPU, seed)
```
Print it to copy:
```bash
cat results/exp0/exp0_noise_injection/results.json
```

## 6. TEAR DOWN THE POD
Spot bills by the second. Stop/terminate the pod as soon as you've copied the JSON.

## 7. What to paste back to me
Paste the **entire contents of `results.json`** (and, if it's handy, the printed
summary table). That file contains, per condition, the `steps`, `kl_mean`, and the
fitted `{exponent_b, prefactor_a, r2}`. With it I will:
- Read the **b exponent** for the no-refresh conditions → **the go/no-go**
  (b > 1 = compounding = hypothesis holds).
- Check that the `noise0.0_refresh0` control is ~0 (validity check).
- Quantify how much **refresh** lowers b / KL (containment evidence).
- Write the GATE-1 go/no-go memo and update the paper's Results 5.1 + Exp-0 figure.

If anything errors during install/run, paste the traceback and I'll debug it.
