#!/usr/bin/env bash
# One-command reproduction of all results and figures for the paper.
# Run from repo root on a CUDA GPU box. Expects deps from paper/arxiv/requirements.txt.
set -euo pipefail
export PYTHONUNBUFFERED=1

echo "[1/5] Perplexity sweep (1.3b + 370m)"
python3 experiments/exp1_real_quant/ppl_fast_scale.py

echo "[2/5] Symmetric vs asymmetric quant"
python3 experiments/exp1_real_quant/ppl_asym.py

echo "[3/5] Powered NIAH retrieval (Wilson CIs)"
NTRIAL=60 python3 experiments/exp1_real_quant/niah_power.py

echo "[4/5] Measured refresh throughput"
python3 experiments/exp1_real_quant/refresh_throughput.py

echo "[5/5] Regenerate figures"
python3 paper/arxiv/make_figures.py
echo "All done. See results/exp1/*.json and paper/arxiv/figures/*.pdf"
