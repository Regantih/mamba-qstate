# Plot Experiment 0: KL divergence vs generation length for each (bit-width, refresh) condition.
# Reads results/exp0/exp0_quant/results.json and writes paper/figures/exp0_kl_growth.png
import json
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results/exp0/exp0_quant/results.json')
    ap.add_argument('--out', default='paper/figures/exp0_kl_growth.png')
    args = ap.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in sorted(data.keys()):
        cond = data[name]
        steps = cond.get('steps')
        kl = cond.get('kl_mean')
        if not steps or not kl:
            continue
        b = cond.get('fit', {}).get('exponent_b', float('nan'))
        r2 = cond.get('fit', {}).get('r2', float('nan'))
        label = name + ' (b=' + format(b, '.2f') + ', R2=' + format(r2, '.2f') + ')'
        ax.plot(steps, kl, marker='o', label=label)

    ax.set_xlabel('generation length (tokens)')
    ax.set_ylabel('mean KL(full || quant)')
    ax.set_title('Exp 0: SSM-state quantization error vs generation length')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
