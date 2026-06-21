import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, 'results', 'exp1')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)
def load(name):
    with open(os.path.join(RES, name)) as fh:
        return json.load(fh)
ppl = load('ppl.json')
ppl370 = load('ppl_370m.json')
bits = [16, 8, 4, 3]
y13 = [ppl['bits%d' % b] for b in bits]
y37 = [ppl370['bits%d' % b] for b in bits]
fig, ax = plt.subplots(figsize=(5, 3.4))
ax.plot(bits, y13, 'o-', label='Mamba-2 1.3b')
ax.plot(bits, y37, 's--', label='Mamba-2 370m')
ax.set_yscale('log')
ax.invert_xaxis()
ax.set_xlabel('State quantization bit-width')
ax.set_ylabel('Perplexity (log)')
ax.set_xticks(bits)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'ppl_bitwidth.pdf'))
plt.close(fig)
rf = load('refresh.json')
cur = rf['curve']
keys = ['refresh0', 'refresh16', 'refresh32', 'refresh64']
ks = [cur[k]['k'] for k in keys]
bs = [cur[k]['b'] for k in keys]
fig, ax = plt.subplots(figsize=(5, 3.4))
ax.plot(ks, bs, 'o-', color='crimson')
ax.axhline(0, color='gray', lw=0.8, ls=':')
ax.set_xlabel('Refresh interval k (steps)')
ax.set_ylabel('Error growth exponent b')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'refresh_b.pdf'))
plt.close(fig)
q = load('quality_power.json')['recall']
acc = [q['bits%d' % b]['acc'] for b in bits]
lo = [q['bits%d' % b]['acc'] - q['bits%d' % b]['ci95_lo'] for b in bits]
hi = [q['bits%d' % b]['ci95_hi'] - q['bits%d' % b]['acc'] for b in bits]
fig, ax = plt.subplots(figsize=(5, 3.4))
ax.errorbar([str(b) for b in bits], acc, yerr=[lo, hi], fmt='none', ecolor='black', capsize=4, zorder=3)
ax.bar([str(b) for b in bits], acc, color=['#22cc77', '#22cc77', '#cc4444', '#cc4444'])
ax.set_ylim(0, 1.05)
ax.set_xlabel('Bit-width')
ax.set_ylabel('NIAH recall (20 trials)')
ax.grid(True, axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'niah_recall.pdf'))
plt.close(fig)
print('Wrote figures to', OUT)
a = load('ppl_asym.json')
import numpy as np
x = np.arange(len(bits))
sym = [a['symmetric']['bits%d' % b] for b in bits]
asym = [a['asymmetric']['bits%d' % b] for b in bits]
fig, ax = plt.subplots(figsize=(5, 3.4))
ax.bar(x-0.2, sym, 0.4, label='symmetric')
ax.bar(x+0.2, asym, 0.4, label='asymmetric')
ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels([str(b) for b in bits])
ax.set_xlabel('Bit-width'); ax.set_ylabel('Perplexity (log)'); ax.legend(); ax.grid(True, axis='y', alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'asym.pdf')); plt.close(fig)
tp = load('refresh_throughput.json')
labels = ['fp16', 'k=16', 'k=32', 'k=64']
tps = [tp['baseline_int8_equiv']['tok_per_s'], tp['k16']['tok_per_s'], tp['k32']['tok_per_s'], tp['k64']['tok_per_s']]
fig, ax = plt.subplots(figsize=(5, 3.4))
ax.bar(labels, tps, color='steelblue')
ax.set_ylabel('Decode throughput (tokens/s)'); ax.set_xlabel('Refresh schedule')
ax.grid(True, axis='y', alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'throughput.pdf')); plt.close(fig)
