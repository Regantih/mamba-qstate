import sys, json, yaml, argparse
import numpy as np, torch
sys.path.insert(0, 'experiments/exp0_noise_injection')
import run_exp0_quant as R
from qstate.metrics import kl_divergence, fit_growth_exponent
ap = argparse.ArgumentParser()
ap.add_argument('--config', required=True)
ap.add_argument('--bits', type=int, default=8)
ap.add_argument('--refresh', type=int, default=0)
ap.add_argument('--out', required=True)
args = ap.parse_args()
cfg = yaml.safe_load(open(args.config))
R.set_seed(cfg['seed'])
device = 'cuda'
model, tok = R._load_model_and_tok(cfg['model'], device)
prompts = R._get_prompts(cfg, tok, device)
max_new = cfg['max_new_tokens']
trunc = cfg.get('prompt_trunc', 64)
prompt_batch = torch.cat([p[:, -trunc:] for p in prompts], dim=0)
max_seqlen = trunc + max_new + 1
log_every = cfg['log_every']
chosen, fp_logits = R._fp_reference(model, prompt_batch, max_new, max_seqlen, log_every)
steps = [t + 1 for t in range(max_new) if (t + 1) % log_every == 0]
pf = R._policy_factory(args.bits, args.refresh, cfg['seed'], cfg.get('quant_granularity','per_head'), cfg.get('quant_symmetric',True))
pol_logits = R._policy_run(model, prompt_batch, chosen, pf, max_seqlen, log_every)
B = prompt_batch.shape[0]
perprompt = []
for j in range(B):
    kl_trace = [kl_divergence(fp_logits[i][j], pol_logits[i][j]) for i in range(len(steps))]
    fit = fit_growth_exponent(steps, kl_trace)
    perprompt.append({'prompt': j, 'kl_final': kl_trace[-1], 'fit': fit})
bs = np.array([p['fit']['exponent_b'] for p in perprompt], dtype=float)
finals = np.array([p['kl_final'] for p in perprompt], dtype=float)
valid = bs[~np.isnan(bs)]
summary = {'bits': args.bits, 'refresh': args.refresh, 'n_prompts': B,
  'b_mean': float(np.mean(valid)), 'b_median': float(np.median(valid)),
  'b_std': float(np.std(valid)), 'b_min': float(np.min(valid)), 'b_max': float(np.max(valid)),
  'n_superlinear': int((valid > 1.0).sum()), 'n_sublinear': int((valid <= 1.0).sum()),
  'klfinal_mean': float(np.mean(finals)), 'klfinal_median': float(np.median(finals)),
  'klfinal_max': float(np.max(finals)), 'klfinal_min': float(np.min(finals))}
json.dump({'summary': summary, 'per_prompt': perprompt}, open(args.out, 'w'), indent=2)
print('=== per-prompt diagnostic bits%d refresh%d ===' % (args.bits, args.refresh))
for k, v in summary.items():
    print('  %s: %s' % (k, v))
print('per-prompt b:', [round(float(x),2) for x in bs])
