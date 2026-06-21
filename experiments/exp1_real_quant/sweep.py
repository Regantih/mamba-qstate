import sys, json, os, torch
sys.path.insert(0, "experiments/exp1_real_quant")
import run as R
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from datasets import load_dataset
os.makedirs("results/exp1", exist_ok=True)
NP = int(os.environ.get("NP", "24"))
ds = load_dataset("NeelNanda/pile-10k", split="train")
prompts = [ds[i]["text"] for i in range(NP*3) if len(ds[i]["text"]) > 200][:NP]
print("n_prompts", len(prompts), flush=True)
model = MambaLMHeadModel.from_pretrained("state-spaces/mamba2-1.3b", device="cuda", dtype=torch.bfloat16).eval()
HOR = [128, 512]
BITS = [16, 8, 4, 3]
out = {}
for H in HOR:
    le = max(8, H // 8)
    for bits in BITS:
        r = R.run(model, prompts, bits=bits, refresh=0, n_steps=H, log_every=le)
        key = "h%d_bits%d" % (H, bits)
        out[key] = r
        kl_last = r["kl_mean"][-1] if r["kl_mean"] else float("nan")
        print(key, "b=%.3f"%r["b"], "r2=%.3f"%r["r2"], "kl_last=%.4g"%kl_last, flush=True)
        json.dump(out, open("results/exp1/sweep.json", "w"), indent=2)
print("DONE sweep", flush=True)
