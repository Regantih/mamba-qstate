import sys, json, os, torch
sys.path.insert(0, "experiments/exp1_real_quant")
import run as R
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from datasets import load_dataset
NP = int(os.environ.get("NP", "12"))
BITS = 4
H = 512
REFRESH = [0, 16, 64]
ds = load_dataset("NeelNanda/pile-10k", split="train")
prompts = []
for i in range(NP * 3):
    t = ds[i]["text"]
    if len(t) > 200:
        prompts.append(t)
prompts = prompts[:NP]
print("n_prompts", len(prompts), flush=True)
mid = "state-spaces/mamba2-1.3b"
model = MambaLMHeadModel.from_pretrained(mid, device="cuda", dtype=torch.bfloat16)
model = model.eval()
out = {}
for k in REFRESH:
    le = max(8, H // 8)
    r = R.run(model, prompts, bits=BITS, refresh=k, n_steps=H, log_every=le)
    key = "refresh" + str(k)
    out[key] = r
    kl = r["kl_mean"]
    klt = kl[-1] if kl else float("nan")
    print(key, "b=" + str(round(r["b"], 4)), "r2=" + str(round(r["r2"], 4)), "klt="
 + str(round(klt, 5)), flush=True)
    json.dump(out, open("results/exp1/refresh.json", "w"), indent=2)
print("DONE refresh_sweep", flush=True)
