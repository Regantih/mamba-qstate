import os, sys, json, time
sys.path.insert(0, "experiments/exp1_real_quant")
import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.generation import InferenceParams
from run import get_ssm_states, DEV
from quant import quant_state_
MID = os.environ.get("MODEL", "state-spaces/mamba2-1.3b")
GEN = int(os.environ.get("GEN", "512"))
WARM = 16
KS = [0, 16, 32, 64]
model = MambaLMHeadModel.from_pretrained(MID, device=DEV, dtype=torch.bfloat16).eval()
print("loaded", MID, flush=True)
@torch.no_grad()
def bench(k):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    ids = torch.randint(0, 50000, (1, 1), device=DEV)
    inf = InferenceParams(max_seqlen=GEN + WARM + 4, max_batch_size=1)
    nxt = ids
    def step(i):
        nonlocal nxt
        lg = model(nxt, inference_params=inf).logits[:, -1]
        if k > 0 and (i % k) != 0:
            for sq in get_ssm_states(model, inf):
                quant_state_(sq, 4)
        inf.seqlen_offset += 1
        nxt = lg.argmax(-1, keepdim=True)
    for i in range(WARM):
        step(i)
    torch.cuda.synchronize(); t0 = time.time()
    for i in range(WARM, WARM + GEN):
        step(i)
    torch.cuda.synchronize(); dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    return GEN / dt, peak
res = {}
base_tps, base_mem = bench(0)
res["baseline_int8_equiv"] = {"tok_per_s": round(base_tps, 2), "peak_gb": round(base_mem, 3)}
print("k=0 tps=%.2f mem=%.3f" % (base_tps, base_mem), flush=True)
for k in [16, 32, 64]:
    tps, mem = bench(k)
    res["k%d" % k] = {"tok_per_s": round(tps, 2), "peak_gb": round(mem, 3), "slowdown_vs_base": round(base_tps / tps, 4)}
    print("k=%d tps=%.2f mem=%.3f slow=%.4f" % (k, tps, mem, base_tps/tps), flush=True)
json.dump(res, open("results/exp1/refresh_throughput.json", "w"), indent=2)
print("DONE refresh_throughput", flush=True)
