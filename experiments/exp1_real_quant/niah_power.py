import os, sys, json, math, random
sys.path.insert(0, "experiments/exp1_real_quant")
import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.generation import InferenceParams
from run import get_tok, get_ssm_states, DEV
from quant import quant_state_
from datasets import load_dataset
MID = os.environ.get("MODEL", "state-spaces/mamba2-1.3b")
NTRIAL = int(os.environ.get("NTRIAL", "200"))
GEN = int(os.environ.get("GEN", "12"))
CTXLEN = int(os.environ.get("CTXLEN", "900"))
BITS = [16, 8, 4, 3]
SEEDS = [1337, 2024, 7]
DEPTHS = [0.25, 0.75]
tok = get_tok()
model = MambaLMHeadModel.from_pretrained(MID, device=DEV, dtype=torch.bfloat16).eval()
ds = load_dataset("NeelNanda/pile-10k", split="train")
filler = [ds[i]["text"] for i in range(2000) if len(ds[i]["text"]) > 400]
print("loaded", MID, "ntrial", NTRIAL, "seeds", SEEDS, "depths", DEPTHS, flush=True)
def build(seed, trial, depth_frac):
    rng = random.Random(seed * 100003 + trial)
    code = "".join(rng.choice("0123456789") for _ in range(4))
    needle = " The secret access code is " + code + ". Remember it. "
    body = filler[(seed + trial) % len(filler)]
    btok = tok(body, return_tensors="pt").input_ids[0][:CTXLEN]
    pos = int(len(btok) * depth_frac)
    ntok = tok(needle, return_tensors="pt").input_ids[0]
    ids = torch.cat([btok[:pos], ntok, btok[pos:]]).unsqueeze(0).to(DEV)
    q = tok(" What is the secret access code? The secret access code is", return_tensors="pt").input_ids.to(DEV)
    return code, torch.cat([ids, q], dim=1)
def wilson(hits, n):
    if n == 0: return (0.0, 0.0, 0.0)
    z = 1.96; phat = hits / n
    denom = 1 + z*z/n
    center = (phat + z*z/(2*n)) / denom
    half = (z * math.sqrt(phat*(1-phat)/n + z*z/(4*n*n))) / denom
    return (phat, max(0.0, center-half), min(1.0, center+half))
@torch.no_grad()
def recall(bits):
    hits = 0; n = 0
    for seed in SEEDS:
        for df in DEPTHS:
            for tr in range(NTRIAL):
                code, ids = build(seed, tr, df)
                L = ids.shape[1]
                inf = InferenceParams(max_seqlen=L + GEN + 2, max_batch_size=1)
                model(ids[:, :1], inference_params=inf)
                inf.seqlen_offset = 1
                logit = None
                for t in range(1, L):
                    logit = model(ids[:, t:t+1], inference_params=inf).logits[:, -1]
                    for sq in get_ssm_states(model, inf):
                        quant_state_(sq, bits)
                    inf.seqlen_offset += 1
                nxt = logit.argmax(-1, keepdim=True)
                out = []
                for g in range(GEN):
                    out.append(nxt.item())
                    logit = model(nxt, inference_params=inf).logits[:, -1]
                    for sq in get_ssm_states(model, inf):
                        quant_state_(sq, bits)
                    inf.seqlen_offset += 1
                    nxt = logit.argmax(-1, keepdim=True)
                if code in tok.decode(out):
                    hits += 1
                n += 1
                torch.cuda.empty_cache()
    return hits, n
res = {}
for b in BITS:
    h, n = recall(b)
    acc, lo, hi = wilson(h, n)
    res["bits%d" % b] = {"hits": h, "trials": n, "acc": round(acc, 4), "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4)}
    print("bits=%d acc=%.4f [%.4f, %.4f] n=%d" % (b, acc, lo, hi, n), flush=True)
out = {"config": {"model": MID, "ctxlen": CTXLEN, "ntrial_per_cell": NTRIAL, "seeds": SEEDS, "depths": DEPTHS, "gen": GEN}, "recall": res}
json.dump(out, open("results/exp1/quality_power.json", "w"), indent=2)
print("DONE niah_power", flush=True)
