import sys, json, os, random
sys.path.insert(0, "experiments/exp1_real_quant")
import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.generation import InferenceParams
from run import get_tok, get_ssm_states, DEV
from quant import quant_state_
from datasets import load_dataset
MID = os.environ.get("MODEL", "state-spaces/mamba2-1.3b")
NTRIAL = int(os.environ.get("NTRIAL", "20"))
DEPTH = int(os.environ.get("DEPTH", "700"))
GEN = int(os.environ.get("GEN", "12"))
BITS = [16, 8, 4, 3]
SEED = 1337
tok = get_tok()
model = MambaLMHeadModel.from_pretrained(MID, device=DEV, dtype=torch.bfloat16).eval()
ds = load_dataset("NeelNanda/pile-10k", split="train")
filler = [ds[i]["text"] for i in range(300) if len(ds[i]["text"]) > 400][:NTRIAL+5]
print("loaded", MID, "trials", NTRIAL, "depth", DEPTH, flush=True)
def build(trial):
    rng = random.Random(SEED + trial)
    code = "".join(rng.choice("0123456789") for _ in range(4))
    needle = "The secret access code is " + code + ". Remember it."
    body = filler[trial % len(filler)]
    ctx = needle + " " + body
    ids = tok(ctx, return_tensors="pt").input_ids.to(DEV)[:, :DEPTH]
    q = tok(" What is the secret access code? The secret access code is", return_tensors="pt").input_ids.to(DEV)
    full = torch.cat([ids, q], dim=1)
    return code, full
@torch.no_grad()
def recall(bits):
    hits = 0
    for tr in range(NTRIAL):
        code, ids = build(tr)
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
        torch.cuda.empty_cache()
    return hits
res = {}
for b in BITS:
    h = recall(b)
    acc = round(h / NTRIAL, 4)
    res["bits%d" % b] = {"hits": h, "trials": NTRIAL, "acc": acc}
    print("bits%d" % b, "hits=%d/%d" % (h, NTRIAL), "acc=%.4f" % acc, flush=True)
out = {"config": {"model": MID, "depth": DEPTH, "ntrial": NTRIAL, "gen": GEN, "seed": SEED}, "recall": res}
json.dump(out, open("/workspace/mamba-qstate/results/exp1/quality.json", "w"), indent=2)
print("DONE niah", flush=True)
