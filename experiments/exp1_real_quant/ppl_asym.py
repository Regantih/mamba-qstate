import sys, json, os, math, torch, torch.nn.functional as F
sys.path.insert(0, "experiments/exp1_real_quant")
from quant import quant_state_, quant_state_asym_
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.generation import InferenceParams
from transformers import AutoTokenizer
from datasets import load_dataset
tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
import os as _os
MODEL=_os.environ.get("MODEL","state-spaces/mamba2-1.3b")
TAG=_os.environ.get("TAG",MODEL.split("/")[-1].replace(".",""))
OUT="results/exp1/ppl_asym.json"
print("loading",MODEL,flush=True)
m = MambaLMHeadModel.from_pretrained(MODEL, device="cuda", dtype=torch.bfloat16).eval()
ds = load_dataset("NeelNanda/pile-10k", split="train")
NP = int(os.environ.get("NP","24")); SEQ = int(os.environ.get("SEQ","512"))
texts = [ds[i]["text"] for i in range(NP*3) if len(ds[i]["text"])>800][:NP]
@torch.no_grad()
def ppl_for(bits, qfn):
    tot_nll=0.0; tot_tok=0
    for txt in texts:
        ids = tok(txt, return_tensors="pt").input_ids.to("cuda")[:, :SEQ]
        if ids.shape[1] < 16: continue
        inf = InferenceParams(max_seqlen=ids.shape[1]+2, max_batch_size=1)
        logit = m(ids[:, :1], inference_params=inf).logits[:, -1]
        inf.seqlen_offset = 1
        for t in range(1, ids.shape[1]):
            lp = F.log_softmax(logit.float(), -1)
            tgt = ids[0, t].item(); tot_nll += -lp[0, tgt].item(); tot_tok += 1
            cur = ids[:, t:t+1]
            logit = m(cur, inference_params=inf).logits[:, -1]
            for i,_ in enumerate(m.backbone.layers):
                if i in inf.key_value_memory_dict: qfn(inf.key_value_memory_dict[i][1], bits)
            inf.seqlen_offset += 1
    return math.exp(tot_nll/tot_tok)
res={}
res = {"symmetric": {}, "asymmetric": {}}
for bits in [16, 8, 4, 3]:
    ps = ppl_for(bits, quant_state_)
    res["symmetric"]["bits%d" % bits] = round(ps, 4)
    print("sym bits", bits, "ppl=%.4f" % ps, flush=True)
    torch.cuda.empty_cache()
    pa = ppl_for(bits, quant_state_asym_)
    res["asymmetric"]["bits%d" % bits] = round(pa, 4)
    print("asym bits", bits, "ppl=%.4f" % pa, flush=True)
    torch.cuda.empty_cache()
json.dump(res, open(OUT, "w"), indent=2)
print("DONE ppl_asym", flush=True)
