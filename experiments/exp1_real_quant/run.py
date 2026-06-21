import sys, json, math
import torch, torch.nn.functional as F
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.generation import InferenceParams
from transformers import AutoTokenizer
sys.path.insert(0, 'experiments/exp1_real_quant')
from quant import quant_state_
DEV='cuda'
_tok=None
def get_tok():
    global _tok
    if _tok is None:
        _tok=AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    return _tok
def get_ssm_states(model, inf):
    out=[]
    for i,_ in enumerate(model.backbone.layers):
        if i in inf.key_value_memory_dict:
            out.append(inf.key_value_memory_dict[i][1])
    return out
def fit_b(steps, kl):
    xs=[(math.log(s), math.log(k)) for s,k in zip(steps,kl) if k>0]
    if len(xs)<2: return float('nan'), float('nan')
    n=len(xs); sx=sum(a for a,_ in xs); sy=sum(b for _,b in xs)
    sxx=sum(a*a for a,_ in xs); sxy=sum(a*b for a,b in xs)
    b=(n*sxy-sx*sy)/(n*sxx-sx*sx); a0=(sy-b*sx)/n
    ybar=sy/n; sst=sum((y-ybar)**2 for _,y in xs); ssr=sum((y-(a0+b*x))**2 for x,y in xs)
    return b, (1-ssr/sst if sst>0 else float("nan"))
@torch.no_grad()
def run(model, prompts, bits, refresh, n_steps, log_every, seed=1337):
    tok=get_tok(); torch.manual_seed(seed)
    kl_accum={}
    for prompt in prompts:
        ids=tok(prompt, return_tensors='pt').input_ids.to(DEV)[:, :8]
        L0=ids.shape[1]
        infF=InferenceParams(max_seqlen=n_steps+L0+2, max_batch_size=1)
        infQ=InferenceParams(max_seqlen=n_steps+L0+2, max_batch_size=1)
        lF=model(ids, inference_params=infF).logits[:, -1]
        model(ids, inference_params=infQ)
        infF.seqlen_offset=L0; infQ.seqlen_offset=L0
        tokF=lF.argmax(-1, keepdim=True); nstep=0
        for t in range(n_steps):
            lF=model(tokF, inference_params=infF).logits[:, -1]
            lQ=model(tokF, inference_params=infQ).logits[:, -1]
            nstep+=1; is_ref=refresh>0 and nstep%refresh==0
            ssF=get_ssm_states(model, infF); ssQ=get_ssm_states(model, infQ)
            for sf,sq in zip(ssF,ssQ):
                if is_ref: sq.copy_(sf)
                else: quant_state_(sq, bits)
            infF.seqlen_offset+=1; infQ.seqlen_offset+=1
            if (t+1)%log_every==0:
                kl=F.kl_div(F.log_softmax(lQ.float(),-1), F.log_softmax(lF.float(),-1), reduction="batchmean", log_target=True).item()
                kl_accum.setdefault(t+1, []).append(kl)
            tokF=lF.argmax(-1, keepdim=True)
    steps=sorted(kl_accum); klm=[sum(kl_accum[s])/len(kl_accum[s]) for s in steps]
    b,r2=fit_b(steps, klm)
    return {"bits":bits,"refresh":refresh,"n_prompts":len(prompts),"steps":steps,"kl_mean":klm,"b":b,"r2":r2}
