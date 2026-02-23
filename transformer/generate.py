import torch
import copy
def greedy_filter(logits):
    next_id = torch.argmax(logits, dim=-1, keepdim=True)
    return next_id


def topk_filter(logits,topk):
    k = min(topk, logits.size(-1))
    topk_vals , _ = torch.topk(logits,k=k,dim=-1)
    kth = topk_vals[:,-1].unsqueeze(-1)
    filter_logits = torch.where(logits<kth,torch.full_like(logits,float('-inf')),logits)
    probs = torch.softmax(filter_logits,dim=-1)
    next_id = torch.multinomial(probs,num_samples=1)
    return next_id

def topp_filter(logits,topp):
    sorted_logits,sorted_indices = torch.sort(logits,dim=-1,descending=True)
    sorted_probs = torch.softmax(sorted_logits,dim=-1)

    cumulative_probs = torch.cumsum(sorted_probs,dim=-1)
    remove = cumulative_probs > topp
    remove[...,1:]=remove[...,:-1].clone()
    remove[...,0] =False

    sorted_probs = sorted_probs.masked_fill(remove,0.0)
    den = sorted_probs.sum(dim=-1,keepdim=True).clamp_min(1e-12)
    normalized_probs = sorted_probs/den
    next_sorted = torch.multinomial(normalized_probs,num_samples=1)
    next_id = torch.gather(sorted_indices,-1,next_sorted)
    return next_id

def apply_penalty(tgt, logits, penalty=1.2, ignore_ids=None):
    # tgt: [B, T], logits: [B, V]
    assert penalty >= 1.0
    B, V = logits.shape
    out = logits.clone()

    if ignore_ids is None:
        ignore_ids = set()

    for b in range(B):
        seen = set(tgt[b].tolist())
        seen = [i for i in seen if 0 <= i < V and i not in ignore_ids]
        if not seen:
            continue

        idx = torch.tensor(seen, device=logits.device, dtype=torch.long)
        vals = out[b, idx]
        vals = torch.where(vals > 0, vals / penalty, vals * penalty)
        out[b, idx] = vals

    return out


@torch.no_grad()
def generate(model,src_ids,bos_id,eos_id,prefix_ids=None,max_length_tokens=256,temperature=1,decode_mode="greedy",topk=0,topp=1,beam_size=1,length_penalty=1.0,use_cache=False):
    assert temperature > 0
    assert isinstance(topk, int) and topk >= 0
    assert 0 < topp <=1
    assert decode_mode in {"greedy", "sample", "beam"}
    if decode_mode == "beam":
        return beam_search_generate(model,src_ids,bos_id,eos_id,prefix_ids=prefix_ids,
                                    max_length_tokens=max_length_tokens,beam_size=beam_size,length_penalty=length_penalty,use_cache=use_cache)
    model.eval()
    device = src_ids.device
    B = src_ids.size(0)
    bos = torch.full((B,1),bos_id,dtype=torch.long,device=device)
    tgt = bos if prefix_ids is None else torch.concat([bos,prefix_ids],dim=1)
    decode_input = tgt
    memory,src_pad_mask = model.encode(src_ids)
    cross_key_values = model.precompute_cross_kv(memory) if use_cache else None
    past_key_values = None
    pe = 0
    for _ in range(max_length_tokens):
        if use_cache:
            logits,past_key_values = model.decode(decode_input,memory,src_pad_mask,past_key_values=past_key_values,use_cache=True,position_offset=pe,cross_key_values=cross_key_values)
        else:
            logits = model.decode(tgt,memory,src_pad_mask,past_key_values=None,use_cache=False)
        logits = logits[:,-1,:]
        logits = apply_penalty(tgt, logits, penalty=1.2, ignore_ids={bos_id, eos_id})
        if decode_mode == "greedy":
            next_id = greedy_filter(logits)
        elif decode_mode == "sample":
            logits = logits/temperature
            if topk > 0 :
                next_id = topk_filter(logits,topk)
            elif topp <1.0:
                next_id = topp_filter(logits,topp)
            else:
                probs = torch.softmax(logits,dim=-1)
                next_id = torch.multinomial(probs,num_samples=1)
        else:
            raise ValueError("Wrong")
        tgt = torch.concat([tgt,next_id],dim=1)
        if use_cache:
            decode_input = next_id
            pe = tgt.size(1)-1
        if (next_id == eos_id).all():
            break
    return tgt


def beam_search_generate(model,src_ids,bos_id,eos_id,prefix_ids=None,
                                    max_length_tokens=256,beam_size=1,length_penalty=1.0,use_cache=False,):
    model.eval()
    device = src_ids.device
    bos = torch.full((1,1),bos_id,dtype=torch.long,device=device)
    init = bos if prefix_ids is None else torch.concat([bos,prefix_ids],dim=1)
    new_past_key_values = None
    memory,src_pad_mask = model.encode(src_ids)
    cross_key_values = model.precompute_cross_kv(memory) if use_cache else None
    beams = [(init,0.0,False,new_past_key_values)]
    pe = 0
    def rank_key(x):
        seq,sc,_,_ =x
        L = seq.size(1)
        return sc/(L**length_penalty) if length_penalty > 0 else sc
    for _ in range(max_length_tokens):
        candidates = []
        all_finished = True
        for seq,score,finished,past_key_values in beams:
            if finished:
                candidates.append((seq,score,True,past_key_values))
                continue
            all_finished = False
            if use_cache:
                if past_key_values is None:
                    decode_input = seq
                    pe = 0
                else:
                    decode_input = seq[:,-1:]
                    pe = seq.size(1)-1
                logits,new_past_key_values = model.decode(decode_input,memory,src_pad_mask,past_key_values=past_key_values,use_cache=True,position_offset=pe,cross_key_values=cross_key_values)
            else:
                logits = model.decode(seq,memory,src_pad_mask,past_key_values=None,use_cache=False)
                new_past_key_values = None
            logits = logits[:,-1,:]
            logits = apply_penalty(seq,logits,penalty=1.2, ignore_ids={bos_id, eos_id})
            log_probs = torch.log_softmax(logits,dim=-1)
            k = min(beam_size,log_probs.size(-1))
            topv,topi = torch.topk(log_probs,k=k,dim=-1)
            for j in range(k):
                tok = topi[0,j].view(1,1)
                new_seq = torch.cat([seq,tok],dim=1)
                new_score = score+topv[0,j].item()
                new_finished = (tok.item()==eos_id)
                candidate_past = candidate_past = [(k, v) for (k, v) in new_past_key_values] if new_past_key_values is not None else None
                candidates.append((new_seq,new_score,new_finished,candidate_past))
        if all_finished:
            break
        candidates.sort(key=rank_key,reverse=True)
        beams = candidates[:beam_size]
    best_seq = max(beams, key=rank_key)[0]
    return best_seq
