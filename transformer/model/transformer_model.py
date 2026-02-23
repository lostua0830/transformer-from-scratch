from .decoder import Decoder
from .encoder import Encoder
from .pe import PositionalEncoding
import torch.nn as nn
import torch

class Transformer(nn.Module):
    def __init__(self,N,src_vocab,tgt_vocab,d_model,d_ff,num_head,pad_id,max_len=5000):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab,d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab,d_model)
        self.pe        = PositionalEncoding(d_model=d_model,max_len=max_len)
        self.En        = Encoder(num_blocks=N,d_model=d_model,d_ff=d_ff,num_heads=num_head)
        self.De        = Decoder(num_blocks=N,d_model=d_model,d_ff=d_ff,num_heads=num_head)
        self.lm_head   = nn.Linear(d_model,tgt_vocab,bias=False)
        self.pad_id  = pad_id

    def forward(self,src_ids,tgt_ids,use_cache=False):
        memory,src_pad_mask = self.encode(src_ids)
        out = self.decode(tgt_ids,memory,src_pad_mask,use_cache=False)
        return out
    
    def encode(self,src_ids):
        src_pad_mask = (src_ids==self.pad_id)
        src = self.src_embed(src_ids)
        src = self.pe(src)
        memory = self.En(src,src_pad_mask)
        return memory,src_pad_mask
    
    def decode(self,tgt_ids,memory,src_pad_mask,past_key_values=None,use_cache=False,cross_key_values=None,position_offset=0):
        tgt_pad_mask = (tgt_ids == self.pad_id)
        tgt = self.tgt_embed(tgt_ids)
        tgt = self.pe(tgt,start_offset=position_offset)
        if use_cache:
            tgt_out,new_past_key_values = self.De(tgt,memory,src_pad_mask,tgt_pad_mask,past_key_values,use_cache,cross_key_values)
            return self.lm_head(tgt_out),new_past_key_values
        tgt_out = self.De(tgt,memory,src_pad_mask,tgt_pad_mask,past_key_values,use_cache,cross_key_values)
        return self.lm_head(tgt_out)
    
    def precompute_cross_kv(self,memory):
        return self.De.precompute_cross_kv(memory)