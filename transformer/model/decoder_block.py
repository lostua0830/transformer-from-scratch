from .MultiHeadAttention import MultiHeadAttention
from .FFN import FFN
import torch.nn as nn
import torch

class DecoderBlock(nn.Module):
    def __init__(self,d_model,d_ff,num_head):
        super().__init__()
        self.mha1 = MultiHeadAttention(d_model=d_model,num_head=num_head)
        self.mha2 = MultiHeadAttention(d_model=d_model,num_head=num_head)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model=d_model,d_ff=d_ff)
    
    def forward(self,x,memory,src_pad_mask,target_pad_mask,past_k=None,past_v=None,use_cache=False,cross_k=None,cross_v=None):
        z = self.ln1(x)
        if use_cache:
            attn_out, new_k, new_v = self.mha1(
            z, causal=True, pad_mask=target_pad_mask,
            past_k=past_k, past_v=past_v, use_cache=True
        )
        else:
            attn_out = self.mha1(
            z, causal=True, pad_mask=target_pad_mask,
            past_k=None, past_v=None, use_cache=False
        )
            new_k, new_v = None, None
        x = x+attn_out
        z = self.ln2(x)
        x = x + self.mha2(z,memory,pad_mask=src_pad_mask,memory_k=cross_k,memory_v=cross_v)

        z = self.ln3(x)
        x = x+self.ffn(z)
        if use_cache:
            return x,new_k,new_v
        return x
        

