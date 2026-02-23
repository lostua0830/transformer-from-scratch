from .decoder_block import DecoderBlock
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self,num_blocks,d_model,d_ff,num_heads):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(d_model=d_model,d_ff=d_ff,num_head=num_heads) for _ in range(num_blocks)
        ])
    def forward(self,x,memory,src_pad_mask,tgt_pad_mask,past_key_values=None,use_cache=False,cross_key_values=None):
        new_past_key_values = [] if use_cache else None
        for i,layer in enumerate(self.layers):
            if use_cache:
                if past_key_values is None:
                    past_k,past_v = None,None
                else:
                    past_k,past_v = past_key_values[i]
                if cross_key_values is None:
                    cross_k,cross_v = None,None
                else:
                    cross_k,cross_v = cross_key_values[i]
                x,new_k,new_v = layer(x,memory,src_pad_mask,tgt_pad_mask,past_k,past_v,use_cache=True,cross_k=cross_k,cross_v=cross_v)
                new_past_key_values.append((new_k,new_v))
            else:
                x = layer(x,memory,src_pad_mask,tgt_pad_mask)
        if use_cache:
            return x,new_past_key_values
        return x
    
    def precompute_cross_kv(self, memory):
        out = []
        for layer in self.layers:
            k = layer.mha2.split_heads(layer.mha2.Wk(memory))
            v = layer.mha2.split_heads(layer.mha2.Wv(memory))
            out.append((k, v))
        return out
    

