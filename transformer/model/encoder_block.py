import torch.nn as nn
import torch
from .MultiHeadAttention import MultiHeadAttention
from .FFN import FFN

class EncoderBlock(nn.Module):
    def __init__(self,d_model,n_head,d_ff):
        super().__init__()
        self.mah = MultiHeadAttention(d_model=d_model,num_head=n_head)
        self.ln1  = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model=d_model,d_ff=d_ff)
        self.ln2  = nn.LayerNorm(d_model)

    def forward(self,x,src_pad_mask):
        z = self.ln1(x)
        x = x+self.mah(z,pad_mask=src_pad_mask)

        z = self.ln2(x)
        x = x+self.ffn(z)
        return x