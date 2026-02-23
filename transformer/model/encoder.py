from .encoder_block import EncoderBlock
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self,num_blocks,d_model,d_ff,num_heads):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model=d_model,n_head=num_heads,d_ff=d_ff) for _ in range(num_blocks)
        ])
    def forward(self,x,src_pad_mask):
        for layer in self.layers:
            x = layer(x,src_pad_mask)
        return x



