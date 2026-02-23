import torch.nn as nn
import torch
import math
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor,start_offset=0) -> torch.Tensor:
        # x: (B, T, d_model)
        T = x.size(1)
        pe_slice = self.pe[start_offset:start_offset + T].unsqueeze(0)
        return x + pe_slice.to(device=x.device,dtype=x.dtype)
    

